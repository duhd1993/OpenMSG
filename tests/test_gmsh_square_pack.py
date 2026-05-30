"""Tests for gmsh-based square-pack fiber composite mesh and homogenization.

Uses the gmsh Python API to create a circular fiber (disk) in a unit square
matrix cross-section. The geometry is fragmented so the fiber/matrix interface
is an exact circle, then meshed with a periodic Tri3 mesh so that opposite
edges have matching nodes (required by OpenMSG's periodic constraint engine).

The 2D cross-section serves as the Structure Genome with active_axes=(x, y);
the fiber direction is Z (out of plane). The suite is skipped when gmsh is not
installed.
"""
from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np

from openmsg.homogenize import homogenize_msg
from openmsg.materials import engineering_constants_from_stiffness, isotropic_stiffness
from openmsg.mesh import SolidMesh

_GMSH_AVAILABLE = importlib.util.find_spec("gmsh") is not None


def _gmsh_square_pack_mesh(
    *,
    fiber_volume_fraction: float,
    mesh_size: float,
) -> SolidMesh:
    """Return a periodic Tri3 SolidMesh for a circular fiber in a square unit cell.

    The unit cell is [0,1]^2 in XY (z=0 for all nodes). The fiber is a circle
    of radius r = sqrt(vf/pi) centered at (0.5, 0.5). Opposite edges are made
    periodic in gmsh before meshing so that OpenMSG's periodic_constraints can
    find matching node pairs.
    """
    import gmsh

    radius = math.sqrt(fiber_volume_fraction / math.pi)
    tol = 1e-4

    gmsh.initialize()
    try:
        gmsh.option.set_number("General.Verbosity", 0)
        gmsh.model.add("square_pack")

        # Rectangle and disk; fragment to share the circular interface curve.
        gmsh.model.occ.add_rectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.add_disk(0.5, 0.5, 0, radius, radius)
        gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
        gmsh.model.occ.synchronize()

        # Identify surfaces by area: the smaller one is the fiber disk.
        surfs = [tag for _, tag in gmsh.model.get_entities(2)]
        by_area = sorted((gmsh.model.occ.get_mass(2, s), s) for s in surfs)
        gmsh.model.add_physical_group(2, [by_area[0][1]], tag=1, name="fiber")
        gmsh.model.add_physical_group(2, [by_area[1][1]], tag=2, name="matrix")

        # Locate the four outer boundary curves via bounding-box queries.
        # The fiber circle doesn't reach the boundary (radius < 0.5), so each
        # side is a single intact straight curve after the fragment operation.
        def _boundary_curves(xlo: float, ylo: float, xhi: float, yhi: float) -> list[int]:
            return [t for _, t in gmsh.model.get_entities_in_bounding_box(
                xlo, ylo, -tol, xhi, yhi, tol, 1)]

        left   = _boundary_curves(-tol,     -tol,     tol,     1.0 + tol)
        right  = _boundary_curves(1.0 - tol, -tol,    1.0 + tol, 1.0 + tol)
        bottom = _boundary_curves(-tol,     -tol,     1.0 + tol, tol)
        top    = _boundary_curves(-tol,     1.0 - tol, 1.0 + tol, 1.0 + tol)

        # Row-major 4×4 affine transforms: right = left + (1,0,0), top = bottom + (0,1,0).
        tx = [1, 0, 0, 1,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
        ty = [1, 0, 0, 0,  0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 1]
        if right and left:
            gmsh.model.mesh.set_periodic(1, right, left, tx)
        if top and bottom:
            gmsh.model.mesh.set_periodic(1, top, bottom, ty)

        gmsh.option.set_number("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.set_number("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)

        # Pull out nodal coordinates (gmsh always gives 3D coords; z = 0 here).
        node_tags, coords, _ = gmsh.model.mesh.get_nodes()
        node_map = {int(t): i for i, t in enumerate(node_tags)}
        nodes = np.asarray(coords, dtype=float).reshape(-1, 3)

        # Collect Tri3 elements per physical group in insertion order.
        elements: list[list[int]] = []
        material_ids: list[str] = []
        for phys_tag, mat_name in [(1, "fiber"), (2, "matrix")]:
            for ent in gmsh.model.get_entities_for_physical_group(2, phys_tag):
                etypes, _, ntags_list = gmsh.model.mesh.get_elements(2, int(ent))
                for etype, ntags in zip(etypes, ntags_list):
                    if etype != 2:  # gmsh element type 2 = 3-node triangle
                        continue
                    for row in np.asarray(ntags, dtype=int).reshape(-1, 3):
                        elements.append([node_map[int(n)] for n in row])
                        material_ids.append(mat_name)

    finally:
        gmsh.finalize()

    return SolidMesh(
        nodes=nodes,
        elements=np.asarray(elements, dtype=int),
        material_ids=tuple(material_ids),
        element_type="tri3",
        active_axes=("x", "y"),
    )


def _extrude_tri3_to_tet4(mesh_2d: SolidMesh, lz: float = 1.0) -> SolidMesh:
    """Extrude a 2D Tri3 mesh in Z to produce an exactly equivalent 3D Tet4 mesh.

    Each 2D triangle [b0, b1, b2] becomes a triangular prism by copying its
    nodes to z = lz, then the prism is split into 3 tetrahedra.  The XY
    cross-section of the resulting 3D mesh is *identical* to the 2D mesh —
    same node coordinates, same element connectivity projected onto z = 0.

    Prism → 3 Tet decomposition (with t_i = b_i + n, n = n_2d_nodes):
        Tet 1: [b0, b1, b2, t2]
        Tet 2: [b0, t1, b1, t2]
        Tet 3: [b0, t0, t1, t2]

    For a counter-clockwise bottom triangle the Jacobian determinant of each
    tet equals lz * 2 * area_triangle > 0, satisfying OpenMSG's tet4 element
    requirement.  CW triangles are detected from the signed area and their
    node ordering is flipped before decomposition.
    """
    assert mesh_2d.element_type == "tri3", "source mesh must be tri3"

    n = mesh_2d.n_nodes
    nodes_top = mesh_2d.nodes.copy()
    nodes_top[:, 2] = lz
    nodes_3d = np.vstack([mesh_2d.nodes, nodes_top])   # bottom then top

    elements_3d: list[list[int]] = []
    material_ids_3d: list[str] = []

    for (b0, b1, b2), mat in zip(mesh_2d.elements, mesh_2d.material_ids):
        # Signed area of the bottom triangle in the XY plane
        xy = mesh_2d.nodes[[b0, b1, b2], :2]
        signed_area = 0.5 * (
            (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
            - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        )
        if signed_area < 0:      # CW → swap b1↔b2 to make CCW
            b1, b2 = b2, b1

        t0, t1, t2 = b0 + n, b1 + n, b2 + n

        # Three tets, each with det(J) = lz * 2 * |area| > 0
        elements_3d.append([b0, b1, b2, t2])
        elements_3d.append([b0, t1, b1, t2])
        elements_3d.append([b0, t0, t1, t2])
        material_ids_3d += [mat, mat, mat]

    return SolidMesh(
        nodes=nodes_3d,
        elements=np.asarray(elements_3d, dtype=int),
        material_ids=tuple(material_ids_3d),
        element_type="tet4",
    )


def _triangle_areas(mesh: SolidMesh) -> np.ndarray:
    """Return the signed area of each Tri3 element (always positive for CCW)."""
    p = mesh.nodes[mesh.elements]           # (n_elem, 3, 3)
    v0 = p[:, 1] - p[:, 0]
    v1 = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0])


@unittest.skipUnless(_GMSH_AVAILABLE, "gmsh is not installed")
class GmshSquarePackTests(unittest.TestCase):
    """Homogenization of a circular fiber composite using a gmsh Tri3 mesh."""

    FIBER_E: float = 70.0
    FIBER_NU: float = 0.22
    MATRIX_E: float = 3.5
    MATRIX_NU: float = 0.35
    TARGET_VF: float = 0.35

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = _gmsh_square_pack_mesh(
            fiber_volume_fraction=cls.TARGET_VF,
            mesh_size=0.08,
        )
        cls.result = homogenize_msg(
            mesh=cls.mesh,
            material_stiffness={
                "fiber": isotropic_stiffness(cls.FIBER_E, cls.FIBER_NU),
                "matrix": isotropic_stiffness(cls.MATRIX_E, cls.MATRIX_NU),
            },
            macro_model="cauchy_3d",
            # Periodic in X and Y (the active SG axes); Z is the fiber axis
            # (out of XY plane) and carries no periodic constraint for a 2D SG.
            constraints=[
                {"type": "periodic", "axes": ["x", "y", "z"]},
                {"type": "mean_zero"},
            ],
        )
        cls.Dbar = cls.result.Dbar.detach().cpu().numpy()
        cls.constants = engineering_constants_from_stiffness(cls.Dbar)

        # Area-weighted fiber volume fraction realized in the discrete mesh.
        areas = _triangle_areas(cls.mesh)
        fiber_mask = np.array([m == "fiber" for m in cls.mesh.material_ids])
        cls.realized_vf = float(areas[fiber_mask].sum() / areas.sum())

    # ── Mechanical well-posedness ─────────────────────────────────────────────

    def test_stiffness_is_symmetric(self) -> None:
        np.testing.assert_allclose(self.Dbar, self.Dbar.T, rtol=1e-8, atol=1e-8)

    def test_stiffness_is_positive_definite(self) -> None:
        min_eig = float(np.min(np.linalg.eigvalsh(self.Dbar)))
        self.assertGreater(min_eig, 0.0, f"smallest eigenvalue = {min_eig:.3e}")

    # ── Physics: fiber direction must be stiffer ──────────────────────────────

    def test_fiber_direction_stiffer_than_transverse(self) -> None:
        """C_33 (fiber / Z direction) must exceed C_11 and C_22 (transverse X, Y)."""
        self.assertGreater(self.Dbar[2, 2], self.Dbar[0, 0])
        self.assertGreater(self.Dbar[2, 2], self.Dbar[1, 1])

    # ── Physics: square unit cell → transverse isotropy in X and Y ───────────

    def test_transverse_normal_stiffness_symmetry(self) -> None:
        """C_11 ≈ C_22: square packing makes X and Y statistically equivalent."""
        rel = abs(self.Dbar[0, 0] - self.Dbar[1, 1]) / self.Dbar[0, 0]
        self.assertLess(rel, 0.05, f"C_11={self.Dbar[0,0]:.4f}, C_22={self.Dbar[1,1]:.4f}, rel={rel:.1%}")

    def test_transverse_shear_stiffness_symmetry(self) -> None:
        """C_44 ≈ C_55: shear in the YZ and XZ planes must also match."""
        rel = abs(self.Dbar[3, 3] - self.Dbar[4, 4]) / self.Dbar[3, 3]
        self.assertLess(rel, 0.05, f"C_44={self.Dbar[3,3]:.4f}, C_55={self.Dbar[4,4]:.4f}, rel={rel:.1%}")

    # ── Quantitative bounds ───────────────────────────────────────────────────

    def test_longitudinal_modulus_near_rule_of_mixtures(self) -> None:
        """E_z should match the fiber-direction rule of mixtures within 5%.

        For a 2D periodic SG, the fiber-direction strain mode requires no
        fluctuation field, so the MSG result equals the Voigt average exactly.
        The comparison uses the mesh-realized vf (area-weighted).
        """
        vf = self.realized_vf
        E_z_rom = vf * self.FIBER_E + (1.0 - vf) * self.MATRIX_E
        rel = abs(self.constants["E_z"] - E_z_rom) / E_z_rom
        self.assertLess(
            rel, 0.05,
            f"E_z = {self.constants['E_z']:.3f}, ROM = {E_z_rom:.3f}, diff = {rel:.1%}",
        )

    def test_transverse_modulus_between_reuss_and_voigt(self) -> None:
        """E_x must lie between the Reuss lower bound and the Voigt upper bound.

        A 10% slack is allowed on each side to accommodate coarse-mesh error
        and the fact that bounds use the same vf as the longitudinal test.
        """
        vf = self.realized_vf
        voigt = vf * self.FIBER_E + (1.0 - vf) * self.MATRIX_E
        reuss = 1.0 / (vf / self.FIBER_E + (1.0 - vf) / self.MATRIX_E)
        E_x = self.constants["E_x"]
        self.assertGreaterEqual(E_x, reuss * 0.90,
                                f"E_x={E_x:.3f} below 90% of Reuss={reuss:.3f}")
        self.assertLessEqual(E_x, voigt * 1.10,
                             f"E_x={E_x:.3f} above 110% of Voigt={voigt:.3f}")


@unittest.skipUnless(_GMSH_AVAILABLE, "gmsh is not installed")
class GmshSquarePack3DConsistencyTests(unittest.TestCase):
    """3D Tet4 SG (extruded from the 2D mesh) must give identical Dbar to the 2D Tri3 SG.

    The 3D mesh is built by calling ``_extrude_tri3_to_tet4`` on the same 2D
    gmsh mesh, so the XY cross-section is *identical* — same node coordinates,
    same element footprints.  Each triangle becomes a prism split into 3 tets.

    For a uniform-in-Z geometry with z-periodic boundary conditions:
      * The exact fluctuation field is z-independent.
      * Linear tet4 elements can represent z-independent functions exactly.
      * The z-periodic BC (enforced by ``periodic_constraints``) forces the
        z=0 and z=lz DOFs to be equal, reducing the 3D problem to the 2D one.

    Therefore Dbar_3d == Dbar_2d up to floating-point round-off from the
    different assembly code paths.  The tolerance is 1e-5 relative, far tighter
    than the 3% tolerance accepted for independently generated meshes.
    """

    FIBER_E: float = GmshSquarePackTests.FIBER_E
    FIBER_NU: float = GmshSquarePackTests.FIBER_NU
    MATRIX_E: float = GmshSquarePackTests.MATRIX_E
    MATRIX_NU: float = GmshSquarePackTests.MATRIX_NU
    TARGET_VF: float = GmshSquarePackTests.TARGET_VF
    MESH_SIZE: float = 0.10

    @classmethod
    def setUpClass(cls) -> None:
        material_stiffness = {
            "fiber": isotropic_stiffness(cls.FIBER_E, cls.FIBER_NU),
            "matrix": isotropic_stiffness(cls.MATRIX_E, cls.MATRIX_NU),
        }
        constraints = [
            {"type": "periodic", "axes": ["x", "y", "z"]},
            {"type": "mean_zero"},
        ]

        # 2D Tri3 SG — the reference
        cls.mesh_2d = _gmsh_square_pack_mesh(
            fiber_volume_fraction=cls.TARGET_VF,
            mesh_size=cls.MESH_SIZE,
        )
        cls.Dbar_2d = homogenize_msg(
            mesh=cls.mesh_2d,
            material_stiffness=material_stiffness,
            macro_model="cauchy_3d",
            constraints=constraints,
        ).Dbar.detach().cpu().numpy()

        # 3D Tet4 SG — exact Python-level extrusion of the same 2D mesh
        cls.mesh_3d = _extrude_tri3_to_tet4(cls.mesh_2d, lz=1.0)
        cls.Dbar_3d = homogenize_msg(
            mesh=cls.mesh_3d,
            material_stiffness=material_stiffness,
            macro_model="cauchy_3d",
            constraints=constraints,
        ).Dbar.detach().cpu().numpy()

    def test_meshes_share_identical_cross_section(self) -> None:
        """The z=0 nodes of the 3D mesh must be bitwise-identical to the 2D nodes."""
        n = self.mesh_2d.n_nodes
        np.testing.assert_array_equal(
            self.mesh_3d.nodes[:n],
            self.mesh_2d.nodes,
        )

    def test_3d_sg_matches_2d_sg(self) -> None:
        """All 36 Dbar components must agree to 1e-5 relative tolerance.

        Because the 3D mesh is an exact extrusion of the 2D mesh and z-periodic
        BC forces z-independence, the two solves are mathematically equivalent.
        Differences arise only from floating-point round-off in the separate
        assembly and solver code paths.
        """
        # atol=1e-10 handles entries that are structurally zero in 2D (exact zeros
        # from the flat assembly) but ~1e-14 floating-point noise in 3D.  All
        # physically significant entries (~1–30 GPa) are governed by the rtol.
        np.testing.assert_allclose(
            self.Dbar_3d, self.Dbar_2d,
            rtol=1e-5,
            atol=1e-10,
            err_msg=(
                f"3D SG Dbar:\n{self.Dbar_3d}\n"
                f"2D SG Dbar:\n{self.Dbar_2d}"
            ),
        )

    def test_fiber_direction_identical(self) -> None:
        """C_33 (fiber / Z direction) must agree to 1e-5 relative tolerance."""
        rel = abs(self.Dbar_3d[2, 2] - self.Dbar_2d[2, 2]) / self.Dbar_2d[2, 2]
        self.assertLess(rel, 1e-5,
                        f"C_33: 3D={self.Dbar_3d[2,2]:.6f}, 2D={self.Dbar_2d[2,2]:.6f}, rel={rel:.2e}")

    def test_transverse_stiffness_identical(self) -> None:
        """C_11 and C_22 must agree to 1e-5 relative tolerance."""
        for i in (0, 1):
            rel = abs(self.Dbar_3d[i, i] - self.Dbar_2d[i, i]) / self.Dbar_2d[i, i]
            self.assertLess(rel, 1e-5,
                            f"C_{i+1}{i+1}: 3D={self.Dbar_3d[i,i]:.6f}, 2D={self.Dbar_2d[i,i]:.6f}, rel={rel:.2e}")


if __name__ == "__main__":
    unittest.main()
