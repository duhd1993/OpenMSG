from __future__ import annotations

import unittest

import numpy as np

from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import homogenize_3d_cauchy
from openmsg.materials import isotropic_stiffness
from openmsg.mesh import SolidMesh, mesh_from_config
from openmsg.assembly import assemble_3d_cauchy


def line2_sg_mesh() -> SolidMesh:
    return SolidMesh(
        nodes=np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
        elements=np.array([[0, 1]], dtype=int),
        material_ids=("m",),
        element_type="line2",
    )


def quad4_sg_mesh() -> SolidMesh:
    return SolidMesh(
        nodes=np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        material_ids=("m",),
        element_type="quad4",
    )


def tri3_sg_mesh() -> SolidMesh:
    return SolidMesh(
        nodes=np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        elements=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        material_ids=("m", "m"),
        element_type="tri3",
    )


class ReducedSGTests(unittest.TestCase):
    def test_lower_dimensional_sg_runs(self) -> None:
        mesh = line2_sg_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(result.metadata["assembly_kernel"], "tensormesh_autograd")
        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)

    def test_lower_dimensional_sg_assembly_matches_homogeneous_reference(self) -> None:
        C = isotropic_stiffness(100.0, 0.25)
        cases = [line2_sg_mesh(), quad4_sg_mesh(), tri3_sg_mesh()]

        for mesh in cases:
            with self.subTest(element_type=mesh.element_type):
                assembly, metadata = assemble_3d_cauchy(mesh, {"m": C})
                result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})
                self.assertEqual(metadata["assembly_kernel"], "tensormesh_autograd")
                np.testing.assert_allclose(assembly.E, assembly.E.T, atol=1e-11)
                np.testing.assert_allclose(assembly.D0, C * assembly.volume, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)

    def test_line2_homogeneous_sg_recovers_3d_cauchy_stiffness(self) -> None:
        mesh = line2_sg_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        assembly, _ = assemble_3d_cauchy(mesh, {"m": C})
        result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(mesh.sg_dimension, 1)
        self.assertEqual(mesh.active_axes, (2,))
        self.assertAlmostEqual(assembly.volume, 1.0)
        np.testing.assert_allclose(assembly.D0, C, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.metadata["sg_dimension"], 1)
        self.assertEqual(result.metadata["active_axes"], [2])

    def test_quad4_homogeneous_sg_recovers_3d_cauchy_stiffness(self) -> None:
        mesh = quad4_sg_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(mesh.sg_dimension, 2)
        self.assertEqual(mesh.active_axes, (1, 2))
        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)

    def test_tri3_homogeneous_sg_recovers_3d_cauchy_stiffness(self) -> None:
        mesh = tri3_sg_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})

        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)

    def test_reduced_sg_dehomogenization_quadrature_counts(self) -> None:
        C = isotropic_stiffness(100.0, 0.25)
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])
        cases = [
            (line2_sg_mesh(), 2),  # 1 element x 2 quadrature points
            (quad4_sg_mesh(), 4),  # 1 element x 4 quadrature points
            (tri3_sg_mesh(), 6),  # 2 elements x 3 quadrature points
        ]

        for mesh, expected_points in cases:
            with self.subTest(element_type=mesh.element_type):
                result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})
                fields = recover_gauss_fields(
                    mesh=mesh,
                    material_stiffness={"m": C},
                    V0=result.V0,
                    macro_strain=macro_strain,
                )
                self.assertEqual(fields["strain"].shape, (expected_points, 6))
                for strain in fields["strain"]:
                    np.testing.assert_allclose(strain, macro_strain, atol=1e-10)

    def test_reduced_sg_config_active_axes(self) -> None:
        line_mesh = mesh_from_config(
            {
                "type": "line2",
                "active_axes": ["z"],
                "nodes": [[0], [1]],
                "elements": [{"nodes": [0, 1], "material": "m"}],
            }
        )
        quad_mesh = mesh_from_config(
            {
                "type": "quad4",
                "active_axes": ["y", "z"],
                "nodes": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "elements": [{"nodes": [0, 1, 2, 3], "material": "m"}],
            }
        )

        self.assertEqual(line_mesh.element_type, "line2")
        self.assertEqual(line_mesh.active_axes, (2,))
        self.assertEqual(line_mesh.nodes.shape, (2, 3))
        self.assertEqual(quad_mesh.element_type, "quad4")
        self.assertEqual(quad_mesh.active_axes, (1, 2))
        self.assertEqual(quad_mesh.nodes.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
