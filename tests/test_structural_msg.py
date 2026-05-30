from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from openmsg.config import run_config
from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import (
    homogenize_euler_bernoulli_beam,
    homogenize_kirchhoff_love_plate,
)
from openmsg.materials import isotropic_stiffness
from openmsg.mesh import SolidMesh
from openmsg.plate import Ply, laminate_abd
from tests.mesh_builders import structured_hex_mesh


def line_thickness_mesh(n_elements: int = 16) -> SolidMesh:
    z = np.linspace(-0.5, 0.5, n_elements + 1)
    nodes = np.array([[0.0, 0.0, zi] for zi in z], dtype=float)
    elements = np.array([[idx, idx + 1] for idx in range(n_elements)], dtype=int)
    return SolidMesh(nodes=nodes, elements=elements, material_ids=("m",) * n_elements, element_type="line2")


def rectangular_cross_section_mesh(n: int = 4) -> SolidMesh:
    ys = np.linspace(-0.5, 0.5, n + 1)
    zs = np.linspace(-0.5, 0.5, n + 1)
    nodes = np.array([[0.0, y, z] for z in zs for y in ys], dtype=float)

    def node_id(i: int, j: int) -> int:
        return i + (n + 1) * j

    elements: list[list[int]] = []
    for j in range(n):
        for i in range(n):
            elements.append([node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)])
    return SolidMesh(
        nodes=nodes,
        elements=np.asarray(elements, dtype=int),
        material_ids=("m",) * len(elements),
        element_type="quad4",
    )


class StructuralMSGTests(unittest.TestCase):
    def test_line2_plate_msg_converges_to_laminate_abd(self) -> None:
        mesh = line_thickness_mesh(n_elements=16)
        C = isotropic_stiffness(100.0, 0.25)
        expected = laminate_abd([Ply(C, thickness=1.0)]).ABD

        result = homogenize_kirchhoff_love_plate(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(result.metadata["macro_model"], "kirchhoff_love_plate")
        self.assertEqual(result.Dbar.shape, (6, 6))
        np.testing.assert_allclose(result.Dbar[:3, :3], expected[:3, :3], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.Dbar[3:, 3:], expected[3:, 3:], rtol=6e-4, atol=6e-3)
        self.assertEqual(result.to_dict()["ABD"], result.to_dict()["Dbar"])

    def test_quad4_beam_msg_recovers_homogeneous_axial_and_bending_terms(self) -> None:
        mesh = rectangular_cross_section_mesh(n=4)
        young = 100.0
        C = isotropic_stiffness(young, 0.25)

        result = homogenize_euler_bernoulli_beam(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(result.metadata["macro_model"], "euler_bernoulli_beam")
        self.assertEqual(result.Dbar.shape, (4, 4))
        self.assertAlmostEqual(result.Dbar[0, 0], young, places=10)
        self.assertAlmostEqual(result.Dbar[2, 2], young / 12.0, delta=0.06)
        self.assertAlmostEqual(result.Dbar[3, 3], young / 12.0, delta=0.06)
        np.testing.assert_allclose(result.Dbar, result.Dbar.T, atol=1e-10)
        self.assertEqual(result.to_dict()["K"], result.to_dict()["Dbar"])

    def test_3d_sg_plate_and_beam_defaults_run_with_retained_axis_periodicity(self) -> None:
        mesh = structured_hex_mesh(
            bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
            cells=(1, 1, 1),
            default_material="m",
        )
        C = isotropic_stiffness(100.0, 0.25)

        plate_result = homogenize_kirchhoff_love_plate(mesh=mesh, material_stiffness={"m": C})
        beam_result = homogenize_euler_bernoulli_beam(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(plate_result.Dbar.shape, (6, 6))
        self.assertEqual(beam_result.Dbar.shape, (4, 4))
        self.assertGreater(np.min(np.linalg.eigvalsh(plate_result.Dbar)), 0.0)
        self.assertGreater(np.min(np.linalg.eigvalsh(beam_result.Dbar)), 0.0)
        self.assertEqual(plate_result.metadata["omega"], 1.0)
        self.assertEqual(beam_result.metadata["omega"], 1.0)

    def test_plate_msg_dehomogenization_accepts_generalized_strain(self) -> None:
        mesh = line_thickness_mesh(n_elements=4)
        C = isotropic_stiffness(100.0, 0.25)
        result = homogenize_kirchhoff_love_plate(mesh=mesh, material_stiffness={"m": C})
        macro_strain = np.array([0.01, -0.02, 0.004, 0.1, -0.05, 0.02])

        fields = recover_gauss_fields(
            mesh=mesh,
            material_stiffness={"m": C},
            V0=result.V0,
            macro_strain=macro_strain,
            macro_model="kirchhoff_love_plate",
        )

        self.assertEqual(fields["strain"].shape, (8, 6))
        self.assertEqual(fields["stress"].shape, (8, 6))

    def test_structural_msg_config_inputs_run(self) -> None:
        C = {"type": "isotropic", "E": 100.0, "nu": 0.25}
        plate_config = {
            "analysis": {"type": "msg_kirchhoff_love_plate"},
            "materials": {"m": C},
            "mesh": {
                "type": "line2",
                "active_axes": ["z"],
                "nodes": [[-0.5], [0.0], [0.5]],
                "elements": [{"nodes": [0, 1], "material": "m"}, {"nodes": [1, 2], "material": "m"}],
            },
        }
        beam_config = {
            "analysis": {"type": "msg_euler_bernoulli_beam"},
            "materials": {"m": C},
            "mesh": {
                "type": "quad4",
                "active_axes": ["y", "z"],
                "nodes": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                "elements": [{"nodes": [0, 1, 2, 3], "material": "m"}],
            },
        }

        plate_result = run_config(plate_config)
        beam_result = run_config(beam_config)

        self.assertEqual(plate_result.metadata["macro_model"], "kirchhoff_love_plate")
        self.assertEqual(beam_result.metadata["macro_model"], "euler_bernoulli_beam")
        self.assertEqual(plate_result.Dbar.shape, (6, 6))
        self.assertEqual(beam_result.Dbar.shape, (4, 4))

    def test_structural_msg_examples_run(self) -> None:
        root = Path(__file__).resolve().parents[1]
        examples = [
            ("homogeneous_plate_msg_line2.json", "kirchhoff_love_plate", (6, 6)),
            ("homogeneous_beam_msg_quad4.json", "euler_bernoulli_beam", (4, 4)),
        ]

        for filename, macro_model, shape in examples:
            with self.subTest(filename=filename):
                result = run_config(root / "examples" / filename)
                self.assertEqual(result.metadata["macro_model"], macro_model)
                self.assertEqual(result.Dbar.shape, shape)

    def test_structural_msg_dense_solver_override_matches_sparse(self) -> None:
        C = isotropic_stiffness(100.0, 0.25)
        cases = [
            (homogenize_kirchhoff_love_plate, line_thickness_mesh(n_elements=4)),
            (homogenize_euler_bernoulli_beam, rectangular_cross_section_mesh(n=2)),
        ]

        for homogenize, mesh in cases:
            with self.subTest(model=homogenize.__name__):
                sparse_result = homogenize(mesh=mesh, material_stiffness={"m": C})
                dense_result = homogenize(mesh=mesh, material_stiffness={"m": C}, linear_solver="dense")
                self.assertEqual(sparse_result.metadata["assembly_kernel"], "tensormesh_element_assembler")
                np.testing.assert_allclose(dense_result.Dbar, sparse_result.Dbar, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(dense_result.H, sparse_result.H, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(dense_result.D0, sparse_result.D0, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
