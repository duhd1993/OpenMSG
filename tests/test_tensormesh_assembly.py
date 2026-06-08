from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.homogenize import effective_stiffness
from openmsg.materials import (
    isotropic_stiffness,
    orthotropic_stiffness,
    rotate_stiffness,
    rotation_matrix_from_axis_angle,
)
from openmsg.mesh import SolidMesh
from tests.mesh_builders import structured_hex_mesh


class TensorMeshAssemblyTests(unittest.TestCase):
    def test_tensormesh_assembly_recovers_homogeneous_cube_stiffness(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(result.metadata["assembly_kernel"], "tensormesh_autograd")
        self.assertEqual(result.metadata["linear_solver"], "sparse")
        E = result.E.to_dense()
        torch.testing.assert_close(E, E.T, rtol=0.0, atol=1e-11)
        torch.testing.assert_close(result.D0, C, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)

    def test_tensormesh_assembly_accepts_per_element_anisotropic_stiffness(self) -> None:
        mesh = structured_hex_mesh(
            bounds=((0, 2), (0, 1), (0, 1)),
            cells=(2, 1, 1),
            default_material="matrix",
            cell_materials=("matrix", "fiber"),
        )
        materials = {
            "matrix": isotropic_stiffness(10.0, 0.3),
            "fiber": orthotropic_stiffness(
                E1=70.0,
                E2=12.0,
                E3=12.0,
                nu12=0.25,
                nu13=0.25,
                nu23=0.35,
                G12=5.0,
                G13=5.0,
                G23=4.0,
            ),
        }

        sparse_result = effective_stiffness(mesh=mesh, material_stiffness=materials)
        dense_result = effective_stiffness(
            mesh=mesh,
            material_stiffness=materials,
            linear_solver="dense",
        )

        np.testing.assert_allclose(sparse_result.Dbar, dense_result.Dbar, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(sparse_result.Dbar, sparse_result.Dbar.T, atol=1e-10)
        self.assertGreater(np.min(np.linalg.eigvalsh(sparse_result.Dbar)), 0.0)

    def test_homogeneous_orientation_recovers_rotated_stiffness(self) -> None:
        base = structured_hex_mesh(
            bounds=((0, 1), (0, 1), (0, 1)),
            cells=(1, 1, 1),
            default_material="m",
        )
        mesh = SolidMesh(
            nodes=base.nodes,
            elements=[
                {
                    "type": "hex8",
                    "connectivity": base.element_blocks[0].elements,
                    "material": "m",
                    "orientation": {
                        "type": "axis_angle",
                        "axis": [0.0, 0.0, 1.0],
                        "angle_degrees": 30.0,
                    },
                }
            ],
        )
        C = orthotropic_stiffness(
            E1=70.0,
            E2=12.0,
            E3=8.0,
            nu12=0.24,
            nu13=0.20,
            nu23=0.18,
            G12=5.0,
            G13=4.0,
            G23=3.0,
        )
        R = rotation_matrix_from_axis_angle(axis=[0.0, 0.0, 1.0], angle_degrees=30.0)
        expected = rotate_stiffness(C, R)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        self.assertTrue(result.metadata["has_material_orientation"])
        torch.testing.assert_close(result.Dbar, expected, rtol=1e-10, atol=1e-10)

    def test_per_element_orientation_list_is_accepted(self) -> None:
        base = structured_hex_mesh(
            bounds=((0, 2), (0, 1), (0, 1)),
            cells=(2, 1, 1),
            default_material="m",
        )
        mesh = SolidMesh(
            nodes=base.nodes,
            elements=[
                {
                    "type": "hex8",
                    "connectivity": base.element_blocks[0].elements,
                    "material": "m",
                    "orientation": [
                        {
                            "type": "axis_angle",
                            "axis": [0.0, 0.0, 1.0],
                            "angle_degrees": 0.0,
                        },
                        {
                            "type": "axis_angle",
                            "axis": [0.0, 0.0, 1.0],
                            "angle_degrees": 90.0,
                        },
                    ],
                }
            ],
        )
        C = orthotropic_stiffness(
            E1=70.0,
            E2=12.0,
            E3=8.0,
            nu12=0.24,
            nu13=0.20,
            nu23=0.18,
            G12=5.0,
            G13=4.0,
            G23=3.0,
        )

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        self.assertTrue(mesh.has_material_orientation)
        self.assertTrue(result.metadata["has_material_orientation"])
        torch.testing.assert_close(result.Dbar, result.Dbar.T, rtol=0.0, atol=1e-10)

    def test_orientation_list_length_must_match_element_count(self) -> None:
        base = structured_hex_mesh(
            bounds=((0, 2), (0, 1), (0, 1)),
            cells=(2, 1, 1),
            default_material="m",
        )

        with self.assertRaisesRegex(ValueError, "orientation list length"):
            SolidMesh(
                nodes=base.nodes,
                elements=[
                    {
                        "type": "hex8",
                        "connectivity": base.element_blocks[0].elements,
                        "material": "m",
                        "orientation": [
                            {
                                "type": "axis_angle",
                                "axis": [0.0, 0.0, 1.0],
                                "angle_degrees": 0.0,
                            }
                        ],
                    }
                ],
            )

    def test_same_type_block_merge_preserves_orientation_specs(self) -> None:
        base = structured_hex_mesh(
            bounds=((0, 2), (0, 1), (0, 1)),
            cells=(2, 1, 1),
            default_material="m",
        )
        conn = base.element_blocks[0].elements

        mesh = SolidMesh(
            nodes=base.nodes,
            elements=[
                {
                    "type": "hex8",
                    "connectivity": conn[:1],
                    "material": "m",
                    "orientation": {
                        "type": "axis_angle",
                        "axis": [0.0, 0.0, 1.0],
                        "angle_degrees": 0.0,
                    },
                },
                {
                    "type": "hex8",
                    "connectivity": conn[1:],
                    "material": "m",
                    "orientation": {
                        "type": "axis_angle",
                        "axis": [0.0, 0.0, 1.0],
                        "angle_degrees": 90.0,
                    },
                },
            ],
        )

        self.assertEqual(len(mesh.element_blocks), 1)
        self.assertEqual(len(mesh.element_blocks[0].orientation_specs), 2)
        self.assertEqual(mesh.element_blocks[0].orientation_specs[1]["angle_degrees"], 90.0)

    def test_dense_solver_override_matches_sparse(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)

        sparse_result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        dense_result = effective_stiffness(
            mesh=mesh,
            material_stiffness={"m": C},
            linear_solver="dense",
        )

        self.assertEqual(dense_result.metadata["linear_solver"], "dense")
        np.testing.assert_allclose(sparse_result.Dbar, dense_result.Dbar, rtol=1e-10, atol=1e-10)

    def test_tet4_assembly_recovers_homogeneous_stiffness(self) -> None:
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        elements = np.array(
            [
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
                [0, 4, 5, 6],
                [0, 5, 1, 6],
            ],
            dtype=int,
        )
        mesh = SolidMesh(
            nodes=nodes,
            elements=[{"type": "tet4", "connectivity": elements, "material": "m"}],
        )
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
