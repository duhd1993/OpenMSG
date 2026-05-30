from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.homogenize import effective_stiffness
from openmsg.materials import isotropic_stiffness, orthotropic_stiffness
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
        mesh = SolidMesh(nodes=nodes, elements=elements, material_ids=("m",) * 6, element_type="tet4")
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
