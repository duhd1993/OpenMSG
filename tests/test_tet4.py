from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.assembly import assemble_msg_system
from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import effective_stiffness
from openmsg.macro import macro_model_from_kind
from openmsg.materials import isotropic_stiffness
from openmsg.mesh import SolidMesh, mesh_from_config


def cube_tet_mesh() -> SolidMesh:
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
    return SolidMesh(nodes=nodes, elements=elements, material_ids=("m",) * 6, element_type="tet4")


class Tet4Tests(unittest.TestCase):
    def test_single_tet4_assembly_volume_and_d0(self) -> None:
        mesh = SolidMesh(
            nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            elements=np.array([[0, 1, 2, 3]], dtype=int),
            material_ids=("m",),
            element_type="tet4",
        )
        C = isotropic_stiffness(100.0, 0.25)
        system = assemble_msg_system(mesh, {"m": C}, macro_model=macro_model_from_kind("cauchy_3d", mesh=mesh))

        self.assertEqual(system.metadata["assembly_kernel"], "tensormesh_autograd")
        self.assertEqual(tuple(system.E.to_dense().shape), (12, 12))
        self.assertEqual(tuple(system.H.shape), (12, 6))
        self.assertAlmostEqual(float(system.volume), 1.0 / 6.0)
        torch.testing.assert_close(system.D0, C / 6.0, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(system.node_weights.sum()), 1.0 / 6.0)

    def test_tet4_homogeneous_periodic_cube_recovers_input_stiffness(self) -> None:
        mesh = cube_tet_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.metadata["n_elements"], 6)

    def test_tet4_dehomogenization_uses_tensormesh_quadrature_points(self) -> None:
        mesh = cube_tet_mesh()
        C = isotropic_stiffness(100.0, 0.25)
        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])

        fields = recover_gauss_fields(mesh=mesh, material_stiffness={"m": C}, V0=result.V0, macro_strain=macro_strain)

        # 6 tetrahedra x 4 quadrature points per element (TensorMesh order-2 rule).
        self.assertEqual(fields["strain"].shape, (24, 6))
        for strain in fields["strain"]:
            np.testing.assert_allclose(strain, macro_strain, atol=1e-10)

    def test_tet4_config_input(self) -> None:
        config = {
            "type": "tet4",
            "nodes": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "elements": [{"nodes": [0, 1, 2, 3], "material": "m"}],
        }

        mesh = mesh_from_config(config)

        self.assertEqual(mesh.element_type, "tet4")
        self.assertEqual(mesh.elements.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
