from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.assembly import assemble_msg_system
from openmsg.macro import macro_model_from_kind
from openmsg.materials import isotropic_stiffness
from tests.mesh_builders import structured_hex_mesh


class AssemblyTests(unittest.TestCase):
    def test_single_homogeneous_cube_d0_is_volume_times_stiffness(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)
        system = assemble_msg_system(mesh, {"m": C}, macro_model=macro_model_from_kind("cauchy_3d", mesh=mesh))
        E = system.E.to_dense()

        self.assertEqual(system.metadata["assembly_kernel"], "tensormesh_autograd")
        self.assertEqual(tuple(E.shape), (mesh.n_dof, mesh.n_dof))
        self.assertEqual(tuple(system.H.shape), (mesh.n_dof, 6))
        self.assertAlmostEqual(float(system.volume), 1.0)
        torch.testing.assert_close(E, E.T, rtol=0.0, atol=1e-11)
        torch.testing.assert_close(system.D0, C, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(system.node_weights.sum()), 1.0)

    def test_assembly_rejects_numpy_material_stiffness(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")

        with self.assertRaisesRegex(TypeError, "must be a torch.Tensor"):
            assemble_msg_system(mesh, {"m": np.eye(6)}, macro_model=macro_model_from_kind("cauchy_3d", mesh=mesh))


if __name__ == "__main__":
    unittest.main()
