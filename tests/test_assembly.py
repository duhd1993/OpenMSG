from __future__ import annotations

import unittest

import numpy as np

from openmsg.materials import isotropic_stiffness
from openmsg.assembly import assemble_3d_cauchy
from tests.mesh_builders import structured_hex_mesh


class AssemblyTests(unittest.TestCase):
    def test_single_homogeneous_cube_d0_is_volume_times_stiffness(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)
        assembly, metadata = assemble_3d_cauchy(mesh, {"m": C})

        self.assertEqual(metadata["assembly_kernel"], "tensormesh_autograd")
        self.assertEqual(assembly.E.shape, (mesh.n_dof, mesh.n_dof))
        self.assertEqual(assembly.H.shape, (mesh.n_dof, 6))
        self.assertAlmostEqual(assembly.volume, 1.0)
        np.testing.assert_allclose(assembly.E, assembly.E.T, atol=1e-11)
        np.testing.assert_allclose(assembly.D0, C, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(assembly.node_weights)), 1.0)


if __name__ == "__main__":
    unittest.main()
