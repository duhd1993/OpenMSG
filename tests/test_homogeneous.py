from __future__ import annotations

import unittest

import numpy as np

from openmsg.homogenize import effective_stiffness
from openmsg.materials import isotropic_stiffness
from tests.mesh_builders import structured_hex_mesh


class HomogeneousTests(unittest.TestCase):
    def test_periodic_homogeneous_sg_recovers_input_stiffness(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(
            mesh=mesh,
            material_stiffness={"m": C},
            constraints=[{"type": "periodic", "axes": ["x", "y", "z"]}, {"type": "mean_zero"}],
        )

        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.Dbar, result.Dbar.T, atol=1e-12)
        self.assertGreater(np.min(np.linalg.eigvalsh(result.Dbar)), 0.0)
        np.testing.assert_allclose(result.G @ result.V0, np.zeros((result.G.shape[0], 6)), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
