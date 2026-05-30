from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from openmsg.config import run_config
from openmsg.materials import isotropic_stiffness, orthotropic_stiffness
from openmsg.plate import Ply, laminate_abd, plane_stress_reduced_stiffness, transform_reduced_stiffness_inplane


class PlateLaminateTests(unittest.TestCase):
    def test_plane_stress_reduced_stiffness_for_isotropic_material(self) -> None:
        young = 100.0
        nu = 0.25
        C = isotropic_stiffness(young, nu)
        Q = plane_stress_reduced_stiffness(C)
        expected = young / (1.0 - nu**2) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )

        np.testing.assert_allclose(Q, expected, rtol=1e-12, atol=1e-12)

    def test_isotropic_reduced_stiffness_is_rotation_invariant(self) -> None:
        Q = plane_stress_reduced_stiffness(isotropic_stiffness(100.0, 0.25))

        np.testing.assert_allclose(transform_reduced_stiffness_inplane(Q, 37.0), Q, rtol=1e-12, atol=1e-12)

    def test_single_layer_abd_matches_closed_form(self) -> None:
        C = isotropic_stiffness(100.0, 0.25)
        Q = plane_stress_reduced_stiffness(C)
        thickness = 2.0

        result = laminate_abd([Ply(stiffness=C, thickness=thickness)])

        np.testing.assert_allclose(result.A, Q * thickness, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.B, np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(result.D, Q * thickness**3 / 12.0, rtol=1e-12, atol=1e-12)

    def test_symmetric_cross_ply_has_zero_B(self) -> None:
        C = orthotropic_stiffness(
            E1=140.0,
            E2=10.0,
            E3=10.0,
            nu12=0.28,
            nu13=0.28,
            nu23=0.4,
            G12=5.0,
            G13=5.0,
            G23=3.57,
        )
        plies = [
            Ply(C, 0.125, 0.0),
            Ply(C, 0.125, 90.0),
            Ply(C, 0.125, 90.0),
            Ply(C, 0.125, 0.0),
        ]

        result = laminate_abd(plies)

        np.testing.assert_allclose(result.B, np.zeros((3, 3)), atol=1e-12)
        self.assertGreater(np.min(np.linalg.eigvalsh(result.A)), 0.0)
        self.assertGreater(np.min(np.linalg.eigvalsh(result.D)), 0.0)

    def test_laminate_abd_config_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_config(root / "examples" / "laminate_abd_cross_ply.json")

        self.assertEqual(result.metadata["model"], "kirchhoff_love_laminate")
        self.assertEqual(result.metadata["n_plies"], 4)
        np.testing.assert_allclose(result.B, np.zeros((3, 3)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()

