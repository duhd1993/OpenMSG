from __future__ import annotations

import unittest

import numpy as np

from openmsg.materials import (
    cubic_stiffness,
    engineering_constants_from_stiffness,
    isotropic_stiffness,
    orthotropic_stiffness,
    transversely_isotropic_stiffness,
)


class MaterialTests(unittest.TestCase):
    def test_isotropic_stiffness_uses_engineering_shear(self) -> None:
        young = 210.0
        nu = 0.3
        C = isotropic_stiffness(young, nu)
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = young / (2.0 * (1.0 + nu))

        self.assertEqual(C.shape, (6, 6))
        np.testing.assert_allclose(C, C.T)
        self.assertAlmostEqual(C[0, 1], lam)
        self.assertAlmostEqual(C[0, 0], lam + 2.0 * mu)
        self.assertAlmostEqual(C[3, 3], mu)
        self.assertAlmostEqual(C[4, 4], mu)
        self.assertAlmostEqual(C[5, 5], mu)
        self.assertGreater(np.min(np.linalg.eigvalsh(C)), 0.0)

    def test_other_material_symmetries_are_positive_definite(self) -> None:
        cubic = cubic_stiffness(c11=250.0, c12=120.0, c44=80.0)
        ortho = orthotropic_stiffness(
            E1=40.0,
            E2=20.0,
            E3=10.0,
            nu12=0.25,
            nu13=0.20,
            nu23=0.18,
            G12=8.0,
            G13=6.0,
            G23=4.0,
        )
        ti_z = transversely_isotropic_stiffness(E_l=140.0, E_t=10.0, nu_lt=0.28, nu_tt=0.4, G_lt=5.0, axis="z")

        for C in (cubic, ortho, ti_z):
            np.testing.assert_allclose(C, C.T, atol=1e-12)
            self.assertGreater(np.min(np.linalg.eigvalsh(C)), 0.0)

        constants = engineering_constants_from_stiffness(ti_z)
        self.assertAlmostEqual(constants["E_z"], 140.0)
        self.assertAlmostEqual(constants["E_x"], 10.0)
        self.assertAlmostEqual(constants["E_y"], 10.0)


if __name__ == "__main__":
    unittest.main()
