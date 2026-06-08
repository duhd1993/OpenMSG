from __future__ import annotations

import unittest

import torch

from openmsg.materials import (
    cubic_stiffness,
    engineering_constants_from_stiffness,
    isotropic_stiffness,
    orientation_matrix_from_spec,
    orthotropic_stiffness,
    rotate_stiffness,
    rotate_stiffness_by_axis_permutation,
    rotation_matrix_from_axis_angle,
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
        self.assertIsInstance(C, torch.Tensor)
        torch.testing.assert_close(C, C.T)
        self.assertAlmostEqual(float(C[0, 1]), lam)
        self.assertAlmostEqual(float(C[0, 0]), lam + 2.0 * mu)
        self.assertAlmostEqual(float(C[3, 3]), mu)
        self.assertAlmostEqual(float(C[4, 4]), mu)
        self.assertAlmostEqual(float(C[5, 5]), mu)
        self.assertGreater(float(torch.linalg.eigvalsh(C).min()), 0.0)

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
            self.assertIsInstance(C, torch.Tensor)
            torch.testing.assert_close(C, C.T, atol=1e-12, rtol=0.0)
            self.assertGreater(float(torch.linalg.eigvalsh(C).min()), 0.0)

        constants = engineering_constants_from_stiffness(ti_z)
        self.assertAlmostEqual(constants["E_z"], 140.0)
        self.assertAlmostEqual(constants["E_x"], 10.0)
        self.assertAlmostEqual(constants["E_y"], 10.0)

    def test_isotropic_stiffness_preserves_parameter_autograd(self) -> None:
        young = torch.tensor(210.0, dtype=torch.float64, requires_grad=True)
        C = isotropic_stiffness(young, 0.3)

        C[0, 0].backward()

        self.assertIsNotNone(young.grad)
        self.assertGreater(float(young.grad), 0.0)

    def test_isotropic_stiffness_is_rotation_invariant(self) -> None:
        C = isotropic_stiffness(210.0, 0.3)
        R = rotation_matrix_from_axis_angle(
            axis=[1.0, 2.0, 3.0],
            angle_degrees=37.0,
        )

        rotated = rotate_stiffness(C, R)

        torch.testing.assert_close(rotated, C, rtol=1e-10, atol=1e-10)

    def test_full_rotation_matches_axis_permutation(self) -> None:
        C = orthotropic_stiffness(
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
        local_to_global = (1, 2, 0)
        R = torch.zeros((3, 3), dtype=torch.float64)
        for local_axis, global_axis in enumerate(local_to_global):
            R[global_axis, local_axis] = 1.0

        full_rotation = rotate_stiffness(C, R)
        permutation = rotate_stiffness_by_axis_permutation(C, local_to_global)

        torch.testing.assert_close(full_rotation, permutation, rtol=1e-12, atol=1e-12)

    def test_axis_angle_rotation_preserves_angle_autograd(self) -> None:
        C = orthotropic_stiffness(
            E1=60.0,
            E2=20.0,
            E3=10.0,
            nu12=0.25,
            nu13=0.20,
            nu23=0.18,
            G12=8.0,
            G13=6.0,
            G23=4.0,
        )
        angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        R = rotation_matrix_from_axis_angle(axis=[0.0, 0.0, 1.0], angle_radians=angle)

        rotated = rotate_stiffness(C, R)
        rotated[0, 0].backward()

        self.assertIsNotNone(angle.grad)
        self.assertTrue(torch.isfinite(angle.grad))
        self.assertGreater(float(angle.grad.abs()), 0.0)

    def test_matrix_orientation_spec_is_local_to_global(self) -> None:
        R = orientation_matrix_from_spec(
            {
                "type": "matrix",
                "local_to_global": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        )

        expected = rotation_matrix_from_axis_angle(
            axis=[0.0, 0.0, 1.0],
            angle_degrees=90.0,
        )
        torch.testing.assert_close(R, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
