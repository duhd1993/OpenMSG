from __future__ import annotations

import unittest

import numpy as np

from openmsg.voigt import gradient_to_voigt_strain, strain_displacement_matrix


class VoigtTests(unittest.TestCase):
    def test_gradient_to_voigt_strain(self) -> None:
        grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_allclose(
            gradient_to_voigt_strain(grad),
            np.array([1.0, 5.0, 9.0, 14.0, 10.0, 6.0]),
        )

    def test_strain_displacement_matrix_shape(self) -> None:
        dN = np.array([[-1.0, 0.0, 0.5], [1.0, 0.0, -0.5]])
        B = strain_displacement_matrix(dN)
        self.assertEqual(B.shape, (6, 6))
        dofs = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        grad = np.array(
            [
                [-1.0 * 2.0 + 1.0 * 7.0, 0.0, 0.5 * 2.0 - 0.5 * 7.0],
                [-1.0 * 3.0 + 1.0 * 11.0, 0.0, 0.5 * 3.0 - 0.5 * 11.0],
                [-1.0 * 5.0 + 1.0 * 13.0, 0.0, 0.5 * 5.0 - 0.5 * 13.0],
            ]
        )
        np.testing.assert_allclose(B @ dofs, gradient_to_voigt_strain(grad))


if __name__ == "__main__":
    unittest.main()

