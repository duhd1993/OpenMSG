from __future__ import annotations

import unittest

import numpy as np

from openmsg.elements import iter_element_quadrature


class ElementQuadratureTests(unittest.TestCase):
    def test_supported_element_quadrature_shapes_and_measures(self) -> None:
        cases = [
            (
                "hex8",
                np.array(
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
                ),
                None,
                8,
                8,
                1.0,
            ),
            (
                "tet4",
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
                None,
                4,
                1,
                1.0 / 6.0,
            ),
            (
                "quad4",
                np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float),
                (1, 2),
                4,
                4,
                1.0,
            ),
            (
                "tri3",
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
                (1, 2),
                3,
                1,
                0.5,
            ),
            (
                "line2",
                np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
                (2,),
                2,
                2,
                1.0,
            ),
        ]

        for element_type, coords, active_axes, n_nodes, n_points, measure in cases:
            with self.subTest(element_type=element_type):
                qps = iter_element_quadrature(element_type, coords, active_axes=active_axes)
                self.assertEqual(len(qps), n_points)
                self.assertAlmostEqual(sum(qp.dV for qp in qps), measure)
                for qp in qps:
                    self.assertEqual(qp.shape.shape, (n_nodes,))
                    self.assertEqual(qp.dN_dx.shape, (n_nodes, 3))
                    self.assertAlmostEqual(float(np.sum(qp.shape)), 1.0)

    def test_unsupported_element_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported element_type"):
            iter_element_quadrature("beam42", np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
