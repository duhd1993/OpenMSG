from __future__ import annotations

import unittest
import importlib.util

import numpy as np

from openmsg.solver import compute_effective_stiffness, solve_constrained, solve_constrained_sparse


class SolverTests(unittest.TestCase):
    def test_unconstrained_dbar_matches_schur_formula(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.normal(size=(7, 7))
        E = A.T @ A + 2.0 * np.eye(7)
        H = rng.normal(size=(7, 6))
        B = rng.normal(size=(6, 6))
        D0 = B.T @ B + 10.0 * np.eye(6)

        V0 = solve_constrained(E, H)
        expected_v0 = -np.linalg.solve(E, H)
        np.testing.assert_allclose(V0, expected_v0, rtol=1e-12, atol=1e-12)

        Dbar = compute_effective_stiffness(D0, H, V0)
        expected = D0 - H.T @ np.linalg.solve(E, H)
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(Dbar, expected, rtol=1e-12, atol=1e-12)

    def test_constrained_solution_satisfies_GV_zero(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.normal(size=(5, 5))
        E = A.T @ A + np.eye(5)
        H = rng.normal(size=(5, 6))
        G = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]])

        V0 = solve_constrained(E, H, G)
        np.testing.assert_allclose(G @ V0, np.zeros((2, 6)), atol=1e-11)

    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "scipy is not installed")
    def test_sparse_constrained_solution_matches_dense(self) -> None:
        rng = np.random.default_rng(11)
        A = rng.normal(size=(8, 8))
        E = A.T @ A + 5.0 * np.eye(8)
        H = rng.normal(size=(8, 6))
        G = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]])

        dense_v0, dense_lagrange = solve_constrained(E, H, G, return_lagrange=True)
        sparse_v0, sparse_lagrange = solve_constrained_sparse(E, H, G, return_lagrange=True)

        np.testing.assert_allclose(sparse_v0, dense_v0, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(sparse_lagrange, dense_lagrange, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
