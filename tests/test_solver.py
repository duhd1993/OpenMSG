from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.solver import compute_effective_stiffness, solve_constrained, solve_constrained_sparse


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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
        np.testing.assert_allclose(_np(V0), expected_v0, rtol=1e-10, atol=1e-10)

        Dbar = compute_effective_stiffness(D0, H, V0)
        expected = D0 - H.T @ np.linalg.solve(E, H)
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(_np(Dbar), expected, rtol=1e-10, atol=1e-10)

    def test_constrained_solution_satisfies_GV_zero(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.normal(size=(5, 5))
        E = A.T @ A + np.eye(5)
        H = rng.normal(size=(5, 6))
        G = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]])

        V0 = solve_constrained(E, H, G)
        np.testing.assert_allclose(G @ _np(V0), np.zeros((2, 6)), atol=1e-10)

    def test_sparse_constrained_solution_matches_dense(self) -> None:
        rng = np.random.default_rng(11)
        A = rng.normal(size=(8, 8))
        E = A.T @ A + 5.0 * np.eye(8)
        H = rng.normal(size=(8, 6))
        G = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]])

        dense_v0, dense_lagrange = solve_constrained(E, H, G, return_lagrange=True)
        sparse_v0, sparse_lagrange = solve_constrained_sparse(E, H, G, return_lagrange=True)

        np.testing.assert_allclose(_np(sparse_v0), _np(dense_v0), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(_np(sparse_lagrange), _np(dense_lagrange), rtol=1e-9, atol=1e-9)

    def test_solver_is_differentiable_wrt_matrix(self) -> None:
        # dDbar/dE through the constrained solve, checked against finite differences.
        rng = np.random.default_rng(3)
        A = rng.normal(size=(6, 6))
        E0 = torch.tensor(A.T @ A + 3.0 * np.eye(6), dtype=torch.float64)
        H = torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float64)
        D0 = torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float64)
        D0 = D0 + D0.T
        G = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

        def loss_from(E_sym):
            V0 = solve_constrained_sparse(E_sym, H, G)
            Dbar = compute_effective_stiffness(D0, H, V0)
            return (Dbar ** 2).sum()

        E = E0.clone().requires_grad_(True)
        loss = loss_from(0.5 * (E + E.T))
        (grad,) = torch.autograd.grad(loss, E)
        self.assertTrue(torch.isfinite(grad).all())

        # Finite-difference check on a single symmetric perturbation direction.
        eps = 1e-6
        d = torch.zeros_like(E0)
        d[0, 1] = d[1, 0] = 1.0
        with torch.no_grad():
            f_plus = loss_from(0.5 * ((E0 + eps * d) + (E0 + eps * d).T))
            f_minus = loss_from(0.5 * ((E0 - eps * d) + (E0 - eps * d).T))
        fd = float((f_plus - f_minus) / (2 * eps))
        analytic = float((grad * d).sum())
        self.assertAlmostEqual(fd, analytic, places=4)


if __name__ == "__main__":
    unittest.main()
