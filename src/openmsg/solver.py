"""Linear algebra core for MSG homogenization."""

from __future__ import annotations

import numpy as np


def solve_constrained(
    E: np.ndarray,
    H: np.ndarray,
    G: np.ndarray | None = None,
    *,
    return_lagrange: bool = False,
    rcond: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Solve the MSG fluctuation influence matrix.

    Without constraints, this solves ``E V0 = -H``. With constraints, it solves:

    ``[E G.T; G 0] [V0; Lambda] = [-H; 0]``.
    """

    E = np.asarray(E, dtype=float)
    H = np.asarray(H, dtype=float)
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError("E must be a square matrix")
    if H.ndim != 2 or H.shape[0] != E.shape[0]:
        raise ValueError("H must have shape (E.shape[0], n_macro)")

    if G is None or np.asarray(G).size == 0:
        V0 = -np.linalg.solve(E, H)
        if return_lagrange:
            return V0, np.zeros((0, H.shape[1]), dtype=float)
        return V0

    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.shape[1] != E.shape[0]:
        raise ValueError("G must have shape (n_constraints, E.shape[0])")

    n_dof = E.shape[0]
    n_constraints = G.shape[0]
    block = np.zeros((n_dof + n_constraints, n_dof + n_constraints), dtype=float)
    block[:n_dof, :n_dof] = E
    block[:n_dof, n_dof:] = G.T
    block[n_dof:, :n_dof] = G

    rhs = np.zeros((n_dof + n_constraints, H.shape[1]), dtype=float)
    rhs[:n_dof, :] = -H

    try:
        solution = np.linalg.solve(block, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(block, rhs, rcond=rcond)[0]

    V0 = solution[:n_dof, :]
    lagrange = solution[n_dof:, :]
    if return_lagrange:
        return V0, lagrange
    return V0


def solve_constrained_sparse(
    E: object,
    H: np.ndarray,
    G: np.ndarray | None = None,
    *,
    return_lagrange: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Solve the MSG constrained system with scipy sparse matrices."""

    scipy_sparse, scipy_linalg = _import_scipy_sparse()
    E_csr = _as_scipy_csr(E, scipy_sparse)
    H = np.asarray(H, dtype=float)
    if E_csr.shape[0] != E_csr.shape[1]:
        raise ValueError("E must be a square matrix")
    if H.ndim != 2 or H.shape[0] != E_csr.shape[0]:
        raise ValueError("H must have shape (E.shape[0], n_macro)")

    if G is None or np.asarray(G).size == 0:
        factor = scipy_linalg.splu(E_csr.tocsc())
        V0 = -factor.solve(H)
        if return_lagrange:
            return V0, np.zeros((0, H.shape[1]), dtype=float)
        return V0

    G_csr = scipy_sparse.csr_matrix(np.asarray(G, dtype=float))
    if G_csr.shape[1] != E_csr.shape[0]:
        raise ValueError("G must have shape (n_constraints, E.shape[0])")

    n_dof = E_csr.shape[0]
    n_constraints = G_csr.shape[0]
    zero = scipy_sparse.csr_matrix((n_constraints, n_constraints), dtype=float)
    block = scipy_sparse.bmat([[E_csr, G_csr.T], [G_csr, zero]], format="csc")
    rhs = np.zeros((n_dof + n_constraints, H.shape[1]), dtype=float)
    rhs[:n_dof, :] = -H

    factor = scipy_linalg.splu(block)
    solution = factor.solve(rhs)
    V0 = solution[:n_dof, :]
    lagrange = solution[n_dof:, :]
    if return_lagrange:
        return V0, lagrange
    return V0


def compute_effective_stiffness(D0: np.ndarray, H: np.ndarray, V0: np.ndarray, *, omega: float = 1.0) -> np.ndarray:
    """Compute and symmetrize the homogenized stiffness matrix."""

    if omega <= 0.0:
        raise ValueError("omega must be positive")
    D0 = np.asarray(D0, dtype=float)
    H = np.asarray(H, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    if D0.shape != (H.shape[1], H.shape[1]):
        raise ValueError("D0 must have shape (n_macro, n_macro)")
    if V0.shape != H.shape:
        raise ValueError("V0 must have the same shape as H")
    Dbar = (D0 + V0.T @ H) / omega
    return 0.5 * (Dbar + Dbar.T)


def _import_scipy_sparse() -> tuple[object, object]:
    try:
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError as exc:
        raise RuntimeError("sparse constrained solve requires scipy") from exc
    return scipy.sparse, scipy.sparse.linalg


def _as_scipy_csr(matrix: object, scipy_sparse: object) -> object:
    if hasattr(matrix, "to_scipy_coo"):
        return matrix.to_scipy_coo().tocsr()
    if scipy_sparse.issparse(matrix):
        return matrix.tocsr()
    return scipy_sparse.csr_matrix(np.asarray(matrix, dtype=float))
