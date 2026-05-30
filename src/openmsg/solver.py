"""Differentiable linear-algebra core for MSG homogenization.

The constrained MSG saddle-point system is solved with the TensorMesh /
``torch-sla`` differentiable sparse solver (canonical ``SparseMatrix.solve``)
or, optionally, a dense ``torch.linalg.solve``. Both paths support
``torch.autograd`` so gradients flow from the homogenized stiffness back to the
assembled matrices (and hence to material and geometry parameters).

MSG sign convention (kept fixed):

* ``H = D_he``
* ``E V0 = -H``
* constrained solve: ``[E G.T; G 0] [V0; Lambda] = [-H; 0]``
* ``Dbar = (D0 + V0.T @ H) / omega``
"""

from __future__ import annotations


def _torch():
    import torch

    return torch


def _as_dense(matrix, torch, *, dtype, device):
    """Return a dense ``[n, n]`` tensor for E given numpy / torch / SparseMatrix."""

    if isinstance(matrix, torch.Tensor):
        return matrix.to(dtype=dtype, device=device)
    if hasattr(matrix, "to_dense"):  # torch_sla SparseTensor / SparseMatrix
        return matrix.to_dense().to(dtype=dtype, device=device)
    import numpy as np

    return torch.as_tensor(np.asarray(matrix, dtype=float), dtype=dtype, device=device)


def _as_sparse(matrix, torch, *, dtype, device):
    """Return a ``tensormesh.sparse.SparseMatrix`` for E."""

    from tensormesh.sparse import SparseMatrix

    if isinstance(matrix, SparseMatrix):
        return matrix
    if isinstance(matrix, torch.Tensor):
        return SparseMatrix.from_dense(matrix.to(dtype=dtype, device=device))
    if hasattr(matrix, "to_dense"):  # generic SparseTensor
        return SparseMatrix.from_dense(matrix.to_dense().to(dtype=dtype, device=device))
    import numpy as np

    return SparseMatrix.from_dense(
        torch.as_tensor(np.asarray(matrix, dtype=float), dtype=dtype, device=device)
    )


def _as_2d(value, torch, *, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    import numpy as np

    return torch.as_tensor(np.asarray(value, dtype=float), dtype=dtype, device=device)


def _constraint_tensor(G, torch, *, dtype, device, n_dof):
    if G is None:
        return None
    Gt = _as_2d(G, torch, dtype=dtype, device=device)
    if Gt.numel() == 0:
        return None
    if Gt.ndim != 2 or Gt.shape[1] != n_dof:
        raise ValueError("G must have shape (n_constraints, E.shape[0])")
    return Gt


def solve_constrained(E, H, G=None, *, return_lagrange: bool = False):
    """Dense, differentiable constrained MSG solve via ``torch.linalg.solve``.

    Raises on a singular or rank-deficient (constrained) system. Unlike the
    former NumPy path there is no implicit least-squares fallback, so an
    ill-posed constraint set fails loudly instead of being silently solved.
    """

    torch = _torch()
    H = _as_2d(H, torch, dtype=torch.float64, device=None)
    dtype, device = H.dtype, H.device
    E_dense = _as_dense(E, torch, dtype=dtype, device=device)
    if E_dense.ndim != 2 or E_dense.shape[0] != E_dense.shape[1]:
        raise ValueError("E must be a square matrix")
    if H.ndim != 2 or H.shape[0] != E_dense.shape[0]:
        raise ValueError("H must have shape (E.shape[0], n_macro)")

    n_dof = E_dense.shape[0]
    Gt = _constraint_tensor(G, torch, dtype=dtype, device=device, n_dof=n_dof)

    if Gt is None:
        V0 = -torch.linalg.solve(E_dense, H)
        if return_lagrange:
            return V0, torch.zeros((0, H.shape[1]), dtype=dtype, device=device)
        return V0

    n_c = Gt.shape[0]
    zero_cc = torch.zeros((n_c, n_c), dtype=dtype, device=device)
    top = torch.cat([E_dense, Gt.transpose(0, 1)], dim=1)
    bottom = torch.cat([Gt, zero_cc], dim=1)
    block = torch.cat([top, bottom], dim=0)
    rhs = torch.cat([-H, torch.zeros((n_c, H.shape[1]), dtype=dtype, device=device)], dim=0)

    solution = torch.linalg.solve(block, rhs)
    V0 = solution[:n_dof, :]
    lagrange = solution[n_dof:, :]
    if return_lagrange:
        return V0, lagrange
    return V0


def solve_constrained_sparse(E, H, G=None, *, return_lagrange: bool = False, method: str = "lu"):
    """Sparse, differentiable constrained MSG solve via TensorMesh's solver.

    Builds the saddle-point matrix as a ``tensormesh.sparse.SparseMatrix`` and
    solves it with the canonical, autograd-aware ``SparseMatrix.solve``. The
    direct factorization raises on a singular or rank-deficient system; there is
    no least-squares fallback, so an ill-posed constraint set fails loudly.
    """

    torch = _torch()
    from tensormesh.sparse import SparseMatrix

    H = _as_2d(H, torch, dtype=torch.float64, device=None)
    dtype, device = H.dtype, H.device
    E_sparse = _as_sparse(E, torch, dtype=dtype, device=device)
    n_dof = int(E_sparse.shape[0])
    if int(E_sparse.shape[1]) != n_dof:
        raise ValueError("E must be a square matrix")
    if H.ndim != 2 or H.shape[0] != n_dof:
        raise ValueError("H must have shape (E.shape[0], n_macro)")

    Gt = _constraint_tensor(G, torch, dtype=dtype, device=device, n_dof=n_dof)

    if Gt is None:
        V0 = -E_sparse.solve(H, method=method)
        if return_lagrange:
            return V0, torch.zeros((0, H.shape[1]), dtype=dtype, device=device)
        return V0

    n_c = Gt.shape[0]
    G_sparse = SparseMatrix.from_dense(Gt)
    kkt = SparseMatrix.combine_matrix([[E_sparse, G_sparse.T], [G_sparse, None]])
    rhs = torch.cat([-H, torch.zeros((n_c, H.shape[1]), dtype=dtype, device=device)], dim=0)

    solution = kkt.solve(rhs, method=method)
    V0 = solution[:n_dof, :]
    lagrange = solution[n_dof:, :]
    if return_lagrange:
        return V0, lagrange
    return V0


def compute_effective_stiffness(D0, H, V0, *, omega=1.0):
    """Compute and symmetrize the homogenized stiffness ``Dbar`` (differentiable)."""

    torch = _torch()
    D0 = _as_2d(D0, torch, dtype=torch.float64, device=None)
    dtype, device = D0.dtype, D0.device
    H = _as_2d(H, torch, dtype=dtype, device=device)
    V0 = _as_2d(V0, torch, dtype=dtype, device=device)
    if D0.shape != (H.shape[1], H.shape[1]):
        raise ValueError("D0 must have shape (n_macro, n_macro)")
    if V0.shape != H.shape:
        raise ValueError("V0 must have the same shape as H")
    if isinstance(omega, torch.Tensor):
        omega_t = omega.to(dtype=dtype, device=device)
    else:
        omega_t = torch.as_tensor(float(omega), dtype=dtype, device=device)
    Dbar = (D0 + V0.transpose(0, 1) @ H) / omega_t
    return 0.5 * (Dbar + Dbar.transpose(0, 1))
