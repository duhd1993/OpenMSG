"""Voigt convention and small-strain operators."""

from __future__ import annotations

import numpy as np

VOIGT_ORDER = ("11", "22", "33", "23", "13", "12")


def gradient_to_voigt_strain(grad_u: np.ndarray) -> np.ndarray:
    """Convert a displacement gradient to engineering-shear Voigt strain."""

    grad = np.asarray(grad_u, dtype=float)
    if grad.shape != (3, 3):
        raise ValueError(f"grad_u must have shape (3, 3), got {grad.shape}")
    return np.array(
        [
            grad[0, 0],
            grad[1, 1],
            grad[2, 2],
            grad[1, 2] + grad[2, 1],
            grad[0, 2] + grad[2, 0],
            grad[0, 1] + grad[1, 0],
        ],
        dtype=float,
    )


def strain_displacement_matrix(dN_dx: np.ndarray) -> np.ndarray:
    """Return the small-strain matrix for nodal gradients.

    Parameters
    ----------
    dN_dx:
        Array of shape ``(n_nodes, 3)``. Row ``a`` stores
        ``[dN_a/dx, dN_a/dy, dN_a/dz]``.
    """

    grads = np.asarray(dN_dx, dtype=float)
    if grads.ndim != 2 or grads.shape[1] != 3:
        raise ValueError("dN_dx must have shape (n_nodes, 3)")

    n_nodes = grads.shape[0]
    B = np.zeros((6, 3 * n_nodes), dtype=float)
    for a, (dNx, dNy, dNz) in enumerate(grads):
        c = 3 * a
        B[0, c + 0] = dNx
        B[1, c + 1] = dNy
        B[2, c + 2] = dNz
        B[3, c + 1] = dNz
        B[3, c + 2] = dNy
        B[4, c + 0] = dNz
        B[4, c + 2] = dNx
        B[5, c + 0] = dNy
        B[5, c + 1] = dNx
    return B


def element_dof_indices(connectivity: np.ndarray) -> np.ndarray:
    """Return global displacement dof indices for an element connectivity."""

    conn = np.asarray(connectivity, dtype=int)
    dofs = np.empty(3 * len(conn), dtype=int)
    for local_index, node in enumerate(conn):
        dofs[3 * local_index : 3 * local_index + 3] = (3 * node, 3 * node + 1, 3 * node + 2)
    return dofs
