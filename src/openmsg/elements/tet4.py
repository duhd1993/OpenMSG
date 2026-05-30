"""Linear Tet4 element quadrature."""

from __future__ import annotations

import numpy as np

from .types import QuadraturePoint


def quadrature(coords: np.ndarray) -> list[QuadraturePoint]:
    """Return one-point quadrature for a Tet4 element."""

    element_coords = np.asarray(coords, dtype=float)
    if element_coords.shape != (4, 3):
        raise ValueError("Tet4 element coordinates must have shape (4, 3)")
    N = np.full(4, 0.25, dtype=float)
    dN_dnat = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    jacobian = element_coords.T @ dN_dnat
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"Tet4 element has non-positive Jacobian determinant {det_j}")
    dN_dx = dN_dnat @ np.linalg.inv(jacobian)
    return [QuadraturePoint(shape=N, dN_dx=dN_dx, dV=det_j / 6.0)]
