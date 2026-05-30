"""Linear Line2 element quadrature."""

from __future__ import annotations

import numpy as np

from .types import QuadraturePoint


def quadrature(coords: np.ndarray, active_axes: tuple[int]) -> list[QuadraturePoint]:
    """Return two-point Gauss quadrature for a Line2 SG element."""

    element_coords = np.asarray(coords, dtype=float)
    if element_coords.shape != (2, 3):
        raise ValueError("Line2 element coordinates must have shape (2, 3)")
    active_coord = element_coords[:, active_axes[0]]
    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    jacobian = float(active_coord @ dN_dxi)
    if jacobian <= 0.0:
        raise ValueError(f"Line2 element has non-positive Jacobian determinant {jacobian}")

    points: list[QuadraturePoint] = []
    a = 1.0 / np.sqrt(3.0)
    for xi in (-a, a):
        N = np.array([(1.0 - xi) * 0.5, (1.0 + xi) * 0.5], dtype=float)
        dN_dx = np.zeros((2, 3), dtype=float)
        dN_dx[:, active_axes[0]] = dN_dxi / jacobian
        points.append(QuadraturePoint(shape=N, dN_dx=dN_dx, dV=jacobian))
    return points
