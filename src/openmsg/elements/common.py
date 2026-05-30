"""Shared helpers for first-order SG element quadrature."""

from __future__ import annotations

import numpy as np


def resolve_active_axes(active_axes: tuple[int, ...] | None, dimension: int) -> tuple[int, ...]:
    """Resolve active physical axes for lower-dimensional SG elements."""

    if active_axes is None:
        if dimension == 2:
            return (1, 2)
        if dimension == 1:
            return (2,)
    axes = tuple(active_axes or ())
    if len(axes) != dimension:
        raise ValueError(f"active_axes must have length {dimension}")
    return axes


def surface_physical_gradients(
    active_coords: np.ndarray,
    dN_dnat: np.ndarray,
    element_name: str,
) -> tuple[np.ndarray, float]:
    """Map 2D natural gradients to active physical coordinates."""

    jacobian = active_coords.T @ dN_dnat
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"{element_name} element has non-positive Jacobian determinant {det_j}")
    return dN_dnat @ np.linalg.inv(jacobian), det_j
