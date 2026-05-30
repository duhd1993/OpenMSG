"""Linear Tri3 element quadrature."""

from __future__ import annotations

import numpy as np

from .common import surface_physical_gradients
from .types import QuadraturePoint


def quadrature(coords: np.ndarray, active_axes: tuple[int, int]) -> list[QuadraturePoint]:
    """Return one-point quadrature for a Tri3 SG element."""

    element_coords = np.asarray(coords, dtype=float)
    if element_coords.shape != (3, 3):
        raise ValueError("Tri3 element coordinates must have shape (3, 3)")
    active_coords = element_coords[:, active_axes]
    N = np.full(3, 1.0 / 3.0, dtype=float)
    dN_dnat = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    dN_dactive, det_j = surface_physical_gradients(active_coords, dN_dnat, "Tri3")
    dN_dx = np.zeros((3, 3), dtype=float)
    dN_dx[:, active_axes] = dN_dactive
    return [QuadraturePoint(shape=N, dN_dx=dN_dx, dV=0.5 * det_j)]
