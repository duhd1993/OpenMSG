"""Bilinear Quad4 element quadrature."""

from __future__ import annotations

import numpy as np

from .common import surface_physical_gradients
from .types import QuadraturePoint


def quadrature(coords: np.ndarray, active_axes: tuple[int, int]) -> list[QuadraturePoint]:
    """Return 2x2 Gauss quadrature for a Quad4 SG element."""

    element_coords = np.asarray(coords, dtype=float)
    if element_coords.shape != (4, 3):
        raise ValueError("Quad4 element coordinates must have shape (4, 3)")
    active_coords = element_coords[:, active_axes]
    points: list[QuadraturePoint] = []
    a = 1.0 / np.sqrt(3.0)
    for xi in (-a, a):
        for eta in (-a, a):
            N = shape_functions(xi, eta)
            dN_dnat = shape_function_gradients_nat(xi, eta)
            dN_dactive, det_j = surface_physical_gradients(active_coords, dN_dnat, "Quad4")
            dN_dx = np.zeros((4, 3), dtype=float)
            dN_dx[:, active_axes] = dN_dactive
            points.append(QuadraturePoint(shape=N, dN_dx=dN_dx, dV=det_j))
    return points


def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Evaluate the four Quad4 shape functions."""

    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def shape_function_gradients_nat(xi: float, eta: float) -> np.ndarray:
    """Evaluate Quad4 shape function gradients in natural coordinates."""

    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )
