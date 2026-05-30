"""Trilinear Hex8 element quadrature."""

from __future__ import annotations

import math

import numpy as np

from .types import QuadraturePoint

REFERENCE_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)


def quadrature(coords: np.ndarray) -> list[QuadraturePoint]:
    """Return 2x2x2 Gauss quadrature for a Hex8 element."""

    points: list[QuadraturePoint] = []
    for xi, eta, zeta, weight in gauss_points_2x2x2():
        N = shape_functions(xi, eta, zeta)
        dN_dnat = shape_function_gradients_nat(xi, eta, zeta)
        dN_dx, det_j = physical_gradients(coords, dN_dnat)
        points.append(QuadraturePoint(shape=N, dN_dx=dN_dx, dV=det_j * weight))
    return points


def gauss_points_2x2x2() -> list[tuple[float, float, float, float]]:
    """Return 2x2x2 Gauss points as ``(xi, eta, zeta, weight)`` tuples."""

    a = 1.0 / math.sqrt(3.0)
    points: list[tuple[float, float, float, float]] = []
    for xi in (-a, a):
        for eta in (-a, a):
            for zeta in (-a, a):
                points.append((xi, eta, zeta, 1.0))
    return points


def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate the eight Hex8 shape functions."""

    natural = np.array([xi, eta, zeta], dtype=float)
    return 0.125 * np.prod(1.0 + REFERENCE_SIGNS * natural, axis=1)


def shape_function_gradients_nat(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate Hex8 shape function gradients in natural coordinates."""

    dN = np.empty((8, 3), dtype=float)
    for a, (sx, sy, sz) in enumerate(REFERENCE_SIGNS):
        dN[a, 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN[a, 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        dN[a, 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return dN


def physical_gradients(coords: np.ndarray, dN_dnat: np.ndarray) -> tuple[np.ndarray, float]:
    """Map natural gradients to physical coordinates."""

    element_coords = np.asarray(coords, dtype=float)
    if element_coords.shape != (8, 3):
        raise ValueError("Hex8 element coordinates must have shape (8, 3)")
    jacobian = element_coords.T @ dN_dnat
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError(f"Hex8 element has non-positive Jacobian determinant {det_j}")
    dN_dx = dN_dnat @ np.linalg.inv(jacobian)
    return dN_dx, det_j
