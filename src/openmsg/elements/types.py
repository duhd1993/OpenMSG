"""Shared element data containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadraturePoint:
    """Shape values, embedded physical gradients, and integration weight."""

    shape: np.ndarray
    dN_dx: np.ndarray
    dV: float
