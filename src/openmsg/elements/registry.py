"""Element quadrature dispatch for SG meshes embedded in 3D space."""

from __future__ import annotations

import numpy as np

from . import hex8, line2, quad4, tet4, tri3
from .common import resolve_active_axes
from .types import QuadraturePoint


def iter_element_quadrature(
    element_type: str,
    coords: np.ndarray,
    active_axes: tuple[int, ...] | None = None,
) -> list[QuadraturePoint]:
    """Return quadrature data for a supported SG element.

    Lower-dimensional SG elements still live in 3D coordinate space. Their
    parametric gradients are embedded into the requested active axes so the
    standard 3D small-strain operator can be reused for the 3D fluctuation
    field.
    """

    if element_type == "hex8":
        return hex8.quadrature(coords)
    if element_type == "tet4":
        return tet4.quadrature(coords)
    if element_type == "quad4":
        return quad4.quadrature(coords, resolve_active_axes(active_axes, 2))
    if element_type == "tri3":
        return tri3.quadrature(coords, resolve_active_axes(active_axes, 2))
    if element_type == "line2":
        return line2.quadrature(coords, resolve_active_axes(active_axes, 1))
    raise ValueError(f"unsupported element_type {element_type!r}")
