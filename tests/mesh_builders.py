from __future__ import annotations

from typing import Iterable

import numpy as np

from openmsg.mesh import SolidMesh


def structured_hex_mesh(
    *,
    bounds: Iterable[Iterable[float]],
    cells: Iterable[int],
    default_material: str,
    cell_materials: Iterable[str] | None = None,
) -> SolidMesh:
    """Create a structured Hex8 mesh for tests."""

    bounds_array = np.asarray(list(bounds), dtype=float)
    if bounds_array.shape != (3, 2):
        raise ValueError("bounds must have shape (3, 2)")
    if np.any(bounds_array[:, 1] <= bounds_array[:, 0]):
        raise ValueError("each bounds row must be [min, max]")

    nx, ny, nz = (int(v) for v in cells)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("cells must be positive in all directions")

    xs = np.linspace(bounds_array[0, 0], bounds_array[0, 1], nx + 1)
    ys = np.linspace(bounds_array[1, 0], bounds_array[1, 1], ny + 1)
    zs = np.linspace(bounds_array[2, 0], bounds_array[2, 1], nz + 1)

    def node_id(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    nodes = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    elements: list[list[int]] = []
    material_ids: list[str] = []

    flat_materials: list[str] | None = None
    if cell_materials is not None:
        flat_materials = [str(v) for v in cell_materials]
        if len(flat_materials) != nx * ny * nz:
            raise ValueError("cell_materials length must equal nx * ny * nz")

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(
                    [
                        node_id(i, j, k),
                        node_id(i + 1, j, k),
                        node_id(i + 1, j + 1, k),
                        node_id(i, j + 1, k),
                        node_id(i, j, k + 1),
                        node_id(i + 1, j, k + 1),
                        node_id(i + 1, j + 1, k + 1),
                        node_id(i, j + 1, k + 1),
                    ]
                )
                flat_index = i + nx * (j + ny * k)
                material_ids.append(
                    flat_materials[flat_index] if flat_materials is not None else str(default_material)
                )

    return SolidMesh(
        nodes=nodes,
        elements=np.asarray(elements, dtype=int),
        material_ids=tuple(material_ids),
        element_type="hex8",
    )
