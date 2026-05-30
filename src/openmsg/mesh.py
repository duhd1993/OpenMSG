"""Mesh containers and input parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_ELEMENT_NODES = {
    "hex8": 8,
    "tet4": 4,
    "quad4": 4,
    "tri3": 3,
    "line2": 2,
}

ELEMENT_DIMENSIONS = {
    "hex8": 3,
    "tet4": 3,
    "quad4": 2,
    "tri3": 2,
    "line2": 1,
}

MESHIO_TO_OPENMSG_ELEMENT = {
    "hexahedron": "hex8",
    "tetra": "tet4",
    "quad": "quad4",
    "triangle": "tri3",
    "line": "line2",
}

OPENMSG_TO_MESHIO_ELEMENT = {value: key for key, value in MESHIO_TO_OPENMSG_ELEMENT.items()}


@dataclass(frozen=True)
class SolidMesh:
    """A Structure Genome mesh embedded in 3D coordinate space."""

    nodes: np.ndarray
    elements: np.ndarray
    material_ids: tuple[str, ...]
    element_type: str = "hex8"
    active_axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        elements = np.asarray(self.elements, dtype=int)
        element_type = _normalize_element_type(self.element_type)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n_nodes, 3)")
        expected_nodes = SUPPORTED_ELEMENT_NODES[element_type]
        if elements.ndim != 2 or elements.shape[1] != expected_nodes:
            raise ValueError(f"{element_type} elements must have shape (n_elements, {expected_nodes})")
        if len(self.material_ids) != len(elements):
            raise ValueError("material_ids length must match number of elements")
        if np.min(elements) < 0 or np.max(elements) >= len(nodes):
            raise ValueError("element connectivity references nodes outside the mesh")
        active_axes = _normalize_active_axes(self.active_axes, element_type)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "material_ids", tuple(str(v) for v in self.material_ids))
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "active_axes", active_axes)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def sg_dimension(self) -> int:
        return ELEMENT_DIMENSIONS[self.element_type]


def mesh_from_config(config: dict[str, object], *, base_dir: str | Path | None = None) -> SolidMesh:
    """Create a mesh from a JSON mesh block."""

    kind = str(config.get("type", "hex8")).lower()
    if kind in {"hex8", "tet4"}:
        return _explicit_mesh_from_config(config, kind)
    if kind in {"quad4", "tri3", "line2"}:
        return _explicit_mesh_from_config(config, kind)
    if kind == "meshio":
        return _meshio_mesh_from_config(config, base_dir=base_dir)
    raise ValueError(f"unsupported mesh type {kind!r}")


def _explicit_mesh_from_config(config: dict[str, object], kind: str) -> SolidMesh:
    active_axes = _parse_active_axes(config.get("active_axes"))
    elements_raw = config["elements"]
    elements: list[list[int]] = []
    material_ids: list[str] = []
    for item in elements_raw:  # type: ignore[assignment]
        if isinstance(item, dict):
            elements.append([int(v) for v in item["nodes"]])
            material_ids.append(str(item["material"]))
        else:
            raise ValueError(f"explicit {kind} elements must be objects with nodes and material")
    return SolidMesh(
        nodes=_as_3d_points(config["nodes"], active_axes=_normalize_active_axes(active_axes, kind)),
        elements=np.asarray(elements),
        material_ids=material_ids,
        element_type=kind,
        active_axes=active_axes,
    )


def _meshio_mesh_from_config(config: dict[str, object], *, base_dir: str | Path | None = None) -> SolidMesh:
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio mesh input requires dependency meshio") from exc

    path = Path(str(config["path"]))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    meshio_mesh = meshio.read(path)
    cell_type = str(config.get("cell_type", "hexahedron"))
    element_type = _normalize_element_type(str(config.get("element_type", MESHIO_TO_OPENMSG_ELEMENT.get(cell_type, cell_type))))
    active_axes = _parse_active_axes(config.get("active_axes"))
    blocks = [block for block in meshio_mesh.cells if block.type == cell_type]
    if not blocks:
        raise ValueError(f"mesh file {path} has no {cell_type!r} cell block")
    if len(blocks) > 1:
        elements = np.vstack([block.data for block in blocks])
    else:
        elements = np.asarray(blocks[0].data, dtype=int)

    material_data = str(config.get("material_data", "material"))
    default_material = str(config.get("default_material", "material"))
    material_ids = _meshio_material_ids(
        meshio_mesh=meshio_mesh,
        cell_type=cell_type,
        material_data=material_data,
        default_material=default_material,
        material_map=config.get("material_map"),
    )
    if len(material_ids) != len(elements):
        raise ValueError(f"meshio material data length does not match selected {cell_type} elements")
    return SolidMesh(
        nodes=_as_3d_points(meshio_mesh.points, active_axes=_normalize_active_axes(active_axes, element_type)),
        elements=elements,
        material_ids=tuple(material_ids),
        element_type=element_type,
        active_axes=active_axes,
    )


def _meshio_material_ids(
    *,
    meshio_mesh: object,
    cell_type: str,
    material_data: str,
    default_material: str,
    material_map: object,
) -> list[str]:
    cell_data = getattr(meshio_mesh, "cell_data", {})
    blocks = getattr(meshio_mesh, "cells", [])
    raw_values: list[object] = []
    if material_data in cell_data:
        for block, values in zip(blocks, cell_data[material_data]):
            if block.type == cell_type:
                raw_values.extend(np.asarray(values).tolist())
    if not raw_values:
        selected_count = sum(len(block.data) for block in blocks if block.type == cell_type)
        return [default_material] * selected_count

    lookup = {str(key): str(value) for key, value in (material_map or {}).items()}  # type: ignore[union-attr]
    material_ids: list[str] = []
    for value in raw_values:
        key = str(value)
        material_ids.append(lookup.get(key, key))
    return material_ids


def _normalize_element_type(element_type: str) -> str:
    key = element_type.lower()
    if key in MESHIO_TO_OPENMSG_ELEMENT:
        key = MESHIO_TO_OPENMSG_ELEMENT[key]
    if key not in SUPPORTED_ELEMENT_NODES:
        supported = ", ".join(sorted(SUPPORTED_ELEMENT_NODES))
        raise ValueError(f"unsupported element_type {element_type!r}; supported: {supported}")
    return key


def _as_3d_points(value: object, *, active_axes: tuple[int, ...] | None = None) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {1, 2, 3}:
        raise ValueError("nodes must have shape (n_nodes, 1), (n_nodes, 2), or (n_nodes, 3)")
    if points.shape[1] == 3:
        return points
    axes = active_axes if active_axes is not None else tuple(range(points.shape[1]))
    if len(axes) != points.shape[1]:
        raise ValueError("lower-dimensional node coordinates must match active_axes length")
    padded = np.zeros((points.shape[0], 3), dtype=float)
    padded[:, axes] = points
    return padded


def _normalize_active_axes(active_axes: object, element_type: str) -> tuple[int, ...]:
    sg_dimension = ELEMENT_DIMENSIONS[element_type]
    if active_axes is None:
        if sg_dimension == 3:
            return (0, 1, 2)
        if sg_dimension == 2:
            return (1, 2)
        if sg_dimension == 1:
            return (2,)
    axes = _parse_active_axes(active_axes)
    assert axes is not None
    if len(axes) != sg_dimension:
        raise ValueError(f"{element_type} requires {sg_dimension} active axes")
    if len(set(axes)) != len(axes) or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("active_axes must be unique axes chosen from x, y, z")
    return axes


def _parse_active_axes(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    axis_map = {"x": 0, "y": 1, "z": 2, "1": 0, "2": 1, "3": 2, 0: 0, 1: 1, 2: 2}
    axes = []
    for axis in value:  # type: ignore[union-attr]
        key = axis.lower() if isinstance(axis, str) else axis
        if key not in axis_map:
            raise ValueError(f"unsupported active axis {axis!r}")
        axes.append(axis_map[key])
    return tuple(axes)
