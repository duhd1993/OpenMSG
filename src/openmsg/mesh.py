"""Mesh containers and input parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class ElementBlock:
    """Connectivity and materials for one SG element type."""

    element_type: str
    elements: object
    material: object
    material_ids: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        kind = _normalize_element_type(self.element_type)
        conn = _normalize_connectivity(self.elements, kind)
        materials = _normalize_block_material(self.material, n_elements=conn.shape[0])
        object.__setattr__(self, "element_type", kind)
        object.__setattr__(self, "elements", conn)
        object.__setattr__(self, "material_ids", materials)

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])


@dataclass(frozen=True)
class SolidMesh:
    """A Structure Genome mesh embedded in 3D coordinate space."""

    nodes: np.ndarray
    elements: object
    active_axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        blocks = _merge_element_blocks(_normalize_element_blocks(self.elements))
        sg_dimension = _infer_sg_dimension(tuple(block.element_type for block in blocks))
        active_axes = _normalize_active_axes_by_dimension(self.active_axes, sg_dimension)
        nodes = _as_3d_points(self.nodes, active_axes=active_axes)
        _validate_element_blocks(blocks, n_nodes=len(nodes))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", blocks)
        object.__setattr__(self, "active_axes", active_axes)
        object.__setattr__(self, "_sg_dimension", sg_dimension)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return sum(block.n_elements for block in self.element_blocks)

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def sg_dimension(self) -> int:
        return int(self._sg_dimension)

    @property
    def element_types(self) -> tuple[str, ...]:
        return tuple(block.element_type for block in self.element_blocks)

    @property
    def element_blocks(self) -> tuple[ElementBlock, ...]:
        return self.elements

    @property
    def material_ids(self) -> tuple[str, ...]:
        return tuple(material for block in self.element_blocks for material in block.material_ids)


def mesh_from_config(config: dict[str, object], *, base_dir: str | Path | None = None) -> SolidMesh:
    """Create a mesh from a JSON mesh block."""

    kind = str(config.get("type", "explicit")).lower()
    if kind in {"explicit", "solid", "sg"}:
        return _explicit_mesh_from_config(config)
    if kind in SUPPORTED_ELEMENT_NODES or kind == "mixed":
        raise ValueError(
            "mesh element types now belong to each element block; "
            "omit mesh.type or use mesh.type='explicit'"
        )
    if kind == "meshio":
        return _meshio_mesh_from_config(config, base_dir=base_dir)
    raise ValueError(f"unsupported mesh type {kind!r}")


def _explicit_mesh_from_config(config: dict[str, object]) -> SolidMesh:
    active_axes = _parse_active_axes(config.get("active_axes"))
    blocks = _merge_element_blocks(_normalize_element_blocks(config["elements"]))
    sg_dimension = _infer_sg_dimension(tuple(block.element_type for block in blocks))
    active_axes = _normalize_active_axes_by_dimension(active_axes, sg_dimension)
    return SolidMesh(
        nodes=_as_3d_points(config["nodes"], active_axes=active_axes),
        elements=blocks,
        active_axes=active_axes,
    )


def _meshio_mesh_from_config(
    config: dict[str, object], *, base_dir: str | Path | None = None
) -> SolidMesh:
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio mesh input requires dependency meshio") from exc

    path = Path(str(config["path"]))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    meshio_mesh = meshio.read(path)
    cell_types = _meshio_cell_types_from_config(config)
    active_axes = _parse_active_axes(config.get("active_axes"))
    material_data = str(config.get("material_data", "material"))
    default_material = str(config.get("default_material", "material"))

    blocks_for_mesh: list[ElementBlock] = []
    for cell_type in cell_types:
        element_type = _normalize_element_type(
            MESHIO_TO_OPENMSG_ELEMENT.get(cell_type, cell_type)
        )
        blocks = [block for block in meshio_mesh.cells if block.type == cell_type]
        if not blocks:
            raise ValueError(f"mesh file {path} has no {cell_type!r} cell block")
        elements = (
            np.vstack([block.data for block in blocks])
            if len(blocks) > 1
            else np.asarray(blocks[0].data, dtype=int)
        )
        material_ids = _meshio_material_ids(
            meshio_mesh=meshio_mesh,
            cell_type=cell_type,
            material_data=material_data,
            default_material=default_material,
            material_map=config.get("material_map"),
        )
        if len(material_ids) != len(elements):
            raise ValueError(
                f"meshio material data length does not match selected {cell_type} elements"
            )
        blocks_for_mesh.append(
            ElementBlock(
                element_type=element_type,
                elements=elements,
                material=material_ids,
            )
        )

    sg_dimension = _infer_sg_dimension(tuple(block.element_type for block in blocks_for_mesh))
    active_axes = _normalize_active_axes_by_dimension(active_axes, sg_dimension)
    return SolidMesh(
        nodes=_as_3d_points(meshio_mesh.points, active_axes=active_axes),
        elements=blocks_for_mesh,
        active_axes=active_axes,
    )


def _meshio_cell_types_from_config(config: dict[str, object]) -> tuple[str, ...]:
    raw = config.get("cell_types", config.get("cell_type", "hexahedron"))
    if isinstance(raw, str):
        return (raw,)
    cell_types = tuple(str(value) for value in raw)  # type: ignore[union-attr]
    if not cell_types:
        raise ValueError("meshio input requires at least one cell_type")
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("meshio cell_types must be unique")
    return cell_types


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

    lookup = {
        str(key): str(value)
        for key, value in (material_map or {}).items()  # type: ignore[union-attr]
    }
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


def _normalize_element_blocks(elements: object) -> tuple[ElementBlock, ...]:
    if isinstance(elements, ElementBlock):
        return (elements,)
    if not isinstance(elements, (list, tuple)):
        raise ValueError("elements must be a non-empty list of element blocks")
    if not elements:
        raise ValueError("elements must contain at least one element block")
    return tuple(_element_block_from_object(item) for item in elements)


def _element_block_from_object(item: object) -> ElementBlock:
    if isinstance(item, ElementBlock):
        return item
    if not isinstance(item, dict):
        raise ValueError("each element block must be an object")
    if "type" not in item and "element_type" not in item:
        raise ValueError("each element block must include an element type")
    element_type = str(item.get("type", item.get("element_type")))
    if "connectivity" in item:
        elements = item["connectivity"]
    elif "elements" in item:
        elements = item["elements"]
    else:
        raise ValueError("each element block must include connectivity")
    if "material" in item:
        material = item["material"]
    elif "materials" in item:
        material = item["materials"]
    else:
        raise ValueError("each element block must include material or materials")
    return ElementBlock(element_type=element_type, elements=elements, material=material)


def _normalize_connectivity(elements: object, element_type: str) -> np.ndarray:
    kind = _normalize_element_type(element_type)
    conn = np.asarray(elements, dtype=int)
    expected_nodes = SUPPORTED_ELEMENT_NODES[kind]
    if conn.ndim != 2 or conn.shape[1] != expected_nodes:
        raise ValueError(f"{kind} elements must have shape (n_elements, {expected_nodes})")
    if conn.shape[0] == 0:
        raise ValueError(f"{kind} element block must not be empty")
    return conn


def _normalize_block_material(material: object, *, n_elements: int) -> tuple[str, ...]:
    if isinstance(material, str):
        return (material,) * n_elements
    if np.isscalar(material):
        return (str(material),) * n_elements
    materials = tuple(str(value) for value in material)  # type: ignore[arg-type]
    if len(materials) != n_elements:
        raise ValueError("block material list length must match number of elements")
    return materials


def _merge_element_blocks(
    blocks: tuple[ElementBlock, ...] | list[ElementBlock],
) -> tuple[ElementBlock, ...]:
    if not blocks:
        raise ValueError("mesh must contain at least one element block")
    grouped_elements: dict[str, list[np.ndarray]] = {}
    grouped_materials: dict[str, list[str]] = {}
    order: list[str] = []
    for block in blocks:
        if block.element_type not in grouped_elements:
            grouped_elements[block.element_type] = []
            grouped_materials[block.element_type] = []
            order.append(block.element_type)
        grouped_elements[block.element_type].append(block.elements)
        grouped_materials[block.element_type].extend(block.material_ids)
    return tuple(
        ElementBlock(
            element_type=kind,
            elements=np.vstack(grouped_elements[kind]),
            material=tuple(grouped_materials[kind]),
        )
        for kind in order
    )


def _validate_element_blocks(blocks: tuple[ElementBlock, ...], *, n_nodes: int) -> None:
    for block in blocks:
        if np.min(block.elements) < 0 or np.max(block.elements) >= n_nodes:
            raise ValueError("element connectivity references nodes outside the mesh")


def _infer_sg_dimension(element_types: tuple[str, ...]) -> int:
    if not element_types:
        raise ValueError("mesh must contain at least one element type")
    dimensions = {
        ELEMENT_DIMENSIONS[_normalize_element_type(element_type)]
        for element_type in element_types
    }
    if len(dimensions) != 1:
        raise ValueError("mixed meshes may only combine elements with the same SG dimension")
    return dimensions.pop()


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


def _normalize_active_axes_by_dimension(active_axes: object, sg_dimension: int) -> tuple[int, ...]:
    if active_axes is None:
        if sg_dimension == 3:
            return (0, 1, 2)
        if sg_dimension == 2:
            return (1, 2)
        if sg_dimension == 1:
            return (2,)
        raise ValueError(f"unsupported SG dimension {sg_dimension}")
    axes = _parse_active_axes(active_axes)
    assert axes is not None
    if len(axes) != sg_dimension:
        raise ValueError(f"SG dimension {sg_dimension} requires {sg_dimension} active axes")
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
