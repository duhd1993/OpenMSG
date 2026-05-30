"""Constraint tensor construction for MSG fluctuation fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from openmsg.mesh import SolidMesh

AXES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


@dataclass
class _UnionFind:
    parent: list[int]

    @classmethod
    def create(cls, n: int) -> "_UnionFind":
        return cls(parent=list(range(n)))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def mean_zero_constraints_tensor(
    mesh: SolidMesh,
    node_weights: object | None = None,
    *,
    dtype: object | None = None,
    device: object | None = None,
) -> object:
    """Return mean-zero constraints for the fluctuation displacement."""

    dtype = dtype or torch.float64
    weights = (
        torch.ones(mesh.n_nodes, dtype=dtype, device=device)
        if node_weights is None
        else torch.as_tensor(node_weights, dtype=dtype, device=device)
    )
    if weights.shape != (mesh.n_nodes,):
        raise ValueError("node_weights must have shape (n_nodes,)")
    G = torch.zeros((3, mesh.n_dof), dtype=weights.dtype, device=weights.device)
    node = torch.arange(mesh.n_nodes, device=weights.device)
    for comp in range(3):
        G[comp, 3 * node + comp] = weights
    return G


def periodic_constraints_tensor(
    mesh: SolidMesh,
    axes: Iterable[str | int] = ("x", "y", "z"),
    *,
    tol: float = 1e-9,
    dtype: object | None = None,
    device: object | None = None,
) -> object:
    """Return independent periodic equality constraints for opposite SG faces."""

    dtype = dtype or torch.float64
    axis_indices = [_axis_index(axis) for axis in axes]
    if not axis_indices:
        return _empty_constraints(mesh.n_dof, dtype=dtype, device=device)

    uf = _UnionFind.create(mesh.n_nodes)
    coords = _mesh_nodes_tensor(mesh, dtype=dtype, device=device)
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values

    for axis in axis_indices:
        other_axes = tuple(i for i in range(3) if i != axis)
        min_nodes = _matching_nodes(coords[:, axis], mins[axis], tol)
        max_nodes = _matching_nodes(coords[:, axis], maxs[axis], tol)
        lookup = {_coord_key(coords[node], other_axes, tol): int(node) for node in min_nodes}
        for node in max_nodes:
            key = _coord_key(coords[node], other_axes, tol)
            if key not in lookup:
                raise ValueError(f"could not find periodic partner for node {node} on axis {axis}")
            uf.union(int(node), lookup[key])

    rows: list[object] = []
    for node in range(mesh.n_nodes):
        root = uf.find(node)
        if root == node:
            continue
        for comp in range(3):
            row = torch.zeros(mesh.n_dof, dtype=coords.dtype, device=coords.device)
            row[3 * node + comp] = 1.0
            row[3 * root + comp] = -1.0
            rows.append(row)
    if not rows:
        return _empty_constraints(mesh.n_dof, dtype=coords.dtype, device=coords.device)
    return torch.stack(rows, dim=0)


def rotation_zero_constraints_tensor(
    mesh: SolidMesh,
    pairs: Iterable[Iterable[str | int]] | None = None,
    *,
    tol: float = 1e-12,
    dtype: object | None = None,
    device: object | None = None,
) -> object:
    """Return average infinitesimal-rotation constraints for ``w``.

    Each pair ``(i, j)`` adds ``∫(dw_i/dx_j - dw_j/dx_i) dOmega = 0``.
    Zero rows are dropped, which makes the constraint safe for lower-dimensional
    SGs where one derivative in a requested pair may not exist.
    """

    dtype = dtype or torch.float64
    axis_pairs = [_axis_pair(pair) for pair in pairs] if pairs is not None else [(0, 1), (0, 2), (1, 2)]
    grad_integral = _node_gradient_integral(mesh, dtype=dtype, device=device)  # [n_nodes, 3]: ∫ dN_node/dx_axis dOmega
    node = torch.arange(mesh.n_nodes, device=grad_integral.device)
    rows: list[object] = []
    for axis_i, axis_j in axis_pairs:
        row = torch.zeros(mesh.n_dof, dtype=grad_integral.dtype, device=grad_integral.device)
        row[3 * node + axis_i] += grad_integral[:, axis_j]
        row[3 * node + axis_j] -= grad_integral[:, axis_i]
        if float(torch.linalg.vector_norm(row).detach().cpu()) > tol:
            rows.append(row)
    if not rows:
        return _empty_constraints(mesh.n_dof, dtype=grad_integral.dtype, device=grad_integral.device)
    return torch.stack(rows, dim=0)


def _node_gradient_integral(
    mesh: SolidMesh,
    *,
    dtype: object | None = None,
    device: object | None = None,
) -> object:
    """Return ``∫ dN_node/dx_axis dOmega`` per global node via TensorMesh quadrature."""

    from openmsg.assembly import tensormesh_quadrature

    dtype = dtype or torch.float64
    _, blocks = tensormesh_quadrature(mesh, dtype=dtype, device=device)
    sample = blocks[0].jxw
    integral = torch.zeros((mesh.n_nodes, 3), dtype=sample.dtype, device=sample.device)
    for block in blocks:
        # contrib[e, node, axis] = sum_q dN_dx[e, q, node, axis] * jxw[e, q]
        contrib = torch.einsum("eqbd,eq->ebd", block.dN_dx, block.jxw)
        integral = integral.index_add(0, block.conn.reshape(-1), contrib.reshape(-1, 3))
    return integral


def build_constraint_tensor(
    mesh: SolidMesh,
    specs: Iterable[str | dict[str, object]] | None,
    *,
    node_weights: object | None = None,
    dtype: object | None = None,
    device: object | None = None,
) -> object | None:
    """Build a combined torch constraint matrix from config-like specs."""

    if specs is None:
        return None

    dtype = dtype or (node_weights.dtype if isinstance(node_weights, torch.Tensor) else torch.float64)
    blocks: list[object] = []
    for spec in specs:
        if isinstance(spec, str):
            kind = spec
            data: dict[str, object] = {}
        else:
            kind = str(spec.get("type", ""))
            data = spec
        kind = kind.lower()
        if kind == "mean_zero":
            blocks.append(
                mean_zero_constraints_tensor(mesh, node_weights=node_weights, dtype=dtype, device=device)
            )
        elif kind == "periodic":
            axes = data.get("axes", ("x", "y", "z"))
            blocks.append(
                periodic_constraints_tensor(mesh, axes=axes, dtype=dtype, device=device)  # type: ignore[arg-type]
            )
        elif kind in {"rotation_zero", "beam_twist"}:
            pairs = data.get("pairs")
            if kind == "beam_twist" and pairs is None:
                pairs = [["y", "z"]]
            blocks.append(
                rotation_zero_constraints_tensor(mesh, pairs=pairs, dtype=dtype, device=device)  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"unsupported constraint type {kind!r}")

    if not blocks:
        return None
    nonempty_blocks = [block for block in blocks if block.numel() > 0]
    if not nonempty_blocks:
        return None
    G = torch.cat(nonempty_blocks, dim=0)
    return G if G.numel() else None


def _axis_index(axis: str | int) -> int:
    key: str | int = axis.lower() if isinstance(axis, str) else axis
    if key not in AXES:
        raise ValueError(f"unsupported axis {axis!r}")
    return AXES[key]


def _axis_pair(pair: Iterable[str | int]) -> tuple[int, int]:
    values = tuple(pair)
    if len(values) != 2:
        raise ValueError("rotation constraint pairs must contain exactly two axes")
    axis_i = _axis_index(values[0])
    axis_j = _axis_index(values[1])
    if axis_i == axis_j:
        raise ValueError("rotation constraint pair axes must be distinct")
    return (axis_i, axis_j) if axis_i < axis_j else (axis_j, axis_i)


def _empty_constraints(n_dof: int, *, dtype: object, device: object | None):
    return torch.zeros((0, n_dof), dtype=dtype, device=device)


def _mesh_nodes_tensor(mesh: SolidMesh, *, dtype: object, device: object | None):
    return torch.as_tensor(mesh.nodes, dtype=dtype, device=device)


def _matching_nodes(values: object, target: object, tol: float) -> list[int]:
    mask = torch.isclose(values, target, rtol=0.0, atol=tol)
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def _coord_key(coord: object, axes: tuple[int, ...], tol: float) -> tuple[int, ...]:
    scale = max(tol, 1e-15)
    return tuple(int(round(float(coord[axis].detach().cpu()) / scale)) for axis in axes)
