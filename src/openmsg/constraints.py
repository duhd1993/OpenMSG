"""Constraint matrix construction for MSG fluctuation fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from openmsg.mesh import HexMesh

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


def mean_zero_constraints(mesh: HexMesh, node_weights: np.ndarray | None = None) -> np.ndarray:
    """Return mean-zero constraints for the fluctuation displacement."""

    weights = np.ones(mesh.n_nodes, dtype=float) if node_weights is None else np.asarray(node_weights, dtype=float)
    if weights.shape != (mesh.n_nodes,):
        raise ValueError("node_weights must have shape (n_nodes,)")
    G = np.zeros((3, mesh.n_dof), dtype=float)
    for node, weight in enumerate(weights):
        for comp in range(3):
            G[comp, 3 * node + comp] = weight
    return G


def periodic_constraints(mesh: HexMesh, axes: Iterable[str | int] = ("x", "y", "z"), *, tol: float = 1e-9) -> np.ndarray:
    """Return independent periodic equality constraints for opposite SG faces."""

    axis_indices = [_axis_index(axis) for axis in axes]
    if not axis_indices:
        return np.zeros((0, mesh.n_dof), dtype=float)

    uf = _UnionFind.create(mesh.n_nodes)
    coords = mesh.nodes
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)

    for axis in axis_indices:
        other_axes = tuple(i for i in range(3) if i != axis)
        min_nodes = np.where(np.isclose(coords[:, axis], mins[axis], atol=tol, rtol=0.0))[0]
        max_nodes = np.where(np.isclose(coords[:, axis], maxs[axis], atol=tol, rtol=0.0))[0]
        lookup = {_coord_key(coords[node], other_axes, tol): int(node) for node in min_nodes}
        for node in max_nodes:
            key = _coord_key(coords[node], other_axes, tol)
            if key not in lookup:
                raise ValueError(f"could not find periodic partner for node {node} on axis {axis}")
            uf.union(int(node), lookup[key])

    rows: list[np.ndarray] = []
    for node in range(mesh.n_nodes):
        root = uf.find(node)
        if root == node:
            continue
        for comp in range(3):
            row = np.zeros(mesh.n_dof, dtype=float)
            row[3 * node + comp] = 1.0
            row[3 * root + comp] = -1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, mesh.n_dof), dtype=float)
    return np.vstack(rows)


def rotation_zero_constraints(
    mesh: HexMesh,
    pairs: Iterable[Iterable[str | int]] | None = None,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return average infinitesimal-rotation constraints for ``w``.

    Each pair ``(i, j)`` adds ``∫(dw_i/dx_j - dw_j/dx_i) dOmega = 0``.
    Zero rows are dropped, which makes the constraint safe for lower-dimensional
    SGs where one derivative in a requested pair may not exist.
    """

    axis_pairs = [_axis_pair(pair) for pair in pairs] if pairs is not None else [(0, 1), (0, 2), (1, 2)]
    grad_integral = _node_gradient_integral(mesh)  # [n_nodes, 3]: ∫ dN_node/dx_axis dOmega
    node = np.arange(mesh.n_nodes)
    rows: list[np.ndarray] = []
    for axis_i, axis_j in axis_pairs:
        row = np.zeros(mesh.n_dof, dtype=float)
        row[3 * node + axis_i] += grad_integral[:, axis_j]
        row[3 * node + axis_j] -= grad_integral[:, axis_i]
        if np.linalg.norm(row) > tol:
            rows.append(row)
    if not rows:
        return np.zeros((0, mesh.n_dof), dtype=float)
    return np.vstack(rows)


def _node_gradient_integral(mesh: HexMesh) -> np.ndarray:
    """Return ``∫ dN_node/dx_axis dOmega`` per global node via TensorMesh quadrature."""

    import torch

    from openmsg.assembly import tensormesh_quadrature

    _, _, blocks = tensormesh_quadrature(mesh, dtype=torch.float64)
    integral = torch.zeros((mesh.n_nodes, 3), dtype=torch.float64)
    for block in blocks:
        # contrib[e, node, axis] = sum_q dN_dx[e, q, node, axis] * jxw[e, q]
        contrib = torch.einsum("eqbd,eq->ebd", block.dN_dx, block.jxw)
        integral = integral.index_add(0, block.conn.reshape(-1), contrib.reshape(-1, 3))
    return integral.detach().cpu().numpy()


def build_constraint_matrix(
    mesh: HexMesh,
    specs: Iterable[str | dict[str, object]] | None,
    *,
    node_weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build a combined constraint matrix from config-like specs."""

    if specs is None:
        return None

    blocks: list[np.ndarray] = []
    for spec in specs:
        if isinstance(spec, str):
            kind = spec
            data: dict[str, object] = {}
        else:
            kind = str(spec.get("type", ""))
            data = spec
        kind = kind.lower()
        if kind == "mean_zero":
            blocks.append(mean_zero_constraints(mesh, node_weights=node_weights))
        elif kind == "periodic":
            axes = data.get("axes", ("x", "y", "z"))
            blocks.append(periodic_constraints(mesh, axes=axes))  # type: ignore[arg-type]
        elif kind in {"rotation_zero", "beam_twist"}:
            pairs = data.get("pairs")
            if kind == "beam_twist" and pairs is None:
                pairs = [["y", "z"]]
            blocks.append(rotation_zero_constraints(mesh, pairs=pairs))  # type: ignore[arg-type]
        else:
            raise ValueError(f"unsupported constraint type {kind!r}")

    if not blocks:
        return None
    nonempty_blocks = [block for block in blocks if block.size > 0]
    if not nonempty_blocks:
        return None
    G = np.vstack(nonempty_blocks)
    return G if G.size else None


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


def _coord_key(coord: np.ndarray, axes: tuple[int, ...], tol: float) -> tuple[int, ...]:
    scale = max(tol, 1e-15)
    return tuple(int(round(float(coord[axis]) / scale)) for axis in axes)
