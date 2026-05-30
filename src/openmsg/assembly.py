"""TensorMesh-backed, autograd-friendly MSG finite element assembly.

The assembly is fully tensorized in PyTorch and differentiable with respect to
both the material stiffness tensors and the node coordinates (geometry). It uses
TensorMesh element shape functions / quadrature for the reference machinery and
returns a ``torch_sla``-backed sparse stiffness matrix that plugs directly into
the differentiable TensorMesh linear solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from openmsg.macro import MacroModel, macro_model_from_kind
from openmsg.mesh import HexMesh, OPENMSG_TO_MESHIO_ELEMENT

ASSEMBLY_KERNEL = "tensormesh_autograd"


@dataclass(frozen=True)
class MSGSystem:
    """Differentiable assembled MSG system (PyTorch tensors).

    ``E`` is a ``tensormesh.sparse.SparseMatrix`` (a ``torch_sla.SparseTensor``
    subclass); every field carries autograd history back to the material
    stiffness tensors and the node-coordinate tensor used during assembly.
    """

    E: object
    H: object
    D0: object
    volume: object
    node_weights: object
    nodes: object
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AssemblyResult:
    """NumPy view of an assembled MSG system (for reporting / inspection)."""

    E: np.ndarray
    H: np.ndarray
    D0: np.ndarray
    volume: float
    node_weights: np.ndarray
    E_sparse: object


@dataclass(frozen=True)
class TensorMeshData:
    """Converted TensorMesh mesh plus material-index metadata."""

    mesh: object
    material_index: np.ndarray
    material_names: tuple[str, ...]


@dataclass(frozen=True)
class ElementQuadrature:
    """Differentiable TensorMesh quadrature for one element block, embedded in 3D.

    This is the single TensorMesh-backed quadrature primitive shared by the
    assembly, local-field recovery, and rotation-constraint construction; it
    replaces the retired hand-written NumPy quadrature. Every tensor carries
    autograd history back to the node coordinates when those require grad.

    ``conn`` stores global node indices (TensorMesh permutes only the local
    node order within an element, so the global DOF numbering ``3 * node + comp``
    matches the original mesh). ``dN_dx`` already embeds the active-axis physical
    gradients into 3D, with zeros along inactive SG axes.
    """

    cell_type: str
    conn: object  # LongTensor [n_e, n_basis]
    shape_val: object  # [n_quad, n_basis]
    dN_dx: object  # [n_e, n_quad, n_basis, 3]
    jxw: object  # [n_e, n_quad]
    points: object  # [n_e, n_quad, 3]


def assemble_msg_system(
    mesh: HexMesh,
    material_stiffness: Mapping[str, object],
    *,
    macro_model: MacroModel,
    nodes: object | None = None,
    dtype: object | None = None,
    device: object | None = None,
) -> MSGSystem:
    """Assemble the differentiable MSG system with TensorMesh machinery.

    Parameters
    ----------
    mesh:
        OpenMSG structure-genome mesh (topology and reference geometry).
    material_stiffness:
        Mapping from material id to a 6x6 stiffness; values may be NumPy arrays
        or ``torch.Tensor`` objects. Pass tensors with ``requires_grad=True`` to
        differentiate the homogenized response with respect to the material.
    macro_model:
        Macroscopic model providing the local Voigt strain modes.
    nodes:
        Optional ``[n_nodes, 3]`` coordinate tensor. Pass a tensor with
        ``requires_grad=True`` to differentiate with respect to geometry. When
        ``None``, the mesh's reference coordinates are used (no grad).
    """

    import torch
    from tensormesh.sparse import SparseMatrix

    nodes_t, tm_data, blocks = tensormesh_quadrature(
        mesh, nodes=nodes, dtype=dtype, device=device
    )
    device = nodes_t.device
    dtype = nodes_t.dtype

    C_element = _element_stiffness_stack(
        mesh, material_stiffness, tm_data, torch, dtype=dtype, device=device
    )

    n_dof = mesh.n_dof
    n_macro = macro_model.n_macro

    rows: list[object] = []
    cols: list[object] = []
    vals: list[object] = []
    H = torch.zeros((n_dof, n_macro), dtype=dtype, device=device)
    D0 = torch.zeros((n_macro, n_macro), dtype=dtype, device=device)
    node_weights = torch.zeros(mesh.n_nodes, dtype=dtype, device=device)
    volume = torch.zeros((), dtype=dtype, device=device)

    for block in blocks:
        conn = block.conn
        jxw = block.jxw
        shape_val = block.shape_val
        n_e, n_quad, n_basis, _ = block.dN_dx.shape
        n_local = 3 * n_basis

        B_node = _voigt_b_from_grad3d(block.dN_dx, torch)  # [n_e, n_q, n_basis, 6, 3]
        B = B_node.permute(0, 1, 3, 2, 4).reshape(n_e, n_quad, 6, n_local)  # column = node*3 + comp

        C_e = C_element  # [n_e, 6, 6] (single element type per mesh)
        CB = torch.einsum("eij,eqjk->eqik", C_e, B)  # C @ B
        E_local = torch.einsum("eqmi,eqmj,eq->eij", B, CB, jxw)  # [n_e, n_local, n_local]

        B_macro = macro_model.strain_modes_batch(block.points, torch)  # [n_e, n_q, 6, n_macro]
        CBm = torch.einsum("eij,eqjk->eqik", C_e, B_macro)
        H_local = torch.einsum("eqmi,eqmk,eq->eik", B, CBm, jxw)  # [n_e, n_local, n_macro]
        D0 = D0 + torch.einsum("eqmi,eqmk,eq->ik", B_macro, CBm, jxw)
        volume = volume + jxw.sum()

        comp = torch.arange(3, device=device)
        dof = (3 * conn.unsqueeze(-1) + comp).reshape(n_e, n_local)  # pos node*3+comp

        rows.append(dof.unsqueeze(-1).expand(n_e, n_local, n_local).reshape(-1))
        cols.append(dof.unsqueeze(1).expand(n_e, n_local, n_local).reshape(-1))
        vals.append(E_local.reshape(-1))

        H = H.index_add(0, dof.reshape(-1), H_local.reshape(-1, n_macro))
        nw_local = torch.einsum("qb,eq->eb", shape_val, jxw)  # [n_e, n_basis]
        node_weights = node_weights.index_add(0, conn.reshape(-1), nw_local.reshape(-1))

    E_sparse = SparseMatrix(
        torch.cat(vals), torch.cat(rows), torch.cat(cols), (n_dof, n_dof)
    )

    metadata = {
        "assembly_kernel": ASSEMBLY_KERNEL,
        "assembly_material_names": list(tm_data.material_names),
        "assembly_cell_type": OPENMSG_TO_MESHIO_ELEMENT[mesh.element_type],
    }
    return MSGSystem(
        E=E_sparse,
        H=H,
        D0=D0,
        volume=volume,
        node_weights=node_weights,
        nodes=nodes_t,
        metadata=metadata,
    )


def assemble_msg(
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    *,
    macro_model: MacroModel,
) -> tuple[AssemblyResult, dict[str, object]]:
    """Assemble MSG matrices and return a NumPy view (detached from autograd)."""

    system = assemble_msg_system(mesh, material_stiffness, macro_model=macro_model)
    result = AssemblyResult(
        E=system.E.to_dense().detach().cpu().numpy(),
        H=system.H.detach().cpu().numpy(),
        D0=system.D0.detach().cpu().numpy(),
        volume=float(system.volume.detach().cpu()),
        node_weights=system.node_weights.detach().cpu().numpy(),
        E_sparse=system.E,
    )
    return result, system.metadata


def assemble_3d_cauchy(
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
) -> tuple[AssemblyResult, dict[str, object]]:
    """Assemble 3D Cauchy MSG matrices (NumPy view)."""

    return assemble_msg(
        mesh,
        material_stiffness,
        macro_model=macro_model_from_kind("cauchy_3d", mesh=mesh),
    )


def to_tensormesh_mesh(mesh: HexMesh) -> TensorMeshData:
    """Convert an OpenMSG SG mesh to a TensorMesh mesh."""

    import meshio
    import tensormesh

    material_names = tuple(sorted(set(mesh.material_ids)))
    material_lookup = {name: idx for idx, name in enumerate(material_names)}
    material_index = np.asarray([material_lookup[name] for name in mesh.material_ids], dtype=np.int64)
    points = mesh.nodes if mesh.sg_dimension == 3 else mesh.nodes[:, mesh.active_axes]
    meshio_mesh = meshio.Mesh(
        points=points,
        cells=[(OPENMSG_TO_MESHIO_ELEMENT[mesh.element_type], mesh.elements)],
        cell_data={"material_index": [material_index]},
    )
    return TensorMeshData(
        mesh=tensormesh.Mesh(meshio_mesh, reorder=True),
        material_index=material_index,
        material_names=material_names,
    )


def tensormesh_quadrature(
    mesh: HexMesh,
    *,
    nodes: object | None = None,
    dtype: object | None = None,
    device: object | None = None,
) -> tuple[object, TensorMeshData, list[ElementQuadrature]]:
    """Differentiable TensorMesh element quadrature embedded in 3D space.

    This is the shared TensorMesh primitive used by assembly, local-field
    recovery, and rotation-constraint construction. It returns
    ``(nodes_tensor, tm_data, blocks)`` where ``blocks`` holds one
    :class:`ElementQuadrature` per element type with shape values, 3D-embedded
    physical shape-function gradients, integration weights, and physical
    quadrature-point coordinates as ``torch`` tensors. Pass ``nodes`` with
    ``requires_grad=True`` to differentiate the quadrature data with respect to
    geometry.
    """

    import torch

    if dtype is None:
        dtype = torch.float64

    nodes_t = _as_node_tensor(mesh, nodes, torch, dtype=dtype, device=device)
    device = nodes_t.device
    dtype = nodes_t.dtype

    active_axes = tuple(mesh.active_axes)
    points = nodes_t if mesh.sg_dimension == 3 else nodes_t[:, active_axes]

    tm_data = to_tensormesh_mesh(mesh)
    provider = _transformation_provider(tm_data.mesh)

    blocks: list[ElementQuadrature] = []
    for cell_type in provider.element_types:
        trans = provider.transformation[cell_type]
        trans.update_points(points)
        conn = provider.elements[cell_type].to(device)
        n_quad = trans.n_quadrature

        shape_val = trans.batch_shape_val(0, n_quad)  # [n_quad, n_basis]
        shape_grad, jxw = trans.batch_shape_grad_jxw(
            quadrature_start=0, quadrature_batch=n_quad
        )  # [n_e, n_quad, n_basis, dim], [n_e, n_quad]
        dN_dx = _embed_grad_3d(shape_grad, active_axes, torch)  # [n_e, n_quad, n_basis, 3]
        x_q = torch.einsum("qb,ebd->eqd", shape_val, nodes_t[conn])  # [n_e, n_quad, 3]
        blocks.append(
            ElementQuadrature(
                cell_type=cell_type,
                conn=conn,
                shape_val=shape_val,
                dN_dx=dN_dx,
                jxw=jxw,
                points=x_q,
            )
        )
    return nodes_t, tm_data, blocks


def _as_node_tensor(mesh, nodes, torch, *, dtype, device):
    if nodes is None:
        return torch.as_tensor(np.asarray(mesh.nodes, dtype=float), dtype=dtype, device=device)
    if isinstance(nodes, torch.Tensor):
        return nodes.to(dtype=dtype, device=device) if device is not None else nodes.to(dtype=dtype)
    return torch.as_tensor(np.asarray(nodes, dtype=float), dtype=dtype, device=device)


def _element_stiffness_stack(mesh, material_stiffness, tm_data, torch, *, dtype, device):
    """Stack per-element 6x6 stiffness, preserving autograd to material tensors."""

    columns = []
    for name in tm_data.material_names:
        value = material_stiffness[name]
        if isinstance(value, torch.Tensor):
            columns.append(value.to(dtype=dtype, device=device))
        else:
            columns.append(torch.as_tensor(np.asarray(value, dtype=float), dtype=dtype, device=device))
    C_stack = torch.stack(columns, dim=0)  # [n_materials, 6, 6]
    index = torch.as_tensor(tm_data.material_index, dtype=torch.long, device=device)
    return C_stack[index]  # [n_elements, 6, 6]


def _transformation_provider(tm_mesh):
    """Build a TensorMesh assembler purely to obtain transformations / connectivity."""

    from tensormesh.assemble import ElementAssembler

    class _Provider(ElementAssembler):  # type: ignore[misc]
        def forward(self, gradu, gradv):  # pragma: no cover - never invoked
            return (gradu * gradv).sum()

    return _Provider.from_mesh(tm_mesh, quadrature_order=2)


def _embed_grad_3d(shape_grad, active_axes, torch):
    """Embed active-axis shape-function gradients into 3D physical space.

    ``shape_grad`` has shape ``[n_e, n_q, n_basis, dim]`` in active-axis order.
    Returns ``[n_e, n_q, n_basis, 3]`` with zeros along inactive SG axes, so the
    standard 3D Voigt operator can be reused for reduced-dimensional SGs.
    """

    zero = torch.zeros_like(shape_grad[..., 0])
    grads = [zero, zero, zero]
    for local_axis, axis in enumerate(active_axes):
        grads[axis] = shape_grad[..., local_axis]
    return torch.stack(grads, dim=-1)


def _voigt_b_from_grad3d(dN_dx, torch):
    """Batched 3D Cauchy Voigt strain-displacement blocks from 3D gradients.

    ``dN_dx`` has shape ``[..., n_basis, 3]``. Returns ``[..., n_basis, 6, 3]``
    using Voigt order ``[e11, e22, e33, 2e23, 2e13, 2e12]`` with engineering
    shear strains.
    """

    gx, gy, gz = dN_dx[..., 0], dN_dx[..., 1], dN_dx[..., 2]
    zero = torch.zeros_like(gx)
    rows = [
        [gx, zero, zero],
        [zero, gy, zero],
        [zero, zero, gz],
        [zero, gz, gy],
        [gz, zero, gx],
        [gy, gx, zero],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
