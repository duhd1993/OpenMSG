"""TensorMesh-backed MSG finite element assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openmsg.macro import MacroModel, macro_model_from_kind
from openmsg.mesh import HexMesh, OPENMSG_TO_MESHIO_ELEMENT


@dataclass(frozen=True)
class AssemblyResult:
    """Matrices and integration data produced by MSG finite element assembly."""

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


def assemble_msg(
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    *,
    macro_model: MacroModel,
) -> tuple[AssemblyResult, dict[str, object]]:
    """Assemble MSG matrices with TensorMesh element machinery."""

    tm_data = to_tensormesh_mesh(mesh)
    import torch
    from tensormesh.assemble import ElementAssembler

    active_axes = tuple(mesh.active_axes)

    class FullStiffnessAssembler(ElementAssembler):  # type: ignore[misc]
        def forward(self, gradu, gradv, C):  # type: ignore[no-untyped-def]
            Ba = _embedded_voigt_shape_grad(gradu, active_axes, torch)
            Bb = _embedded_voigt_shape_grad(gradv, active_axes, torch)
            return Ba.T @ C @ Bb

    tm_mesh = tm_data.mesh
    dtype = tm_mesh.points.dtype
    device = tm_mesh.points.device
    C_element = torch.stack(
        [
            torch.as_tensor(material_stiffness[material_id], dtype=dtype, device=device)
            for material_id in mesh.material_ids
        ],
        dim=0,
    )

    assembler = FullStiffnessAssembler.from_mesh(tm_mesh, quadrature_order=2)
    E_sparse = assembler(tm_mesh.points, element_data={"C": C_element})
    E = E_sparse.to_dense().detach().cpu().numpy()
    H, D0, volume, node_weights = _assemble_coupling_terms(
        assembler=assembler,
        mesh=mesh,
        macro_model=macro_model,
        C_element=C_element,
        torch=torch,
        active_axes=active_axes,
    )
    assembly = AssemblyResult(E=E, H=H, D0=D0, volume=volume, node_weights=node_weights, E_sparse=E_sparse)
    return assembly, {
        "assembly_kernel": "tensormesh_element_assembler",
        "assembly_material_names": list(tm_data.material_names),
        "assembly_cell_type": OPENMSG_TO_MESHIO_ELEMENT[mesh.element_type],
    }


def assemble_3d_cauchy(
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
) -> tuple[AssemblyResult, dict[str, object]]:
    """Assemble 3D Cauchy MSG matrices."""

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


def _assemble_coupling_terms(
    *,
    assembler: object,
    mesh: HexMesh,
    macro_model: MacroModel,
    C_element: object,
    torch: object,
    active_axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n_dof = mesh.n_dof
    n_macro = macro_model.n_macro
    H = torch.zeros((n_dof, n_macro), dtype=C_element.dtype, device=C_element.device)
    D0 = torch.zeros((n_macro, n_macro), dtype=C_element.dtype, device=C_element.device)
    node_weights = torch.zeros(mesh.n_nodes, dtype=C_element.dtype, device=C_element.device)
    volume = torch.zeros((), dtype=C_element.dtype, device=C_element.device)
    full_nodes = torch.as_tensor(mesh.nodes, dtype=C_element.dtype, device=C_element.device)

    for element_type in assembler.element_types:
        elements = assembler.elements[element_type]
        transformation = assembler.transformation[element_type]
        shape_val = transformation.batch_shape_val(0, transformation.n_quadrature)
        shape_grad, jxw = transformation.batch_shape_grad_jxw(
            quadrature_start=0,
            quadrature_batch=transformation.n_quadrature,
        )

        for elem_index in range(elements.shape[0]):
            C = C_element[elem_index]
            conn = elements[elem_index]
            coords = full_nodes[conn]
            for q in range(transformation.n_quadrature):
                dV = jxw[elem_index, q]
                point = shape_val[q] @ coords
                B_macro = macro_model.strain_modes_torch(point, torch)
                D0 = D0 + B_macro.T @ C @ B_macro * dV
                volume = volume + dV
                node_weights[conn] = node_weights[conn] + shape_val[q] * dV
                for local_node in range(conn.shape[0]):
                    Ba = _embedded_voigt_shape_grad(shape_grad[elem_index, q, local_node], active_axes, torch)
                    local_H = Ba.T @ C @ B_macro * dV
                    global_node = int(conn[local_node].item())
                    H[3 * global_node : 3 * global_node + 3, :] = (
                        H[3 * global_node : 3 * global_node + 3, :] + local_H
                    )

    return (
        H.detach().cpu().numpy(),
        D0.detach().cpu().numpy(),
        float(volume.detach().cpu().item()),
        node_weights.detach().cpu().numpy(),
    )


def _embedded_voigt_shape_grad(gradu: object, active_axes: tuple[int, ...], torch: object) -> object:
    """Embed a TensorMesh shape gradient into 3D Cauchy Voigt form."""

    zero = gradu.new_zeros(())  # type: ignore[attr-defined]
    grads = [zero, zero, zero]
    for local_axis, axis in enumerate(active_axes):
        grads[axis] = gradu[local_axis]  # type: ignore[index]
    gx, gy, gz = grads
    return torch.stack(
        [
            torch.stack([gx, zero, zero]),
            torch.stack([zero, gy, zero]),
            torch.stack([zero, zero, gz]),
            torch.stack([zero, gz, gy]),
            torch.stack([gz, zero, gx]),
            torch.stack([gy, gx, zero]),
        ]
    )
