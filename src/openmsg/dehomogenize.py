"""Local field recovery for MSG results."""

from __future__ import annotations

import numpy as np

from openmsg.assembly import _element_stiffness_stack, _voigt_b_from_grad3d, tensormesh_quadrature
from openmsg.macro import MacroModel, macro_model_from_kind
from openmsg.mesh import HexMesh


def recover_gauss_fields(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    V0: np.ndarray,
    macro_strain: np.ndarray,
    macro_model: str | MacroModel = "cauchy_3d",
) -> dict[str, np.ndarray]:
    """Recover local strain and stress at all element quadrature points.

    Uses the shared TensorMesh quadrature primitive
    (:func:`openmsg.assembly.tensormesh_quadrature`); the returned NumPy arrays
    are ordered element-major (all quadrature points of element 0, then 1, ...).
    """

    import torch

    model = macro_model_from_kind(macro_model, mesh=mesh)
    eps_bar = np.asarray(macro_strain, dtype=float)
    if eps_bar.shape != (model.n_macro,):
        raise ValueError(f"macro_strain must have shape ({model.n_macro},)")
    V0 = np.asarray(V0, dtype=float)
    if V0.shape != (mesh.n_dof, model.n_macro):
        raise ValueError(f"V0 must have shape (mesh.n_dof, {model.n_macro})")

    dtype = torch.float64
    nodes_t, tm_data, blocks = tensormesh_quadrature(mesh, dtype=dtype)
    device = nodes_t.device
    C_element = _element_stiffness_stack(
        mesh, material_stiffness, tm_data, torch, dtype=dtype, device=device
    )
    eps_bar_t = torch.as_tensor(eps_bar, dtype=dtype, device=device)
    fluctuation = torch.as_tensor(V0 @ eps_bar, dtype=dtype, device=device)  # [n_dof]

    points: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    element_ids: list[np.ndarray] = []
    element_offset = 0

    for block in blocks:
        conn = block.conn  # [n_e, n_basis]
        n_e, n_quad, _, _ = block.dN_dx.shape

        B_node = _voigt_b_from_grad3d(block.dN_dx, torch)  # [n_e, n_q, n_basis, 6, 3]
        comp = torch.arange(3, device=device)
        elem_dofs = (3 * conn.unsqueeze(-1) + comp)  # [n_e, n_basis, 3]
        w_elem = fluctuation[elem_dofs]  # [n_e, n_basis, 3]

        B_macro = model.strain_modes_batch(block.points, torch)  # [n_e, n_q, 6, n_macro]
        strain = torch.einsum("eqbic,ebc->eqi", B_node, w_elem) + torch.einsum(
            "eqik,k->eqi", B_macro, eps_bar_t
        )  # [n_e, n_q, 6]
        stress = torch.einsum("eij,eqj->eqi", C_element, strain)  # [n_e, n_q, 6]

        points.append(block.points.reshape(-1, 3).detach().cpu().numpy())
        strains.append(strain.reshape(-1, 6).detach().cpu().numpy())
        stresses.append(stress.reshape(-1, 6).detach().cpu().numpy())
        element_ids.append(
            np.repeat(np.arange(element_offset, element_offset + n_e), n_quad)
        )
        element_offset += n_e

    return {
        "element_id": np.concatenate(element_ids).astype(int),
        "point": np.concatenate(points).astype(float),
        "strain": np.concatenate(strains).astype(float),
        "stress": np.concatenate(stresses).astype(float),
    }
