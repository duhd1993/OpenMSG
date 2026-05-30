"""Local field recovery for MSG results."""

from __future__ import annotations

import numpy as np

from openmsg.elements import iter_element_quadrature
from openmsg.macro import MacroModel, macro_model_from_kind
from openmsg.mesh import HexMesh
from openmsg.voigt import element_dof_indices, strain_displacement_matrix


def recover_gauss_fields(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    V0: np.ndarray,
    macro_strain: np.ndarray,
    macro_model: str | MacroModel = "cauchy_3d",
) -> dict[str, np.ndarray]:
    """Recover local strain and stress at all element quadrature points."""

    model = macro_model_from_kind(macro_model, mesh=mesh)
    eps_bar = np.asarray(macro_strain, dtype=float)
    if eps_bar.shape != (model.n_macro,):
        raise ValueError(f"macro_strain must have shape ({model.n_macro},)")
    V0 = np.asarray(V0, dtype=float)
    if V0.shape != (mesh.n_dof, model.n_macro):
        raise ValueError(f"V0 must have shape (mesh.n_dof, {model.n_macro})")

    fluctuation_dofs = V0 @ eps_bar
    points: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    element_ids: list[int] = []

    for element_id, (conn, material_id) in enumerate(zip(mesh.elements, mesh.material_ids)):
        C = np.asarray(material_stiffness[material_id], dtype=float)
        coords = mesh.nodes[conn]
        edofs = element_dof_indices(conn)
        elem_w = fluctuation_dofs[edofs]

        for qp in iter_element_quadrature(mesh.element_type, coords, active_axes=mesh.active_axes):
            B = strain_displacement_matrix(qp.dN_dx)
            B_macro = model.strain_modes(qp.shape @ coords)
            strain = B @ elem_w + B_macro @ eps_bar
            stress = C @ strain
            points.append(qp.shape @ coords)
            strains.append(strain)
            stresses.append(stress)
            element_ids.append(element_id)

    return {
        "element_id": np.asarray(element_ids, dtype=int),
        "point": np.asarray(points, dtype=float),
        "strain": np.asarray(strains, dtype=float),
        "stress": np.asarray(stresses, dtype=float),
    }
