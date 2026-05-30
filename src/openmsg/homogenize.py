"""High-level MSG homogenization API."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openmsg.assembly import assemble_msg
from openmsg.constraints import build_constraint_matrix
from openmsg.macro import MacroModel, default_constraints_for_macro_model, macro_model_from_kind
from openmsg.mesh import HexMesh
from openmsg.solver import compute_effective_stiffness, solve_constrained, solve_constrained_sparse


@dataclass(frozen=True)
class MSGResult:
    """Result of a linear elastic MSG homogenization solve."""

    Dbar: np.ndarray
    V0: np.ndarray
    E: np.ndarray
    H: np.ndarray
    D0: np.ndarray
    G: np.ndarray | None
    volume: float
    lagrange: np.ndarray = field(repr=False)
    metadata: dict[str, object] = field(default_factory=dict)
    local_fields: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    def to_dict(self, *, include_internal: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "Dbar": self.Dbar.tolist(),
            "volume": self.volume,
            "metadata": self.metadata,
            "diagnostics": {
                "Dbar_symmetric": bool(np.allclose(self.Dbar, self.Dbar.T, rtol=1e-10, atol=1e-10)),
                "Dbar_eigenvalues": np.linalg.eigvalsh(self.Dbar).tolist(),
                "n_constraints": 0 if self.G is None else int(self.G.shape[0]),
            },
        }
        macro_model = self.metadata.get("macro_model")
        if macro_model == "kirchhoff_love_plate" and self.Dbar.shape == (6, 6):
            data["ABD"] = self.Dbar.tolist()
            data["A"] = self.Dbar[:3, :3].tolist()
            data["B"] = self.Dbar[:3, 3:].tolist()
            data["D"] = self.Dbar[3:, 3:].tolist()
        elif macro_model == "euler_bernoulli_beam":
            data["K"] = self.Dbar.tolist()
        if include_internal:
            data["V0"] = self.V0.tolist()
            data["E"] = self.E.tolist()
            data["H"] = self.H.tolist()
            data["D0"] = self.D0.tolist()
            data["G"] = None if self.G is None else self.G.tolist()
        if self.local_fields:
            data["local_fields"] = {
                name: {key: value.tolist() for key, value in fields.items()}
                for name, fields in self.local_fields.items()
            }
        return data


def homogenize_3d_cauchy(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | None = (
        {"type": "periodic", "axes": ["x", "y", "z"]},
        {"type": "mean_zero"},
    ),
    linear_solver: str = "auto",
) -> MSGResult:
    """Run MSG homogenization for a 3D Cauchy macroscopic model."""

    return homogenize_msg(
        mesh=mesh,
        material_stiffness=material_stiffness,
        macro_model="cauchy_3d",
        constraints=constraints,
        linear_solver=linear_solver,
    )


def homogenize_msg(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    macro_model: str | MacroModel = "cauchy_3d",
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | str | None = "auto",
    linear_solver: str = "auto",
) -> MSGResult:
    """Run MSG homogenization for a supported macroscopic model."""

    model = macro_model_from_kind(macro_model, mesh=mesh)
    if constraints == "auto":
        constraint_specs = default_constraints_for_macro_model(model, mesh)
    else:
        constraint_specs = constraints
    assembly, assembly_metadata = assemble_msg(mesh, material_stiffness, macro_model=model)

    omega = model.normalization(mesh, assembly.volume)
    G = build_constraint_matrix(mesh, constraint_specs, node_weights=assembly.node_weights)
    selected_solver = _select_linear_solver(linear_solver, assembly.E_sparse)
    if selected_solver == "sparse":
        sparse_E = assembly.E_sparse if assembly.E_sparse is not None else assembly.E
        V0, lagrange = solve_constrained_sparse(sparse_E, assembly.H, G, return_lagrange=True)
    else:
        V0, lagrange = solve_constrained(assembly.E, assembly.H, G, return_lagrange=True)
    Dbar = compute_effective_stiffness(assembly.D0, assembly.H, V0, omega=omega)
    return MSGResult(
        Dbar=Dbar,
        V0=V0,
        E=assembly.E,
        H=assembly.H,
        D0=assembly.D0,
        G=G,
        volume=assembly.volume,
        lagrange=lagrange,
        metadata={
            "model": model.model_name,
            "n_nodes": mesh.n_nodes,
            "n_elements": mesh.n_elements,
            "element_type": mesh.element_type,
            "sg_dimension": mesh.sg_dimension,
            "active_axes": list(mesh.active_axes),
            "omega": omega,
            "materials": sorted(set(mesh.material_ids)),
            "linear_solver": selected_solver,
            **model.metadata(),
            **assembly_metadata,
        },
    )


def homogenize_kirchhoff_love_plate(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | str | None = "auto",
    linear_solver: str = "auto",
) -> MSGResult:
    """Run MSG homogenization for a Kirchhoff-Love plate macroscopic model."""

    return homogenize_msg(
        mesh=mesh,
        material_stiffness=material_stiffness,
        macro_model="kirchhoff_love_plate",
        constraints=constraints,
        linear_solver=linear_solver,
    )


def homogenize_euler_bernoulli_beam(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, np.ndarray],
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | str | None = "auto",
    linear_solver: str = "auto",
) -> MSGResult:
    """Run MSG homogenization for an Euler-Bernoulli beam macroscopic model."""

    return homogenize_msg(
        mesh=mesh,
        material_stiffness=material_stiffness,
        macro_model="euler_bernoulli_beam",
        constraints=constraints,
        linear_solver=linear_solver,
    )


def _select_linear_solver(requested: str, sparse_matrix: object | None) -> str:
    key = requested.lower()
    if key == "auto":
        return "sparse" if sparse_matrix is not None else "dense"
    if key in {"dense", "sparse"}:
        return key
    raise ValueError(f"unsupported linear_solver {requested!r}")
