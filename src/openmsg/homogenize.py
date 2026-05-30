"""High-level MSG homogenization API.

``effective_stiffness`` is the differentiable entry point: it returns the
homogenized stiffness as a ``torch.Tensor`` with autograd history back to the
material stiffness tensors and the node-coordinate tensor, so gradients of any
scalar objective can be obtained with ``torch.autograd``. ``homogenize_msg``
wraps it and returns a detached NumPy :class:`MSGResult` for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openmsg.assembly import assemble_msg_system
from openmsg.constraints import build_constraint_matrix
from openmsg.macro import MacroModel, default_constraints_for_macro_model, macro_model_from_kind
from openmsg.mesh import HexMesh
from openmsg.solver import compute_effective_stiffness, solve_constrained, solve_constrained_sparse


@dataclass(frozen=True)
class MSGResult:
    """Result of a linear elastic MSG homogenization solve (NumPy view)."""

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


@dataclass(frozen=True)
class MSGTorchResult:
    """Differentiable MSG homogenization result (PyTorch tensors).

    ``Dbar`` carries autograd history; call ``Dbar.backward()`` or
    ``torch.autograd.grad`` to obtain gradients with respect to any
    ``requires_grad`` material stiffness or node-coordinate input.
    """

    Dbar: object
    V0: object
    E: object
    H: object
    D0: object
    G: object | None
    volume: object
    omega: object
    lagrange: object
    node_weights: object
    nodes: object
    metadata: dict[str, object] = field(default_factory=dict)


def effective_stiffness(
    *,
    mesh: HexMesh,
    material_stiffness: dict[str, object],
    nodes: object | None = None,
    macro_model: str | MacroModel = "cauchy_3d",
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | str | None = "auto",
    linear_solver: str = "auto",
    dtype: object | None = None,
    device: object | None = None,
) -> MSGTorchResult:
    """Differentiable MSG homogenization.

    Pass material stiffness values and/or a ``nodes`` tensor with
    ``requires_grad=True`` to obtain gradients of ``Dbar`` with respect to the
    material and the geometry, respectively.
    """

    import torch

    model = macro_model_from_kind(macro_model, mesh=mesh)
    if constraints == "auto":
        constraint_specs = default_constraints_for_macro_model(model, mesh)
    else:
        constraint_specs = constraints

    system = assemble_msg_system(
        mesh, material_stiffness, macro_model=model, nodes=nodes, dtype=dtype, device=device
    )
    H = system.H
    omega = _omega_torch(model, system.nodes, system.volume, mesh, torch)

    G_np = build_constraint_matrix(
        mesh, constraint_specs, node_weights=system.node_weights.detach().cpu().numpy()
    )
    G_t = None
    if G_np is not None and G_np.size:
        G_t = torch.as_tensor(G_np, dtype=H.dtype, device=H.device)

    selected_solver = _select_linear_solver(linear_solver, system.E)
    if selected_solver == "sparse":
        V0, lagrange = solve_constrained_sparse(system.E, H, G_t, return_lagrange=True)
    else:
        V0, lagrange = solve_constrained(system.E, H, G_t, return_lagrange=True)
    Dbar = compute_effective_stiffness(system.D0, H, V0, omega=omega)

    metadata = {
        "model": model.model_name,
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "element_type": mesh.element_type,
        "sg_dimension": mesh.sg_dimension,
        "active_axes": list(mesh.active_axes),
        "omega": float(omega.detach().cpu()) if hasattr(omega, "detach") else float(omega),
        "materials": sorted(set(mesh.material_ids)),
        "linear_solver": selected_solver,
        **model.metadata(),
        **system.metadata,
    }
    return MSGTorchResult(
        Dbar=Dbar,
        V0=V0,
        E=system.E,
        H=H,
        D0=system.D0,
        G=G_t,
        volume=system.volume,
        omega=omega,
        lagrange=lagrange,
        node_weights=system.node_weights,
        nodes=system.nodes,
        metadata=metadata,
    )


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
    """Run MSG homogenization and return a detached NumPy :class:`MSGResult`."""

    result = effective_stiffness(
        mesh=mesh,
        material_stiffness=material_stiffness,
        macro_model=macro_model,
        constraints=constraints,
        linear_solver=linear_solver,
    )
    G = None if result.G is None else result.G.detach().cpu().numpy()
    return MSGResult(
        Dbar=result.Dbar.detach().cpu().numpy(),
        V0=result.V0.detach().cpu().numpy(),
        E=result.E.to_dense().detach().cpu().numpy(),
        H=result.H.detach().cpu().numpy(),
        D0=result.D0.detach().cpu().numpy(),
        G=G,
        volume=float(result.volume.detach().cpu()) if hasattr(result.volume, "detach") else float(result.volume),
        lagrange=result.lagrange.detach().cpu().numpy(),
        metadata=result.metadata,
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


def _omega_torch(model: MacroModel, nodes_t: object, volume: object, mesh: HexMesh, torch: object) -> object:
    """Differentiable MSG normalization factor (omega)."""

    if model.kind == "cauchy_3d":
        return volume

    active = set(mesh.active_axes)
    factor = torch.ones((), dtype=volume.dtype, device=volume.device)  # type: ignore[attr-defined]

    def _extent(axis: int) -> object:
        col = nodes_t[:, axis]  # type: ignore[index]
        return col.max() - col.min()

    if model.kind == "kirchhoff_love_plate":
        axes = model.inplane_axes
    elif model.kind == "euler_bernoulli_beam":
        axes = (model.axial_axis,)
    else:
        raise ValueError(f"unsupported macro model {model.kind!r}")

    for axis in axes:
        if axis in active:
            extent = _extent(axis)
            if float(extent.detach()) > 0.0:  # type: ignore[attr-defined]
                factor = factor * extent
    return factor


def _select_linear_solver(requested: str, sparse_matrix: object | None) -> str:
    key = requested.lower()
    if key == "auto":
        return "sparse" if sparse_matrix is not None else "dense"
    if key in {"dense", "sparse"}:
        return key
    raise ValueError(f"unsupported linear_solver {requested!r}")
