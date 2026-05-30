"""High-level MSG homogenization API.

``effective_stiffness`` returns a PyTorch-backed :class:`MSGResult`. ``Dbar``
carries autograd history back to the material stiffness tensors, so gradients
of scalar objectives can be obtained with ``torch.autograd``. JSON-friendly
conversion happens only in :meth:`MSGResult.to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openmsg.assembly import assemble_msg_system
from openmsg.constraints import build_constraint_tensor
from openmsg.macro import MacroModel, default_constraints_for_macro_model, macro_model_from_kind
from openmsg.mesh import SolidMesh
from openmsg.solver import compute_effective_stiffness, solve_constrained, solve_constrained_sparse


@dataclass(frozen=True)
class MSGResult:
    """PyTorch-backed result of a linear elastic MSG homogenization solve.

    ``Dbar`` carries autograd history; call ``Dbar.backward()`` or
    ``torch.autograd.grad`` to obtain gradients with respect to any
    ``requires_grad`` material stiffness input. ``E`` is the sparse TensorMesh /
    ``torch-sla`` stiffness matrix; use ``E.to_dense()`` only for diagnostics or
    serialization.
    """

    Dbar: object
    V0: object
    E: object
    H: object
    D0: object
    G: object | None
    volume: object
    omega: object
    lagrange: object = field(repr=False)
    node_weights: object = field(repr=False)
    metadata: dict[str, object] = field(default_factory=dict)
    local_fields: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)

    def to_dict(self, *, include_internal: bool = False) -> dict[str, object]:
        import torch

        Dbar = _as_detached_cpu(self.Dbar)
        data: dict[str, object] = {
            "Dbar": _to_list(Dbar),
            "volume": _to_float(self.volume),
            "omega": _to_float(self.omega),
            "metadata": self.metadata,
            "diagnostics": {
                "Dbar_symmetric": bool(torch.allclose(Dbar, Dbar.transpose(0, 1), rtol=1e-10, atol=1e-10)),
                "Dbar_eigenvalues": _to_list(torch.linalg.eigvalsh(Dbar)),
                "n_constraints": 0 if self.G is None else int(self.G.shape[0]),
            },
        }
        macro_model = self.metadata.get("macro_model")
        if macro_model == "kirchhoff_love_plate" and tuple(Dbar.shape) == (6, 6):
            data["ABD"] = _to_list(Dbar)
            data["A"] = _to_list(Dbar[:3, :3])
            data["B"] = _to_list(Dbar[:3, 3:])
            data["D"] = _to_list(Dbar[3:, 3:])
        elif macro_model == "euler_bernoulli_beam":
            data["K"] = _to_list(Dbar)
        if include_internal:
            data["V0"] = _to_list(self.V0)
            data["E"] = _to_list(self.E.to_dense())
            data["H"] = _to_list(self.H)
            data["D0"] = _to_list(self.D0)
            data["G"] = None if self.G is None else _to_list(self.G)
        if self.local_fields:
            data["local_fields"] = {
                name: {key: _to_list(value) for key, value in fields.items()}
                for name, fields in self.local_fields.items()
            }
        return data


def effective_stiffness(
    *,
    mesh: SolidMesh,
    material_stiffness: dict[str, object],
    macro_model: str | MacroModel = "cauchy_3d",
    constraints: list[str | dict[str, object]] | tuple[str | dict[str, object], ...] | str | None = "auto",
    linear_solver: str = "auto",
    dtype: object | None = None,
    device: object | None = None,
) -> MSGResult:
    """Differentiable MSG homogenization.

    ``material_stiffness`` values must be 6x6 ``torch.Tensor`` objects. Pass
    tensors or material-builder parameters with ``requires_grad=True`` to obtain
    gradients of ``Dbar`` with respect to material stiffness. Mesh node
    coordinates are treated as fixed geometry, then converted to torch tensors
    for quadrature, constraints, and normalization.
    """

    import torch

    model = macro_model_from_kind(macro_model, mesh=mesh)
    if constraints == "auto":
        constraint_specs = default_constraints_for_macro_model(model, mesh)
    else:
        constraint_specs = constraints

    system = assemble_msg_system(
        mesh, material_stiffness, macro_model=model, dtype=dtype, device=device
    )
    H = system.H
    omega = _omega_torch(model, system.volume, mesh, torch)

    G_t = build_constraint_tensor(
        mesh,
        constraint_specs,
        node_weights=system.node_weights,
        dtype=H.dtype,
        device=H.device,
    )

    solver_key = linear_solver.lower()
    if solver_key == "auto":
        selected_solver = "sparse"
    elif solver_key in {"dense", "sparse"}:
        selected_solver = solver_key
    else:
        raise ValueError(f"unsupported linear_solver {linear_solver!r}")

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
    return MSGResult(
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
        metadata=metadata,
    )


def _omega_torch(model: MacroModel, volume: object, mesh: SolidMesh, torch: object) -> object:
    """Return the MSG normalization factor as a tensor on ``volume``'s device."""

    if model.kind == "cauchy_3d":
        return volume

    if model.kind not in {"kirchhoff_love_plate", "euler_bernoulli_beam"}:
        raise ValueError(f"unsupported macro model {model.kind!r}")

    axes = model.inplane_axes if model.kind == "kirchhoff_love_plate" else (model.axial_axis,)
    nodes = torch.as_tensor(mesh.nodes, dtype=volume.dtype, device=volume.device)  # type: ignore[attr-defined]
    active_axes = set(mesh.active_axes)
    omega = torch.ones((), dtype=volume.dtype, device=volume.device)  # type: ignore[attr-defined]
    for axis in axes:
        if axis not in active_axes:
            continue
        extent = nodes[:, axis].max() - nodes[:, axis].min()
        if float(extent.detach().cpu()) > 0.0:
            omega = omega * extent
    return omega


def _as_detached_cpu(value: object):
    if hasattr(value, "detach"):
        return value.detach().cpu()
    raise TypeError("MSGResult fields must be torch tensors")


def _to_float(value: object) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)  # local field scalar fallback


def _to_list(value: object) -> object:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
