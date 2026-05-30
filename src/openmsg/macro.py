"""Macroscopic model strain modes for MSG homogenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from openmsg.mesh import SolidMesh

AXIS_MAP = {"x": 0, "y": 1, "z": 2, "1": 0, "2": 1, "3": 2, 0: 0, 1: 1, 2: 2}
AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class MacroModel:
    """Macroscopic structural model used by an MSG solve."""

    kind: str
    labels: tuple[str, ...]
    reference_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    thickness_axis: int = 2
    inplane_axes: tuple[int, int] = (0, 1)
    axial_axis: int = 0
    cross_section_axes: tuple[int, int] = (1, 2)

    @property
    def n_macro(self) -> int:
        return len(self.labels)

    @property
    def model_name(self) -> str:
        if self.kind == "cauchy_3d":
            return "msg_3d_cauchy"
        if self.kind == "kirchhoff_love_plate":
            return "msg_kirchhoff_love_plate"
        if self.kind == "euler_bernoulli_beam":
            return "msg_euler_bernoulli_beam"
        return f"msg_{self.kind}"

    def strain_modes(self, point: np.ndarray) -> np.ndarray:
        """Return the local 3D Voigt strain modes at a physical SG point."""

        y = np.asarray(point, dtype=float)
        if y.shape != (3,):
            raise ValueError("point must have shape (3,)")
        if self.kind == "cauchy_3d":
            return np.eye(6, dtype=float)
        if self.kind == "kirchhoff_love_plate":
            z = y[self.thickness_axis] - self.reference_point[self.thickness_axis]
            modes = np.zeros((6, 6), dtype=float)
            modes[0, 0] = 1.0
            modes[1, 1] = 1.0
            modes[5, 2] = 1.0
            modes[0, 3] = z
            modes[1, 4] = z
            modes[5, 5] = z
            return modes
        if self.kind == "euler_bernoulli_beam":
            y2_axis, y3_axis = self.cross_section_axes
            y2 = y[y2_axis] - self.reference_point[y2_axis]
            y3 = y[y3_axis] - self.reference_point[y3_axis]
            modes = np.zeros((6, 4), dtype=float)
            modes[0, 0] = 1.0
            modes[4, 1] = y2
            modes[5, 1] = -y3
            modes[0, 2] = y3
            modes[0, 3] = -y2
            return modes
        raise ValueError(f"unsupported macro model {self.kind!r}")

    def strain_modes_torch(self, point: object, torch: object) -> object:
        """Torch equivalent of :meth:`strain_modes` for TensorMesh assembly."""

        zero = point.new_zeros(())  # type: ignore[attr-defined]
        one = zero + 1.0
        if self.kind == "cauchy_3d":
            return torch.eye(6, dtype=point.dtype, device=point.device)  # type: ignore[attr-defined]
        if self.kind == "kirchhoff_love_plate":
            z = point[self.thickness_axis] - self.reference_point[self.thickness_axis]  # type: ignore[index]
            return torch.stack(
                [
                    torch.stack([one, zero, zero, z, zero, zero]),
                    torch.stack([zero, one, zero, zero, z, zero]),
                    torch.stack([zero, zero, zero, zero, zero, zero]),
                    torch.stack([zero, zero, zero, zero, zero, zero]),
                    torch.stack([zero, zero, zero, zero, zero, zero]),
                    torch.stack([zero, zero, one, zero, zero, z]),
                ]
            )
        if self.kind == "euler_bernoulli_beam":
            y2_axis, y3_axis = self.cross_section_axes
            y2 = point[y2_axis] - self.reference_point[y2_axis]  # type: ignore[index]
            y3 = point[y3_axis] - self.reference_point[y3_axis]  # type: ignore[index]
            return torch.stack(
                [
                    torch.stack([one, zero, y3, -y2]),
                    torch.stack([zero, zero, zero, zero]),
                    torch.stack([zero, zero, zero, zero]),
                    torch.stack([zero, zero, zero, zero]),
                    torch.stack([zero, y2, zero, zero]),
                    torch.stack([zero, -y3, zero, zero]),
                ]
            )
        raise ValueError(f"unsupported macro model {self.kind!r}")

    def strain_modes_batch(self, points: object, torch: object) -> object:
        """Vectorized :meth:`strain_modes_torch` over a batch of SG points.

        ``points`` has shape ``[..., 3]`` (physical SG coordinates) and the
        return value has shape ``[..., 6, n_macro]``. This is the differentiable
        kernel used by the autograd assembly path; gradients flow through the
        point coordinates for geometry-dependent macro models.
        """

        batch = tuple(points.shape[:-1])  # type: ignore[attr-defined]
        dtype = points.dtype  # type: ignore[attr-defined]
        device = points.device  # type: ignore[attr-defined]
        if self.kind == "cauchy_3d":
            eye = torch.eye(6, dtype=dtype, device=device)
            return eye.broadcast_to((*batch, 6, 6))

        zero = torch.zeros(batch, dtype=dtype, device=device)
        one = torch.ones(batch, dtype=dtype, device=device)
        if self.kind == "kirchhoff_love_plate":
            z = points[..., self.thickness_axis] - self.reference_point[self.thickness_axis]  # type: ignore[index]
            rows = [
                [one, zero, zero, z, zero, zero],
                [zero, one, zero, zero, z, zero],
                [zero, zero, zero, zero, zero, zero],
                [zero, zero, zero, zero, zero, zero],
                [zero, zero, zero, zero, zero, zero],
                [zero, zero, one, zero, zero, z],
            ]
            return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
        if self.kind == "euler_bernoulli_beam":
            y2_axis, y3_axis = self.cross_section_axes
            y2 = points[..., y2_axis] - self.reference_point[y2_axis]  # type: ignore[index]
            y3 = points[..., y3_axis] - self.reference_point[y3_axis]  # type: ignore[index]
            rows = [
                [one, zero, y3, -y2],
                [zero, zero, zero, zero],
                [zero, zero, zero, zero],
                [zero, zero, zero, zero],
                [zero, y2, zero, zero],
                [zero, -y3, zero, zero],
            ]
            return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
        raise ValueError(f"unsupported macro model {self.kind!r}")

    def normalization(self, mesh: SolidMesh, sg_measure: float) -> float:
        """Return the MSG omega factor for this macroscopic model."""

        if self.kind == "cauchy_3d":
            return float(sg_measure)
        if self.kind == "kirchhoff_love_plate":
            return _active_extent_product(mesh, self.inplane_axes)
        if self.kind == "euler_bernoulli_beam":
            return _active_extent_product(mesh, (self.axial_axis,))
        raise ValueError(f"unsupported macro model {self.kind!r}")

    def metadata(self) -> dict[str, object]:
        data: dict[str, object] = {
            "macro_model": self.kind,
            "macro_labels": list(self.labels),
            "reference_point": list(self.reference_point),
        }
        if self.kind == "kirchhoff_love_plate":
            data.update(
                {
                    "thickness_axis": AXIS_NAMES[self.thickness_axis],
                    "inplane_axes": [AXIS_NAMES[axis] for axis in self.inplane_axes],
                }
            )
        if self.kind == "euler_bernoulli_beam":
            data.update(
                {
                    "axial_axis": AXIS_NAMES[self.axial_axis],
                    "cross_section_axes": [AXIS_NAMES[axis] for axis in self.cross_section_axes],
                }
            )
        return data


def macro_model_from_analysis(analysis: dict[str, object], mesh: SolidMesh) -> MacroModel:
    """Create a :class:`MacroModel` from an analysis config block."""

    analysis_type = str(analysis.get("type", "msg_3d_cauchy")).lower()
    spec_raw = analysis.get("macro_model", analysis.get("model"))
    if isinstance(spec_raw, dict):
        spec = spec_raw
        key = str(spec.get("type", analysis_type)).lower()
    elif spec_raw is not None:
        spec = analysis
        key = str(spec_raw).lower()
    else:
        spec = analysis
        key = analysis_type
    return macro_model_from_kind(key, mesh=mesh, spec=spec)


def macro_model_from_kind(kind: str | MacroModel, *, mesh: SolidMesh, spec: dict[str, object] | None = None) -> MacroModel:
    """Create a macroscopic model by name."""

    if isinstance(kind, MacroModel):
        return kind
    key = _normalize_macro_kind(kind)
    data = spec or {}
    reference_point = _reference_point(data, mesh)
    if key == "cauchy_3d":
        return MacroModel(
            kind=key,
            labels=("e11", "e22", "e33", "2e23", "2e13", "2e12"),
            reference_point=reference_point,
        )
    if key == "kirchhoff_love_plate":
        thickness_axis = _axis_index(data.get("thickness_axis", "z"))
        inplane_axes = _axis_tuple(data.get("inplane_axes", [axis for axis in range(3) if axis != thickness_axis]))
        if len(inplane_axes) != 2 or thickness_axis in inplane_axes:
            raise ValueError("plate macro model requires two in-plane axes distinct from thickness_axis")
        return MacroModel(
            kind=key,
            labels=("e11", "e22", "2e12", "k11", "k22", "2k12"),
            reference_point=reference_point,
            thickness_axis=thickness_axis,
            inplane_axes=inplane_axes,  # type: ignore[arg-type]
        )
    if key == "euler_bernoulli_beam":
        axial_axis = _axis_index(data.get("axial_axis", "x"))
        cross_section_axes = _axis_tuple(
            data.get("cross_section_axes", [axis for axis in range(3) if axis != axial_axis])
        )
        if len(cross_section_axes) != 2 or axial_axis in cross_section_axes:
            raise ValueError("beam macro model requires two cross-section axes distinct from axial_axis")
        return MacroModel(
            kind=key,
            labels=("e1", "k1", "k2", "k3"),
            reference_point=reference_point,
            axial_axis=axial_axis,
            cross_section_axes=cross_section_axes,  # type: ignore[arg-type]
        )
    raise ValueError(f"unsupported macro model {kind!r}")


def default_constraints_for_macro_model(model: MacroModel, mesh: SolidMesh | None = None) -> list[dict[str, object]]:
    """Return conservative default constraints for a macroscopic model."""

    if model.kind == "cauchy_3d":
        return [{"type": "periodic", "axes": ["x", "y", "z"]}, {"type": "mean_zero"}]
    if model.kind == "kirchhoff_love_plate":
        constraints: list[dict[str, object]] = []
        periodic_axes = _active_axis_names(mesh, model.inplane_axes)
        if periodic_axes:
            constraints.append({"type": "periodic", "axes": periodic_axes})
        constraints.extend([{"type": "mean_zero"}, {"type": "rotation_zero"}])
        return constraints
    if model.kind == "euler_bernoulli_beam":
        constraints = []
        periodic_axes = _active_axis_names(mesh, (model.axial_axis,))
        if periodic_axes:
            constraints.append({"type": "periodic", "axes": periodic_axes})
        pair = [[AXIS_NAMES[model.cross_section_axes[0]], AXIS_NAMES[model.cross_section_axes[1]]]]
        constraints.extend([{"type": "mean_zero"}, {"type": "rotation_zero", "pairs": pair}])
        return constraints
    return [{"type": "mean_zero"}]


def _normalize_macro_kind(kind: str) -> str:
    key = kind.lower()
    aliases = {
        "msg": "cauchy_3d",
        "cauchy": "cauchy_3d",
        "cauchy_3d": "cauchy_3d",
        "3d_cauchy": "cauchy_3d",
        "msg_3d_cauchy": "cauchy_3d",
        "plate": "kirchhoff_love_plate",
        "kirchhoff_plate": "kirchhoff_love_plate",
        "kirchhoff_love": "kirchhoff_love_plate",
        "kirchhoff_love_plate": "kirchhoff_love_plate",
        "msg_plate": "kirchhoff_love_plate",
        "msg_kirchhoff_love_plate": "kirchhoff_love_plate",
        "beam": "euler_bernoulli_beam",
        "euler_beam": "euler_bernoulli_beam",
        "euler_bernoulli": "euler_bernoulli_beam",
        "euler_bernoulli_beam": "euler_bernoulli_beam",
        "msg_beam": "euler_bernoulli_beam",
        "msg_euler_bernoulli_beam": "euler_bernoulli_beam",
    }
    if key not in aliases:
        raise ValueError(f"unsupported macro model {kind!r}")
    return aliases[key]


def _reference_point(data: dict[str, object], mesh: SolidMesh) -> tuple[float, float, float]:
    value = data.get("reference_point", data.get("origin"))
    if value is not None:
        point = np.asarray(value, dtype=float)
        if point.shape != (3,):
            raise ValueError("reference_point must have shape (3,)")
        return (float(point[0]), float(point[1]), float(point[2]))
    mins = np.min(mesh.nodes, axis=0)
    maxs = np.max(mesh.nodes, axis=0)
    point = 0.5 * (mins + maxs)
    return (float(point[0]), float(point[1]), float(point[2]))


def _axis_index(axis: object) -> int:
    key = axis.lower() if isinstance(axis, str) else axis
    if key not in AXIS_MAP:
        raise ValueError(f"unsupported axis {axis!r}")
    return AXIS_MAP[key]


def _axis_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        return (_axis_index(value),)
    return tuple(_axis_index(axis) for axis in value)


def _active_extent_product(mesh: SolidMesh, axes: tuple[int, ...]) -> float:
    active_axes = set(mesh.active_axes)
    factor = 1.0
    for axis in axes:
        if axis not in active_axes:
            continue
        extent = float(np.max(mesh.nodes[:, axis]) - np.min(mesh.nodes[:, axis]))
        if extent > 0.0:
            factor *= extent
    return factor


def _active_axis_names(mesh: SolidMesh | None, axes: tuple[int, ...]) -> list[str]:
    if mesh is None:
        return []
    active_axes = set(mesh.active_axes)
    return [AXIS_NAMES[axis] for axis in axes if axis in active_axes]
