"""Kirchhoff-Love laminate ABD utilities for 1D SG plate models."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


IN_PLANE = np.array([0, 1, 5], dtype=int)
OUT_OF_PLANE = np.array([2, 3, 4], dtype=int)


@dataclass(frozen=True)
class Ply:
    """A laminate ply definition."""

    stiffness: np.ndarray
    thickness: float
    angle_deg: float = 0.0
    material: str | None = None


@dataclass(frozen=True)
class LaminateABDResult:
    """ABD result for a Kirchhoff-Love laminate."""

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    ABD: np.ndarray
    z: np.ndarray
    Qbar: list[np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self, *, include_internal: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "D": self.D.tolist(),
            "ABD": self.ABD.tolist(),
            "z": self.z.tolist(),
            "metadata": self.metadata,
        }
        if include_internal:
            data["Qbar"] = [Q.tolist() for Q in self.Qbar]
        return data


def plane_stress_reduced_stiffness(C: np.ndarray) -> np.ndarray:
    """Return the in-plane plane-stress stiffness ``Q``.

    The returned order is ``[e11, e22, 2e12]`` mapped to
    ``[s11, s22, s12]``. If a 3x3 matrix is passed, it is assumed to already be
    in this reduced order.
    """

    stiffness = np.asarray(C, dtype=float)
    if stiffness.shape == (3, 3):
        return _assert_symmetric_3x3(stiffness)
    if stiffness.shape != (6, 6):
        raise ValueError("stiffness must have shape (6, 6) or (3, 3)")
    Cii = stiffness[np.ix_(IN_PLANE, IN_PLANE)]
    Cio = stiffness[np.ix_(IN_PLANE, OUT_OF_PLANE)]
    Coi = stiffness[np.ix_(OUT_OF_PLANE, IN_PLANE)]
    Coo = stiffness[np.ix_(OUT_OF_PLANE, OUT_OF_PLANE)]
    Q = Cii - Cio @ np.linalg.solve(Coo, Coi)
    return _assert_symmetric_3x3(0.5 * (Q + Q.T))


def transform_reduced_stiffness_inplane(Q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a reduced in-plane stiffness matrix by a ply angle."""

    Q = _assert_symmetric_3x3(Q)
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    axes = np.array([[c, -s], [s, c]], dtype=float)
    T_eps = _inplane_strain_transform(axes)
    T_sig = _inplane_stress_transform(axes)
    Qbar = T_sig @ Q @ T_eps
    return _assert_symmetric_3x3(0.5 * (Qbar + Qbar.T))


def laminate_abd(plies: list[Ply], *, z_bottom: float | None = None) -> LaminateABDResult:
    """Compute Kirchhoff-Love laminate ``A``, ``B``, and ``D`` matrices."""

    if not plies:
        raise ValueError("at least one ply is required")
    thicknesses = np.asarray([ply.thickness for ply in plies], dtype=float)
    if np.any(thicknesses <= 0.0):
        raise ValueError("all ply thicknesses must be positive")

    total_thickness = float(np.sum(thicknesses))
    z0 = -0.5 * total_thickness if z_bottom is None else float(z_bottom)
    z = np.concatenate(([z0], z0 + np.cumsum(thicknesses)))

    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)
    qbars: list[np.ndarray] = []
    for idx, ply in enumerate(plies):
        Q = plane_stress_reduced_stiffness(ply.stiffness)
        Qbar = transform_reduced_stiffness_inplane(Q, ply.angle_deg)
        qbars.append(Qbar)
        zk = z[idx + 1]
        zk_1 = z[idx]
        A += Qbar * (zk - zk_1)
        B += 0.5 * Qbar * (zk**2 - zk_1**2)
        D += (1.0 / 3.0) * Qbar * (zk**3 - zk_1**3)

    ABD = np.block([[A, B], [B, D]])
    return LaminateABDResult(
        A=0.5 * (A + A.T),
        B=0.5 * (B + B.T),
        D=0.5 * (D + D.T),
        ABD=0.5 * (ABD + ABD.T),
        z=z,
        Qbar=qbars,
        metadata={
            "model": "kirchhoff_love_laminate",
            "n_plies": len(plies),
            "total_thickness": total_thickness,
        },
    )


def laminate_abd_from_config(
    *,
    materials: dict[str, np.ndarray],
    laminate: dict[str, object],
) -> LaminateABDResult:
    """Build a laminate ABD result from a config laminate block."""

    plies: list[Ply] = []
    for item in laminate.get("plies", []):  # type: ignore[union-attr]
        if not isinstance(item, dict):
            raise ValueError("each laminate ply must be an object")
        material = str(item["material"])
        if material not in materials:
            raise KeyError(f"laminate references undefined material {material!r}")
        plies.append(
            Ply(
                stiffness=materials[material],
                thickness=float(item["thickness"]),
                angle_deg=float(item.get("angle", item.get("angle_deg", 0.0))),
                material=material,
            )
        )
    z_bottom = laminate.get("z_bottom")
    result = laminate_abd(plies, z_bottom=None if z_bottom is None else float(z_bottom))
    result.metadata["materials"] = [ply.material for ply in plies]
    result.metadata["angles_deg"] = [ply.angle_deg for ply in plies]
    return result


def _inplane_strain_transform(axes: np.ndarray) -> np.ndarray:
    columns = []
    for vec in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        tensor = np.array([[vec[0], 0.5 * vec[2]], [0.5 * vec[2], vec[1]]], dtype=float)
        local = axes.T @ tensor @ axes
        columns.append(np.array([local[0, 0], local[1, 1], 2.0 * local[0, 1]]))
    return np.column_stack(columns)


def _inplane_stress_transform(axes: np.ndarray) -> np.ndarray:
    columns = []
    for vec in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        tensor = np.array([[vec[0], vec[2]], [vec[2], vec[1]]], dtype=float)
        global_tensor = axes @ tensor @ axes.T
        columns.append(np.array([global_tensor[0, 0], global_tensor[1, 1], global_tensor[0, 1]]))
    return np.column_stack(columns)


def _assert_symmetric_3x3(Q: np.ndarray) -> np.ndarray:
    matrix = np.asarray(Q, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"reduced stiffness must have shape (3, 3), got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
        raise ValueError("reduced stiffness must be symmetric")
    return matrix

