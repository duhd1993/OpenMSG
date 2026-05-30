"""Compare a 1D MSG Kirchhoff-Love plate solve with classical laminate ABD.

This is intentionally an example-level reference calculation, not part of the
OpenMSG solver API. The core library only solves MSG problems.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math

import numpy as np

from openmsg import SolidMesh, effective_stiffness, isotropic_stiffness


IN_PLANE = np.array([0, 1, 5], dtype=int)
OUT_OF_PLANE = np.array([2, 3, 4], dtype=int)


@dataclass(frozen=True)
class Ply:
    """A classical laminate ply used by this example's reference calculation."""

    stiffness: object
    thickness: float
    angle_deg: float = 0.0
    material: str | None = None


@dataclass(frozen=True)
class LaminateABDResult:
    """Classical laminate ABD result for this example."""

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    ABD: np.ndarray
    z: np.ndarray
    Qbar: list[np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlateComparison:
    """MSG-vs-classical comparison data."""

    ABD_msg: np.ndarray
    ABD_reference: np.ndarray
    max_abs_error: float
    relative_error: float


def plane_stress_reduced_stiffness(C: object) -> np.ndarray:
    """Return the reduced in-plane plane-stress stiffness in [e11, e22, 2e12] order."""

    stiffness = _as_numpy_array(C)
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
    """Compute the classical Kirchhoff-Love laminate ABD reference matrix."""

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
            "model": "classical_laminate_reference",
            "n_plies": len(plies),
            "total_thickness": total_thickness,
        },
    )


def line_thickness_mesh(*, thickness: float = 1.0, n_elements: int = 16) -> SolidMesh:
    """Build a simple 1D thickness SG mesh for the comparison."""

    z = np.linspace(-0.5 * thickness, 0.5 * thickness, n_elements + 1)
    nodes = np.array([[0.0, 0.0, zi] for zi in z], dtype=float)
    elements = np.array([[idx, idx + 1] for idx in range(n_elements)], dtype=int)
    return SolidMesh(nodes=nodes, elements=elements, material_ids=("m",) * n_elements, element_type="line2")


def run_single_layer_comparison(
    *,
    young: float = 100.0,
    nu: float = 0.25,
    thickness: float = 1.0,
    n_elements: int = 16,
) -> PlateComparison:
    """Compare MSG plate ABD with the classical ABD of one centered isotropic layer."""

    stiffness = isotropic_stiffness(young, nu)
    mesh = line_thickness_mesh(thickness=thickness, n_elements=n_elements)
    msg = effective_stiffness(mesh=mesh, material_stiffness={"m": stiffness}, macro_model="kirchhoff_love_plate")
    reference = laminate_abd([Ply(stiffness=stiffness, thickness=thickness)])
    msg_abd = msg.Dbar.detach().cpu().numpy()
    diff = msg_abd - reference.ABD
    max_abs_error = float(np.max(np.abs(diff)))
    scale = max(float(np.max(np.abs(reference.ABD))), 1.0)
    return PlateComparison(
        ABD_msg=msg_abd,
        ABD_reference=reference.ABD,
        max_abs_error=max_abs_error,
        relative_error=max_abs_error / scale,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=16, help="Number of Line2 elements through thickness.")
    parser.add_argument("--young", type=float, default=100.0, help="Young's modulus for the isotropic layer.")
    parser.add_argument("--nu", type=float, default=0.25, help="Poisson ratio for the isotropic layer.")
    parser.add_argument("--thickness", type=float, default=1.0, help="Plate thickness.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = run_single_layer_comparison(
        young=args.young,
        nu=args.nu,
        thickness=args.thickness,
        n_elements=args.elements,
    )
    np.set_printoptions(precision=8, suppress=True)
    print("MSG ABD:")
    print(comparison.ABD_msg)
    print("\nClassical laminate ABD reference:")
    print(comparison.ABD_reference)
    print(f"\nmax_abs_error: {comparison.max_abs_error:.6e}")
    print(f"relative_error: {comparison.relative_error:.6e}")
    return 0


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


def _as_numpy_array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


if __name__ == "__main__":
    raise SystemExit(main())
