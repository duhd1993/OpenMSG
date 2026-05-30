"""Build and run a square-pack fiber composite benchmark.

This example intentionally keeps mesh generation outside the OpenMSG core. It
creates an explicit Hex8 JSON input file, runs homogenization, and compares the
apparent moduli with simple rules of mixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openmsg.config import run_config
from openmsg.materials import engineering_constants_from_stiffness


def build_square_pack_config(
    *,
    cells_xy: int,
    cells_z: int,
    fiber_volume_fraction: float,
    matrix_E: float,
    matrix_nu: float,
    fiber_E: float,
    fiber_nu: float,
) -> dict[str, object]:
    if not (0.0 < fiber_volume_fraction < math.pi / 4.0):
        raise ValueError("fiber_volume_fraction must be between 0 and pi/4 for a circular fiber in a unit square")

    nx = ny = int(cells_xy)
    nz = int(cells_z)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("cell counts must be positive")

    xs = [i / nx for i in range(nx + 1)]
    ys = [j / ny for j in range(ny + 1)]
    zs = [k / nz for k in range(nz + 1)]

    def node_id(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    nodes = [[x, y, z] for z in zs for y in ys for x in xs]
    elements: list[dict[str, object]] = []
    radius = math.sqrt(fiber_volume_fraction / math.pi)
    realized_fiber_cells = 0

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cx = (i + 0.5) / nx
                cy = (j + 0.5) / ny
                material = "fiber" if math.hypot(cx - 0.5, cy - 0.5) <= radius else "matrix"
                if material == "fiber":
                    realized_fiber_cells += 1
                elements.append(
                    {
                        "nodes": [
                            node_id(i, j, k),
                            node_id(i + 1, j, k),
                            node_id(i + 1, j + 1, k),
                            node_id(i, j + 1, k),
                            node_id(i, j, k + 1),
                            node_id(i + 1, j, k + 1),
                            node_id(i + 1, j + 1, k + 1),
                            node_id(i, j + 1, k + 1),
                        ],
                        "material": material,
                    }
                )

    realized_vf = realized_fiber_cells / len(elements)
    return {
        "analysis": {
            "type": "msg_3d_cauchy",
            "constraints": [
                {"type": "periodic", "axes": ["x", "y", "z"]},
                {"type": "mean_zero"},
            ],
        },
        "metadata": {
            "example": "square_pack_fiber",
            "target_fiber_volume_fraction": fiber_volume_fraction,
            "realized_fiber_volume_fraction": realized_vf,
        },
        "materials": {
            "matrix": {"type": "isotropic", "E": matrix_E, "nu": matrix_nu},
            "fiber": {"type": "isotropic", "E": fiber_E, "nu": fiber_nu},
        },
        "mesh": {
            "type": "hex8",
            "nodes": nodes,
            "elements": elements,
        },
    }


def rule_of_mixture(matrix_E: float, fiber_E: float, vf: float) -> dict[str, float]:
    vm = 1.0 - vf
    return {
        "voigt_E_z": vf * fiber_E + vm * matrix_E,
        "reuss_E_transverse": 1.0 / (vf / fiber_E + vm / matrix_E),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a square-pack fiber MSG benchmark.")
    parser.add_argument("--cells-xy", type=int, default=8)
    parser.add_argument("--cells-z", type=int, default=1)
    parser.add_argument("--vf", type=float, default=0.35)
    parser.add_argument("--matrix-E", type=float, default=3.5)
    parser.add_argument("--matrix-nu", type=float, default=0.35)
    parser.add_argument("--fiber-E", type=float, default=70.0)
    parser.add_argument("--fiber-nu", type=float, default=0.22)
    parser.add_argument("--write-input", type=Path, help="Optional path for the generated explicit Hex8 JSON input.")
    args = parser.parse_args(argv)

    config = build_square_pack_config(
        cells_xy=args.cells_xy,
        cells_z=args.cells_z,
        fiber_volume_fraction=args.vf,
        matrix_E=args.matrix_E,
        matrix_nu=args.matrix_nu,
        fiber_E=args.fiber_E,
        fiber_nu=args.fiber_nu,
    )
    if args.write_input:
        with args.write_input.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    result = run_config(config)
    constants = engineering_constants_from_stiffness(result.Dbar)
    vf = float(config["metadata"]["realized_fiber_volume_fraction"])  # type: ignore[index]
    rom = rule_of_mixture(args.matrix_E, args.fiber_E, vf)
    summary = {
        "realized_fiber_volume_fraction": vf,
        "apparent": constants,
        "rules_of_mixture": rom,
        "relative_difference": {
            "E_z_vs_voigt": (constants["E_z"] - rom["voigt_E_z"]) / rom["voigt_E_z"],
            "E_x_vs_reuss": (constants["E_x"] - rom["reuss_E_transverse"]) / rom["reuss_E_transverse"],
            "E_y_vs_reuss": (constants["E_y"] - rom["reuss_E_transverse"]) / rom["reuss_E_transverse"],
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
