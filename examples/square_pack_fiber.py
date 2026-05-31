"""Build and run a gmsh-based 2D SG square-pack fiber benchmark.

This example intentionally keeps mesh generation outside the OpenMSG core. It
uses gmsh to create a circular fiber in a square matrix cross-section, extracts
a periodic Tri3 2D Structure Genome, runs MSG homogenization, and compares the
apparent moduli with simple rules of mixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openmsg.config import run_config
from openmsg.materials import engineering_constants_from_stiffness


def build_square_pack_config(
    *,
    mesh_size: float,
    fiber_volume_fraction: float,
    matrix_E: float,
    matrix_nu: float,
    fiber_E: float,
    fiber_nu: float,
) -> dict[str, object]:
    """Return an explicit OpenMSG input generated from a gmsh Tri3 2D SG."""

    if not (0.0 < fiber_volume_fraction < math.pi / 4.0):
        raise ValueError(
            "fiber_volume_fraction must be between 0 and pi/4 for a circular "
            "fiber in a unit square"
        )
    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive")

    nodes, connectivity, material_ids = gmsh_square_pack_tri3_mesh(
        fiber_volume_fraction=fiber_volume_fraction,
        mesh_size=mesh_size,
    )
    areas = triangle_areas(nodes, connectivity)
    fiber_mask = np.array([name == "fiber" for name in material_ids], dtype=bool)
    realized_vf = float(areas[fiber_mask].sum() / areas.sum())

    return {
        "analysis": {
            "type": "msg_3d_cauchy",
            "constraints": [
                {"type": "periodic", "axes": ["x", "y"]},
                {"type": "mean_zero"},
            ],
        },
        "metadata": {
            "example": "gmsh_square_pack_fiber_2d_sg",
            "target_fiber_volume_fraction": fiber_volume_fraction,
            "realized_fiber_volume_fraction": realized_vf,
            "mesh_size": mesh_size,
        },
        "materials": {
            "matrix": {"type": "isotropic", "E": matrix_E, "nu": matrix_nu},
            "fiber": {"type": "isotropic", "E": fiber_E, "nu": fiber_nu},
        },
        "mesh": {
            "active_axes": ["x", "y"],
            "nodes": nodes.tolist(),
            "elements": [
                {
                    "type": "tri3",
                    "connectivity": connectivity.tolist(),
                    "material": material_ids,
                }
            ],
        },
    }


def gmsh_square_pack_tri3_mesh(
    *,
    fiber_volume_fraction: float,
    mesh_size: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a periodic Tri3 square-pack cross-section with gmsh."""

    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("square_pack_fiber.py requires the optional gmsh package") from exc

    radius = math.sqrt(fiber_volume_fraction / math.pi)
    tol = 1e-4

    gmsh.initialize()
    try:
        gmsh.option.set_number("General.Verbosity", 0)
        gmsh.model.add("openmsg_square_pack")

        gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        gmsh.model.occ.add_disk(0.5, 0.5, 0.0, radius, radius)
        gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
        gmsh.model.occ.synchronize()

        surfaces = [tag for _, tag in gmsh.model.get_entities(2)]
        by_area = sorted((gmsh.model.occ.get_mass(2, tag), tag) for tag in surfaces)
        gmsh.model.add_physical_group(2, [by_area[0][1]], tag=1, name="fiber")
        gmsh.model.add_physical_group(2, [by_area[1][1]], tag=2, name="matrix")

        def boundary_curves(xlo: float, ylo: float, xhi: float, yhi: float) -> list[int]:
            return [
                tag
                for _, tag in gmsh.model.get_entities_in_bounding_box(
                    xlo, ylo, -tol, xhi, yhi, tol, 1
                )
            ]

        left = boundary_curves(-tol, -tol, tol, 1.0 + tol)
        right = boundary_curves(1.0 - tol, -tol, 1.0 + tol, 1.0 + tol)
        bottom = boundary_curves(-tol, -tol, 1.0 + tol, tol)
        top = boundary_curves(-tol, 1.0 - tol, 1.0 + tol, 1.0 + tol)

        tx = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        ty = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        if right and left:
            gmsh.model.mesh.set_periodic(1, right, left, tx)
        if top and bottom:
            gmsh.model.mesh.set_periodic(1, top, bottom, ty)

        gmsh.option.set_number("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.set_number("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.get_nodes()
        node_map = {int(tag): idx for idx, tag in enumerate(node_tags)}
        nodes = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]

        connectivity: list[list[int]] = []
        material_ids: list[str] = []
        for physical_tag, material_name in ((1, "fiber"), (2, "matrix")):
            for entity in gmsh.model.get_entities_for_physical_group(2, physical_tag):
                element_types, _, node_tag_blocks = gmsh.model.mesh.get_elements(2, int(entity))
                for element_type, node_tags_block in zip(element_types, node_tag_blocks):
                    if element_type != 2:
                        continue
                    for row in np.asarray(node_tags_block, dtype=int).reshape(-1, 3):
                        connectivity.append([node_map[int(node)] for node in row])
                        material_ids.append(material_name)
    finally:
        gmsh.finalize()

    if not connectivity:
        raise RuntimeError("gmsh did not produce any Tri3 elements")
    return nodes, np.asarray(connectivity, dtype=int), material_ids


def triangle_areas(nodes: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Return positive areas for Tri3 elements in the 2D cross-section."""

    points = nodes[connectivity]
    v0 = points[:, 1] - points[:, 0]
    v1 = points[:, 2] - points[:, 0]
    return 0.5 * np.abs(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0])


def rule_of_mixture(matrix_E: float, fiber_E: float, vf: float) -> dict[str, float]:
    vm = 1.0 - vf
    return {
        "voigt_E_z": vf * fiber_E + vm * matrix_E,
        "reuss_E_transverse": 1.0 / (vf / fiber_E + vm / matrix_E),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a gmsh 2D SG square-pack MSG benchmark.")
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--vf", type=float, default=0.35)
    parser.add_argument("--matrix-E", type=float, default=3.5)
    parser.add_argument("--matrix-nu", type=float, default=0.35)
    parser.add_argument("--fiber-E", type=float, default=70.0)
    parser.add_argument("--fiber-nu", type=float, default=0.22)
    parser.add_argument(
        "--write-input",
        type=Path,
        help="Optional path for the generated Tri3 JSON input.",
    )
    args = parser.parse_args(argv)

    config = build_square_pack_config(
        mesh_size=args.mesh_size,
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
    metadata = config["metadata"]  # type: ignore[index]
    vf = float(metadata["realized_fiber_volume_fraction"])  # type: ignore[index]
    rom = rule_of_mixture(args.matrix_E, args.fiber_E, vf)
    summary = {
        "target_fiber_volume_fraction": args.vf,
        "realized_fiber_volume_fraction": vf,
        "mesh_size": args.mesh_size,
        "n_nodes": result.metadata["n_nodes"],
        "n_elements": result.metadata["n_elements"],
        "element_types": result.metadata["element_types"],
        "apparent": constants,
        "rules_of_mixture": rom,
        "relative_difference": {
            "E_z_vs_voigt": (constants["E_z"] - rom["voigt_E_z"]) / rom["voigt_E_z"],
            "E_x_vs_reuss": (
                (constants["E_x"] - rom["reuss_E_transverse"])
                / rom["reuss_E_transverse"]
            ),
            "E_y_vs_reuss": (
                (constants["E_y"] - rom["reuss_E_transverse"])
                / rom["reuss_E_transverse"]
            ),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
