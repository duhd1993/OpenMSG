"""JSON input handling for OpenMSG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import MSGResult, effective_stiffness
from openmsg.macro import default_constraints_for_macro_model, macro_model_from_analysis
from openmsg.materials import stiffness_from_config
from openmsg.mesh import mesh_from_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON OpenMSG input file."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def run_config(config_or_path: dict[str, Any] | str | Path) -> MSGResult:
    """Run an OpenMSG analysis from a loaded config or JSON file path."""

    base_dir: Path | None = None
    if isinstance(config_or_path, (str, Path)):
        input_path = Path(config_or_path)
        config = load_config(input_path)
        base_dir = input_path.resolve().parent
    else:
        config = config_or_path
    analysis = config.get("analysis", {})
    analysis_type = str(analysis.get("type", "msg_3d_cauchy")).lower()

    materials = {
        str(name): stiffness_from_config(material_config)
        for name, material_config in config.get("materials", {}).items()
    }
    if not materials:
        raise ValueError("at least one material is required")
    supported_msg_types = {
        "msg",
        "msg_3d_cauchy",
        "cauchy_3d",
        "msg_plate",
        "msg_kirchhoff_love_plate",
        "kirchhoff_love_plate",
        "msg_beam",
        "msg_euler_bernoulli_beam",
        "euler_bernoulli_beam",
    }
    if analysis_type not in supported_msg_types:
        raise ValueError(
            "supported analysis types are 'msg_3d_cauchy', 'msg_kirchhoff_love_plate', "
            "and 'msg_euler_bernoulli_beam'"
        )
    if "backend" in analysis:
        raise ValueError("analysis.backend has been removed; OpenMSG uses TensorMesh assembly directly")
    mesh = mesh_from_config(config["mesh"], base_dir=base_dir)
    macro_model = macro_model_from_analysis(analysis, mesh)
    constraints = analysis.get("constraints", default_constraints_for_macro_model(macro_model, mesh))
    linear_solver = str(analysis.get("linear_solver", "auto"))
    result = effective_stiffness(
        mesh=mesh,
        material_stiffness=materials,
        macro_model=macro_model,
        constraints=constraints,
        linear_solver=linear_solver,
    )
    if "metadata" in config:
        result.metadata["input_metadata"] = config["metadata"]
    if "dehomogenization" in config:
        macro_strains = config["dehomogenization"].get("macro_strains", {})  # type: ignore[union-attr]
        if isinstance(macro_strains, list):
            macro_strains = {f"case_{idx}": value for idx, value in enumerate(macro_strains)}
        for name, strain in macro_strains.items():
            result.local_fields[str(name)] = recover_gauss_fields(
                mesh=mesh,
                material_stiffness=materials,
                V0=result.V0,
                macro_strain=np.asarray(strain, dtype=float),
                macro_model=macro_model,
            )
    return result


def write_result_json(result: MSGResult, path: str | Path, *, include_internal: bool = False) -> None:
    """Write a result JSON file."""

    output = result.to_dict(include_internal=include_internal)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


def result_to_json(result: MSGResult, *, include_internal: bool = False) -> str:
    """Return a formatted JSON string for a result."""

    return json.dumps(result.to_dict(include_internal=include_internal), indent=2)
