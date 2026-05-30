from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import importlib.util

import numpy as np

from openmsg.config import run_config
from openmsg.materials import isotropic_stiffness
from openmsg.mesh import mesh_from_config


class ConfigTests(unittest.TestCase):
    def test_example_config_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_config(root / "examples" / "homogeneous_3d.json")
        expected = isotropic_stiffness(100.0, 0.25)
        np.testing.assert_allclose(result.Dbar, expected, rtol=1e-10, atol=1e-10)

    def test_tet4_example_config_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_config(root / "examples" / "homogeneous_tet4.json")
        expected = isotropic_stiffness(100.0, 0.25)
        np.testing.assert_allclose(result.Dbar, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.metadata["n_elements"], 6)

    def test_reduced_sg_example_configs_run(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = isotropic_stiffness(100.0, 0.25)
        for name, dimension in (("homogeneous_line2.json", 1), ("homogeneous_quad4.json", 2)):
            with self.subTest(name=name):
                result = run_config(root / "examples" / name)
                np.testing.assert_allclose(result.Dbar, expected, rtol=1e-10, atol=1e-10)
                self.assertEqual(result.metadata["sg_dimension"], dimension)

    def test_config_can_request_dehomogenization(self) -> None:
        config = {
            "analysis": {
                "type": "msg_3d_cauchy",
                "constraints": ["periodic", "mean_zero"],
            },
            "materials": {"matrix": {"type": "isotropic", "E": 100.0, "nu": 0.25}},
            "mesh": {
                "type": "hex8",
                "nodes": [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                "elements": [{"nodes": [0, 1, 2, 3, 4, 5, 6, 7], "material": "matrix"}],
            },
            "dehomogenization": {"macro_strains": {"unit_z": [0, 0, 0.01, 0, 0, 0]}},
        }

        result = run_config(config)
        self.assertIn("unit_z", result.local_fields)
        self.assertEqual(result.local_fields["unit_z"]["strain"].shape, (8, 6))

    def test_example_square_pack_script_can_write_explicit_input(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "examples" / "square_pack_fiber.py"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "square_pack.json"
            spec = importlib.util.spec_from_file_location("square_pack_fiber_example", script)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            config = module.build_square_pack_config(
                cells_xy=4,
                cells_z=1,
                fiber_volume_fraction=0.25,
                matrix_E=3.5,
                matrix_nu=0.35,
                fiber_E=70.0,
                fiber_nu=0.22,
            )
            output.write_text(__import__("json").dumps(config), encoding="utf-8")
            result = run_config(output)
            self.assertGreater(result.Dbar[2, 2], result.Dbar[0, 0])

    def test_analysis_backend_field_is_rejected(self) -> None:
        config = {
            "analysis": {"type": "msg_3d_cauchy", "backend": "tensormesh"},
            "materials": {"m": {"type": "isotropic", "E": 100.0, "nu": 0.25}},
            "mesh": {
                "type": "line2",
                "active_axes": ["z"],
                "nodes": [[0.0], [1.0]],
                "elements": [{"nodes": [0, 1], "material": "m"}],
            },
        }
        with self.assertRaisesRegex(ValueError, "analysis.backend has been removed"):
            run_config(config)

    def test_structured_hex_is_not_core_input_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported mesh type"):
            mesh_from_config(
                {
                    "type": "structured_hex",
                    "bounds": [[0, 1], [0, 1], [0, 1]],
                    "cells": [1, 1, 1],
                    "default_material": "m",
                }
            )

    def test_meshio_input_reports_missing_dependency(self) -> None:
        if importlib.util.find_spec("meshio") is not None:
            self.skipTest("meshio is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "meshio"):
            mesh_from_config({"type": "meshio", "path": "dummy.vtu"})


if __name__ == "__main__":
    unittest.main()
