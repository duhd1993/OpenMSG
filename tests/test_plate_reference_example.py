from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from openmsg.materials import isotropic_stiffness


def load_plate_example() -> object:
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "plate_msg_vs_laminate_abd.py"
    spec = importlib.util.spec_from_file_location("plate_msg_vs_laminate_abd_example", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PlateReferenceExampleTests(unittest.TestCase):
    def test_reference_plane_stress_reduced_stiffness_for_isotropic_material(self) -> None:
        module = load_plate_example()
        young = 100.0
        nu = 0.25
        Q = module.plane_stress_reduced_stiffness(isotropic_stiffness(young, nu))
        expected = young / (1.0 - nu**2) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )

        np.testing.assert_allclose(Q, expected, rtol=1e-12, atol=1e-12)

    def test_example_compares_msg_plate_to_classical_abd(self) -> None:
        module = load_plate_example()
        comparison = module.run_single_layer_comparison(n_elements=16)

        np.testing.assert_allclose(comparison.ABD_msg[:3, :3], comparison.ABD_reference[:3, :3], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(comparison.ABD_msg[3:, 3:], comparison.ABD_reference[3:, 3:], rtol=6e-4, atol=6e-3)
        self.assertLess(comparison.max_abs_error, 6e-3)


if __name__ == "__main__":
    unittest.main()
