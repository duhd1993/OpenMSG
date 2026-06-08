from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.macro import MacroModel, default_constraints_for_macro_model
from openmsg.mesh import SolidMesh


class MacroModelTests(unittest.TestCase):
    def test_plate_strain_modes_accept_torch_quadrature_batch(self) -> None:
        model = MacroModel(
            kind="kirchhoff_love_plate",
            labels=("e11", "e22", "2e12", "k11", "k22", "2k12"),
            reference_point=(0.0, 0.0, 0.0),
        )
        points = torch.tensor(
            [
                [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]],
                [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]],
            ],
            dtype=torch.float64,
        )

        modes = model.strain_modes(points)

        self.assertEqual(tuple(modes.shape), (2, 2, 6, 6))
        self.assertEqual(modes[0, 0, 0, 0], 1.0)
        self.assertEqual(modes[0, 0, 1, 1], 1.0)
        self.assertEqual(modes[0, 0, 5, 2], 1.0)
        self.assertEqual(modes[0, 0, 0, 3], 0.5)
        self.assertEqual(modes[0, 1, 0, 3], -0.5)

    def test_cauchy_strain_modes_accept_torch_batch(self) -> None:
        model = MacroModel(kind="cauchy_3d", labels=("e11", "e22", "e33", "2e23", "2e13", "2e12"))

        modes = model.strain_modes(torch.zeros((2, 3, 3), dtype=torch.float64))

        self.assertEqual(tuple(modes.shape), (2, 3, 6, 6))
        np.testing.assert_allclose(modes[1, 2].detach().cpu().numpy(), np.eye(6))

    def test_beam_strain_modes_accept_torch_batch(self) -> None:
        model = MacroModel(
            kind="euler_bernoulli_beam",
            labels=("e1", "k1", "k2", "k3"),
            reference_point=(0.0, 0.0, 0.0),
        )
        points = torch.tensor([[[0.0, 2.0, 3.0], [0.0, -1.0, 4.0]]], dtype=torch.float64)

        modes = model.strain_modes(points)

        self.assertEqual(tuple(modes.shape), (1, 2, 6, 4))
        expected = np.zeros((6, 4))
        expected[0] = [1.0, 0.0, 3.0, -2.0]
        expected[4] = [0.0, 2.0, 0.0, 0.0]
        expected[5] = [0.0, -3.0, 0.0, 0.0]
        np.testing.assert_allclose(modes[0, 0].detach().cpu().numpy(), expected)

    def test_strain_modes_reject_numpy_input(self) -> None:
        model = MacroModel(kind="cauchy_3d", labels=("e11", "e22", "e33", "2e23", "2e13", "2e12"))

        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            model.strain_modes(np.zeros(3))

    def test_structural_default_constraints_match_msg_theory(self) -> None:
        plate_model = MacroModel(
            kind="kirchhoff_love_plate",
            labels=("e11", "e22", "2e12", "k11", "k22", "2k12"),
        )
        plate_mesh = SolidMesh(
            nodes=[[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]],
            elements=[{"type": "line2", "connectivity": [[0, 1]], "material": "m"}],
        )

        beam_model = MacroModel(kind="euler_bernoulli_beam", labels=("e1", "k1", "k2", "k3"))
        beam_mesh = SolidMesh(
            nodes=[
                [0.0, -0.5, -0.5],
                [0.0, 0.5, -0.5],
                [0.0, 0.5, 0.5],
                [0.0, -0.5, 0.5],
            ],
            elements=[{"type": "quad4", "connectivity": [[0, 1, 2, 3]], "material": "m"}],
        )

        self.assertEqual(
            default_constraints_for_macro_model(plate_model, plate_mesh),
            [{"type": "mean_zero"}],
        )
        self.assertEqual(
            default_constraints_for_macro_model(beam_model, beam_mesh),
            [{"type": "mean_zero"}, {"type": "rotation_zero", "pairs": [["y", "z"]]}],
        )


if __name__ == "__main__":
    unittest.main()
