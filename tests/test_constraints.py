from __future__ import annotations

import unittest

import torch

from openmsg.constraints import build_constraint_tensor
from tests.mesh_builders import structured_hex_mesh


class ConstraintTensorTests(unittest.TestCase):
    def test_build_constraint_tensor_preserves_requested_dtype(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        node_weights = torch.ones(mesh.n_nodes, dtype=torch.float32)

        G = build_constraint_tensor(
            mesh,
            [{"type": "periodic", "axes": ["x"]}, {"type": "mean_zero"}],
            node_weights=node_weights,
            dtype=torch.float32,
        )

        self.assertIsInstance(G, torch.Tensor)
        self.assertEqual(G.dtype, torch.float32)
        self.assertEqual(G.shape[1], mesh.n_dof)
        self.assertGreater(G.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
