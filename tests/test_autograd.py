"""End-to-end automatic differentiation tests for the MSG pipeline.

These exercise the TensorMesh-backed assembly and the differentiable
``torch-sla`` solver, verifying that gradients of the homogenized stiffness
``Dbar`` flow back to both the material stiffness and the node geometry.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from openmsg.homogenize import effective_stiffness, homogenize_3d_cauchy
from openmsg.materials import isotropic_stiffness
from tests.mesh_builders import structured_hex_mesh


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(array, dtype=float), dtype=torch.float64)


def _hetero_bar():
    """Two-element bar (matrix | fiber) along x with a movable interface."""

    return structured_hex_mesh(
        bounds=((0, 2), (0, 1), (0, 1)),
        cells=(2, 1, 1),
        default_material="matrix",
        cell_materials=("matrix", "fiber"),
    )


class AutogradTests(unittest.TestCase):
    def test_homogenize_matches_effective_stiffness(self) -> None:
        mesh = _hetero_bar()
        materials = {"matrix": isotropic_stiffness(10.0, 0.3), "fiber": isotropic_stiffness(70.0, 0.2)}

        torch_result = effective_stiffness(mesh=mesh, material_stiffness=materials, macro_model="cauchy_3d")
        numpy_result = homogenize_3d_cauchy(mesh=mesh, material_stiffness=materials)

        np.testing.assert_allclose(
            torch_result.Dbar.detach().cpu().numpy(), numpy_result.Dbar, rtol=1e-10, atol=1e-10
        )

    def test_material_gradient_scale_is_analytic_for_homogeneous_cube(self) -> None:
        # For a homogeneous cell Dbar == C, so d(Dbar)/d(scale) == C exactly.
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C0 = _tensor(isotropic_stiffness(120.0, 0.3))
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        result = effective_stiffness(
            mesh=mesh, material_stiffness={"m": scale * C0}, macro_model="cauchy_3d"
        )
        # Differentiate a few independent entries of Dbar wrt the scale.
        for i, j, expected in ((0, 0, C0[0, 0]), (0, 1, C0[0, 1]), (3, 3, C0[3, 3])):
            (grad,) = torch.autograd.grad(result.Dbar[i, j], scale, retain_graph=True)
            self.assertAlmostEqual(float(grad), float(expected), places=8)

    def test_material_gradcheck_heterogeneous(self) -> None:
        mesh = _hetero_bar()
        C_matrix = _tensor(isotropic_stiffness(10.0, 0.3))
        C_fiber0 = _tensor(isotropic_stiffness(70.0, 0.2))

        def dbar_from_fiber(C_fiber: torch.Tensor) -> torch.Tensor:
            result = effective_stiffness(
                mesh=mesh,
                material_stiffness={"matrix": C_matrix, "fiber": C_fiber},
                macro_model="cauchy_3d",
            )
            return result.Dbar

        C_fiber = C_fiber0.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(dbar_from_fiber, (C_fiber,), eps=1e-6, atol=1e-5, rtol=1e-3)
        )

    def test_geometry_gradient_matches_finite_difference(self) -> None:
        mesh = _hetero_bar()
        materials = {"matrix": isotropic_stiffness(10.0, 0.3), "fiber": isotropic_stiffness(70.0, 0.2)}
        base_nodes = np.asarray(mesh.nodes, dtype=float)
        interface = np.where(np.isclose(base_nodes[:, 0], 1.0))[0]
        self.assertEqual(len(interface), 4)  # 2x2 shared face at x = 1

        def dbar00(node_array: np.ndarray) -> torch.Tensor:
            nodes = torch.tensor(node_array, dtype=torch.float64, requires_grad=False)
            result = effective_stiffness(
                mesh=mesh, material_stiffness=materials, nodes=nodes, macro_model="cauchy_3d"
            )
            return result.Dbar[0, 0]

        # Autograd directional derivative: move all interface nodes along +x.
        nodes = torch.tensor(base_nodes, dtype=torch.float64, requires_grad=True)
        result = effective_stiffness(
            mesh=mesh, material_stiffness=materials, nodes=nodes, macro_model="cauchy_3d"
        )
        (grad,) = torch.autograd.grad(result.Dbar[0, 0], nodes)
        directional = float(grad[interface, 0].sum())
        self.assertGreater(abs(directional), 1e-6)  # heterogeneous: genuinely non-zero

        # Central finite difference along the same direction.
        eps = 1e-6
        plus = base_nodes.copy()
        plus[interface, 0] += eps
        minus = base_nodes.copy()
        minus[interface, 0] -= eps
        fd = float((dbar00(plus) - dbar00(minus)) / (2 * eps))

        self.assertAlmostEqual(directional, fd, places=4)

    def test_backward_populates_material_and_geometry_grads(self) -> None:
        mesh = _hetero_bar()
        C_matrix = _tensor(isotropic_stiffness(10.0, 0.3)).requires_grad_(True)
        C_fiber = _tensor(isotropic_stiffness(70.0, 0.2)).requires_grad_(True)
        nodes = torch.tensor(np.asarray(mesh.nodes, dtype=float), dtype=torch.float64, requires_grad=True)

        result = effective_stiffness(
            mesh=mesh,
            material_stiffness={"matrix": C_matrix, "fiber": C_fiber},
            nodes=nodes,
            macro_model="cauchy_3d",
        )
        # A scalar objective: the effective axial stiffness.
        loss = result.Dbar[0, 0]
        loss.backward()

        for tensor in (C_matrix, C_fiber, nodes):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        # The axial response must depend on the (stiffer) fiber material.
        self.assertGreater(float(C_fiber.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
