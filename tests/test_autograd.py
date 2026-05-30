"""End-to-end automatic differentiation tests for the MSG pipeline.

These exercise the TensorMesh-backed assembly and the differentiable
``torch-sla`` solver, verifying that gradients of the homogenized stiffness
``Dbar`` flow back to material stiffness tensors.
"""

from __future__ import annotations

import unittest

import torch

from openmsg.homogenize import effective_stiffness, homogenize_msg
from openmsg.materials import isotropic_stiffness
from tests.mesh_builders import structured_hex_mesh


def _hetero_bar():
    """Two-element bar (matrix | fiber) along x."""

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

        direct_result = effective_stiffness(mesh=mesh, material_stiffness=materials, macro_model="cauchy_3d")
        wrapped_result = homogenize_msg(mesh=mesh, material_stiffness=materials)

        self.assertIsInstance(wrapped_result.Dbar, torch.Tensor)
        torch.testing.assert_close(direct_result.Dbar, wrapped_result.Dbar, rtol=1e-10, atol=1e-10)

    def test_material_gradient_scale_is_analytic_for_homogeneous_cube(self) -> None:
        # For a homogeneous cell Dbar == C, so d(Dbar)/d(scale) == C exactly.
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C0 = isotropic_stiffness(120.0, 0.3)
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
        C_matrix = isotropic_stiffness(10.0, 0.3)
        C_fiber0 = isotropic_stiffness(70.0, 0.2)

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

    def test_backward_populates_material_grads(self) -> None:
        mesh = _hetero_bar()
        C_matrix = isotropic_stiffness(10.0, 0.3).requires_grad_(True)
        C_fiber = isotropic_stiffness(70.0, 0.2).requires_grad_(True)

        result = effective_stiffness(
            mesh=mesh,
            material_stiffness={"matrix": C_matrix, "fiber": C_fiber},
            macro_model="cauchy_3d",
        )
        # A scalar objective: the effective axial stiffness.
        loss = result.Dbar[0, 0]
        loss.backward()

        for tensor in (C_matrix, C_fiber):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        # The axial response must depend on the (stiffer) fiber material.
        self.assertGreater(float(C_fiber.grad.abs().sum()), 0.0)

    def test_material_parameter_gradient_flows_through_stiffness_builder(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        young = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        poisson = torch.tensor(0.25, dtype=torch.float64)
        C = isotropic_stiffness(young, poisson)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C}, macro_model="cauchy_3d")
        result.Dbar[0, 0].backward()

        self.assertIsNotNone(young.grad)
        self.assertGreater(float(young.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
