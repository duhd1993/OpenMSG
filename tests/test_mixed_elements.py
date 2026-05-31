from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from openmsg.config import run_config
from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import effective_stiffness
from openmsg.materials import isotropic_stiffness
from openmsg.mesh import SolidMesh, mesh_from_config


def mixed_quad_tri_mesh() -> SolidMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    return SolidMesh(
        nodes=nodes,
        elements=[
            {"type": "quad4", "connectivity": [[0, 1, 4, 3]], "material": "m"},
            {
                "type": "tri3",
                "connectivity": [[1, 2, 5], [1, 5, 4]],
                "material": "m",
            },
        ],
    )


def mixed_quad_tri_config() -> dict[str, object]:
    return {
        "analysis": {"type": "msg_3d_cauchy"},
        "materials": {"m": {"type": "isotropic", "E": 100.0, "nu": 0.25}},
        "mesh": {
            "active_axes": ["y", "z"],
            "nodes": [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            "elements": [
                {"type": "quad4", "connectivity": [[0, 1, 4, 3]], "material": "m"},
                {
                    "type": "tri3",
                    "connectivity": [[1, 2, 5], [1, 5, 4]],
                    "material": "m",
                },
            ],
        },
    }


class MixedElementTests(unittest.TestCase):
    def test_mixed_quad_tri_homogeneous_cauchy(self) -> None:
        mesh = mixed_quad_tri_mesh()
        C = isotropic_stiffness(100.0, 0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})

        self.assertEqual(mesh.element_types, ("quad4", "tri3"))
        self.assertEqual(mesh.n_elements, 3)
        self.assertEqual(mesh.sg_dimension, 2)
        self.assertEqual(mesh.active_axes, (1, 2))
        np.testing.assert_allclose(result.Dbar, C, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.metadata["element_types"], ["quad4", "tri3"])
        self.assertEqual(result.metadata["assembly_cell_types"], ["quad", "triangle"])

    def test_mixed_quad_tri_config_input(self) -> None:
        mesh = mesh_from_config(mixed_quad_tri_config()["mesh"])  # type: ignore[index]

        self.assertEqual(mesh.element_types, ("quad4", "tri3"))
        self.assertEqual(mesh.material_ids, ("m", "m", "m"))

        result = run_config(mixed_quad_tri_config())
        expected = isotropic_stiffness(100.0, 0.25)
        np.testing.assert_allclose(result.Dbar, expected, rtol=1e-10, atol=1e-10)

    def test_mixed_quad_tri_meshio_input(self) -> None:
        if importlib.util.find_spec("meshio") is None:
            self.skipTest("meshio is not installed in this environment")

        import meshio

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 1.0],
            ]
        )
        meshio_mesh = meshio.Mesh(
            points=points,
            cells=[
                ("quad", np.array([[0, 1, 4, 3]], dtype=int)),
                ("triangle", np.array([[1, 2, 5], [1, 5, 4]], dtype=int)),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.vtu"
            meshio.write(path, meshio_mesh)
            result = run_config(
                {
                    "analysis": {"type": "msg_3d_cauchy"},
                    "materials": {"m": {"type": "isotropic", "E": 100.0, "nu": 0.25}},
                    "mesh": {
                        "type": "meshio",
                        "path": str(path),
                        "cell_types": ["quad", "triangle"],
                        "active_axes": ["y", "z"],
                        "default_material": "m",
                    },
                }
            )

        expected = isotropic_stiffness(100.0, 0.25)
        np.testing.assert_allclose(result.Dbar, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.metadata["element_types"], ["quad4", "tri3"])

    def test_mixed_quad_tri_dehomogenization(self) -> None:
        mesh = mixed_quad_tri_mesh()
        C = isotropic_stiffness(100.0, 0.25)
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        fields = recover_gauss_fields(
            mesh=mesh,
            material_stiffness={"m": C},
            V0=result.V0,
            macro_strain=macro_strain,
        )

        self.assertEqual(fields["strain"].shape, (10, 6))
        for strain in fields["strain"]:
            np.testing.assert_allclose(strain, macro_strain, atol=1e-10)

    def test_mixed_quad_tri_material_autograd(self) -> None:
        mesh = mixed_quad_tri_mesh()
        young = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
        C = isotropic_stiffness(young=young, poisson=0.25)

        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        result.Dbar[0, 0].backward()

        self.assertIsNotNone(young.grad)
        self.assertGreater(float(young.grad), 0.0)

    def test_mixed_dimensions_are_rejected(self) -> None:
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "same SG dimension"):
            SolidMesh(
                nodes=nodes,
                elements=[
                    {"type": "line2", "connectivity": [[0, 1]], "material": "m"},
                    {"type": "tri3", "connectivity": [[0, 1, 2]], "material": "m"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
