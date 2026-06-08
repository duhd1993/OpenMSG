from __future__ import annotations

import unittest

import numpy as np

from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import effective_stiffness
from openmsg.materials import (
    isotropic_stiffness,
    orthotropic_stiffness,
    rotate_stiffness,
    rotation_matrix_from_axis_angle,
)
from openmsg.mesh import SolidMesh
from tests.mesh_builders import structured_hex_mesh


class DehomogenizeTests(unittest.TestCase):
    def test_homogeneous_recovery_matches_macro_strain_and_stress(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)
        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])

        fields = recover_gauss_fields(mesh=mesh, material_stiffness={"m": C}, V0=result.V0, macro_strain=macro_strain)

        for strain in fields["strain"]:
            np.testing.assert_allclose(strain, macro_strain, atol=1e-10)
        for stress in fields["stress"]:
            np.testing.assert_allclose(stress, C.detach().cpu().numpy() @ macro_strain, atol=1e-10)

    def test_homogeneous_recovery_uses_material_orientation(self) -> None:
        base = structured_hex_mesh(
            bounds=((0, 1), (0, 1), (0, 1)),
            cells=(1, 1, 1),
            default_material="m",
        )
        mesh = SolidMesh(
            nodes=base.nodes,
            elements=[
                {
                    "type": "hex8",
                    "connectivity": base.element_blocks[0].elements,
                    "material": "m",
                    "orientation": {
                        "type": "axis_angle",
                        "axis": [0.0, 0.0, 1.0],
                        "angle_degrees": 45.0,
                    },
                }
            ],
        )
        C = orthotropic_stiffness(
            E1=70.0,
            E2=12.0,
            E3=8.0,
            nu12=0.24,
            nu13=0.20,
            nu23=0.18,
            G12=5.0,
            G13=4.0,
            G23=3.0,
        )
        R = rotation_matrix_from_axis_angle(axis=[0.0, 0.0, 1.0], angle_degrees=45.0)
        C_global = rotate_stiffness(C, R).detach().cpu().numpy()
        result = effective_stiffness(mesh=mesh, material_stiffness={"m": C})
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])

        fields = recover_gauss_fields(
            mesh=mesh,
            material_stiffness={"m": C},
            V0=result.V0,
            macro_strain=macro_strain,
        )

        for stress in fields["stress"]:
            np.testing.assert_allclose(stress, C_global @ macro_strain, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
