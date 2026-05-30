from __future__ import annotations

import unittest

import numpy as np

from openmsg.dehomogenize import recover_gauss_fields
from openmsg.homogenize import homogenize_3d_cauchy
from openmsg.materials import isotropic_stiffness
from tests.mesh_builders import structured_hex_mesh


class DehomogenizeTests(unittest.TestCase):
    def test_homogeneous_recovery_matches_macro_strain_and_stress(self) -> None:
        mesh = structured_hex_mesh(bounds=((0, 1), (0, 1), (0, 1)), cells=(1, 1, 1), default_material="m")
        C = isotropic_stiffness(100.0, 0.25)
        result = homogenize_3d_cauchy(mesh=mesh, material_stiffness={"m": C})
        macro_strain = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])

        fields = recover_gauss_fields(mesh=mesh, material_stiffness={"m": C}, V0=result.V0, macro_strain=macro_strain)

        for strain in fields["strain"]:
            np.testing.assert_allclose(strain, macro_strain, atol=1e-10)
        for stress in fields["stress"]:
            np.testing.assert_allclose(stress, C @ macro_strain, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
