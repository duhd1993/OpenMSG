# OpenMSG

OpenMSG is a focused Mechanics of Structure Genome homogenization tool. The
current version implements a testable core:

- linear elastic MSG homogenization to 3D Cauchy continuum,
  Kirchhoff-Love plate, and Euler-Bernoulli beam macroscopic models;
- Hex8, Tet4, Quad4, Tri3, and Line2 Structure Genome finite elements;
- 1D/2D/3D SG meshes with the selected macroscopic model controlling the
  output stiffness shape;
- mean-zero and periodic fluctuation constraints;
- TensorMesh assembly for supported 1D/2D/3D SG meshes;
- Gauss-point local strain/stress recovery.

The implementation follows the sign and shape conventions in
`docs/IMPLEMENTATION_CONTRACT.md`.

## Quick Start

```bash
python -m pip install -e .
PYTHONPATH=src python -m openmsg examples/homogeneous_3d.json
```

This prints the homogenized stiffness matrix for a homogeneous periodic unit
cube. It should match the input isotropic stiffness.

Write results to a file:

```bash
PYTHONPATH=src python -m openmsg examples/homogeneous_3d.json --output result.json
```

Install dependencies and run tests:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover
```

OpenMSG uses TensorMesh for MSG finite element assembly. `linear_solver:
"auto"` uses a sparse scipy saddle-point solve for the MSG constraint system.

## Python API

```python
from openmsg import SolidMesh, homogenize_3d_cauchy, isotropic_stiffness

mesh = SolidMesh(
    nodes=[
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    elements=[[0, 1, 2, 3, 4, 5, 6, 7]],
    material_ids=("matrix",),
    element_type="hex8",
)
C = isotropic_stiffness(young=100.0, poisson=0.25)

result = homogenize_3d_cauchy(
    mesh=mesh,
    material_stiffness={"matrix": C},
    constraints=[
        {"type": "periodic", "axes": ["x", "y", "z"]},
        {"type": "mean_zero"},
    ],
)

print(result.Dbar)
```

## Input Format

See `docs/INPUT_FORMAT.md` and `examples/`.

The main input path is an explicit SG mesh with nodes, element connectivity,
and material names. Mesh generation is intentionally outside the solver core;
example scripts may generate JSON inputs for testing. Reduced 1D/2D SG meshes
use `active_axes` to state which physical axes remain nonuniform.

Set `analysis.type` to choose the macroscopic model:

- `msg_3d_cauchy` returns a 6x6 Cauchy material stiffness.
- `msg_kirchhoff_love_plate` returns `ABD` for generalized strains
  `[e11, e22, 2e12, k11, k22, 2k12]`.
- `msg_euler_bernoulli_beam` returns a 4x4 beam section stiffness for
  `[e1, k1, k2, k3]`.

The separate `laminate_abd` analysis is still available as an analytical
classical laminate utility without an SG finite element solve.

## Square-Pack Benchmark

```bash
PYTHONPATH=src python examples/square_pack_fiber.py --cells-xy 8 --vf 0.35
```

The script is a benchmark/input generator, not part of the public solver API.
It writes no mesh by default; add `--write-input square_pack.json` to save the
generated explicit Hex8 input. It reports apparent engineering constants and
compares the longitudinal modulus with the Voigt rule of mixture.

## References

This implementation follows the Mechanics of Structure Genome (MSG) theory
developed by Wenbin Yu and collaborators:

- W. Yu, "A unified theory for constitutive modeling of composites,"
  *Journal of Mechanics of Materials and Structures*, vol. 11, no. 4,
  pp. 379–411, 2016. <https://cdmhub.org/resources/1102>
- W. Yu, "Simplified Formulation of Mechanics of Structure Genome,"
  *AIAA Journal*, 2019. <https://doi.org/10.2514/1.J057500>

These papers are copyrighted and are not redistributed here; obtain them from
the publishers via the links above. The local `reference/` directory holds the
project discussion notes (`reference/ChatGPT-MSG理论介绍.md`) used to define this
implementation scope.
