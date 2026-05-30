# OpenMSG Development Notes

This repository implements a focused Mechanics of Structure Genome (MSG)
homogenization tool.

The current implementation scope is intentionally small:

- Linear elastic MSG homogenization to 3D Cauchy continuum,
  Kirchhoff-Love plate, and Euler-Bernoulli beam macroscopic models.
- 1D, 2D, and 3D Structure Genome meshes, where lower-dimensional SGs represent
  translational/symmetry reductions of a 3D continuum problem.
- Hex8, Tet4, Quad4, Tri3, and Line2 first-order SG finite elements.
- Homogenized stiffness `Dbar`: 6x6 for Cauchy, 6x6 `ABD` for
  Kirchhoff-Love plate, and 4x4 for Euler-Bernoulli beam.
- Local Gauss-point strain and stress recovery.
- Periodic and mean-zero constraints on the fluctuation field `w`; periodic
  constraints are not imposed on the full displacement field.
- TensorMesh-backed assembly and solve for supported 1D/2D/3D SG elements. The
  pipeline is fully tensorized in PyTorch and differentiable: `effective_stiffness`
  returns `Dbar` as a `torch.Tensor` with autograd history back to the material
  stiffness tensors and the node-coordinate tensor (pass either with
  `requires_grad=True`). The constrained saddle-point system is solved with the
  differentiable `torch-sla` sparse solver (`SparseMatrix.solve(method="lu")`).
  `homogenize_msg` wraps `effective_stiffness` and returns a detached NumPy
  `MSGResult` for reporting. The former hand-written NumPy assembly/solve path
  has been removed.
- Explicit input meshes are preferred. Mesh generation belongs in examples,
  tests, or external preprocessing scripts, not in the solver core.
- `analysis.type: "laminate_abd"` is the implemented analytical classical
  laminate ABD utility. Maintain this path, but do not confuse it with
  `analysis.type: "msg_kirchhoff_love_plate"`, which performs an SG finite
  element solve and outputs plate `ABD`.

Do not add shell, nonlinear, damage, large-deformation, Reissner-Mindlin plate,
Timoshenko beam, or other new macroscopic models unless explicitly requested.

Before changing numerical code, read:

1. `docs/THEORY_MSG.md`
2. `docs/IMPLEMENTATION_CONTRACT.md`
3. `docs/INPUT_FORMAT.md`

The implementation uses engineering shear strains with Voigt order:

`[e11, e22, e33, 2e23, 2e13, 2e12]`

Keep the MSG sign convention fixed:

- `H = D_hε`
- `E V0 = -H`
- constrained solve:
  `[E G.T; G 0] [V0; Lambda] = [-H; 0]`
- `Dbar = (D0 + V0.T @ H) / omega`

Every numerical change should include or update tests. Run:

```bash
uv run python -m unittest discover -s tests
```

Use `uv run` for running tests and Python in this project (the `openmsg`
package is installed editable in the uv environment, so no `PYTHONPATH` is
needed). Run scripts the same way, e.g. `uv run python examples/square_pack_fiber.py`.
