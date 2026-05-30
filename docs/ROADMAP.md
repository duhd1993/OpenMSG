# Roadmap

## Implemented

- Linear elastic MSG for 3D Cauchy continuum, Kirchhoff-Love plate, and
  Euler-Bernoulli beam macroscopic models.
- Hex8, Tet4, Quad4, Tri3, and Line2 finite element assembly for `E`, `H`, and
  `D0`.
- 1D/2D/3D SG meshes using `active_axes` for reduced-dimensional SG symmetry.
- Periodic, mean-zero, and average fluctuation-rotation constraints.
- JSON input format.
- Explicit solid mesh input as the primary mesh path.
- Material definitions for isotropic, cubic, transversely isotropic,
  orthotropic, and full stiffness matrices.
- TensorMesh-backed assembly for supported 1D/2D/3D SG meshes.
- A single TensorMesh-backed quadrature primitive (`assembly.tensormesh_quadrature`)
  shared by assembly, rotation-constraint construction, and local-field recovery;
  the hand-written NumPy quadrature (`voigt.py`, `elements/`) has been retired.
- Sparse saddle-point solving for TensorMesh-backed constrained SG problems.
- Fully differentiable assembly and solve (PyTorch autograd via TensorMesh /
  torch-sla); `effective_stiffness` returns `Dbar` with gradients with respect to
  material stiffness and node geometry.
- SG-based Kirchhoff-Love plate `ABD` output and analytical laminate ABD output
  with ply angle transforms.
- SG-based Euler-Bernoulli beam 4x4 stiffness output.
- Local Gauss-point strain and stress recovery.
- Square-pack fiber benchmark script with rules-of-mixture comparison.
- Unit tests for materials, algebra, constraints, assembly, homogeneous recovery,
  TensorMesh assembly validation, dehomogenization, config loading, end-to-end
  autograd (material and geometry gradients), and the square-pack input generator.

## Next Useful Steps

1. Add full 3D material orientation transforms beyond axis permutations and
   laminate in-plane rotations.
2. Add richer meshio material-region handling and mesh validation diagnostics.
3. Add reference benchmarks from SwiftComp/VABS/VAPAS/VAMUCH examples.
4. Add higher-order SG elements where TensorMesh already provides basis support,
   especially for curvature-dominated plate/beam convergence.
5. Add optional Reissner-Mindlin plate, Timoshenko beam, and shell
   macroscopic models when requested.
6. Support multiple element types within a single mesh — per-type connectivity,
   material indexing, and DOF scatter in `assemble_msg_system` (a single element
   type per mesh is currently assumed).
7. Expand autograd test coverage: Kirchhoff-Love plate and Euler-Bernoulli beam
   gradients (including the geometry path through the `omega` extent factor),
   2D/1D SG autograd, `gradcheck` with respect to node coordinates, and
   dense-vs-sparse gradient equivalence.
8. Drop the NumPy-facing layer: collapse `MSGResult`/`MSGTorchResult` into a
   single PyTorch-backed result type and have `homogenize_msg` return it
   directly; remove `AssemblyResult` and the `assemble_msg` NumPy wrappers.
   Convert to NumPy only at the serialization boundary (`to_dict`/CLI) and inside
   `recover_gauss_fields`, keeping a NumPy boundary only where genuinely needed.
