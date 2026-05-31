# Roadmap

## Implemented

- Linear elastic MSG for 3D Cauchy continuum, Kirchhoff-Love plate, and
  Euler-Bernoulli beam macroscopic models.
- Hex8, Tet4, Quad4, Tri3, and Line2 finite element assembly for `E`, `H`, and
  `D0`.
- Mixed same-dimensional element meshes within one SG mesh, including per-type
  connectivity, material indexing, TensorMesh quadrature extraction, global DOF
  scatter, local-field recovery, and material autograd.
- 1D/2D/3D SG meshes using `active_axes` for reduced-dimensional SG symmetry.
- Periodic, mean-zero, and average fluctuation-rotation constraints.
- JSON input format.
- Explicit solid mesh input as the primary mesh path.
- Element-block mesh topology: each block carries its element type,
  connectivity matrix, and block-wide or per-element material assignment; there
  is no top-level single mesh element type.
- Torch material definitions for isotropic, cubic, transversely isotropic,
  orthotropic, and full stiffness matrices.
- TensorMesh-backed assembly for supported 1D/2D/3D SG meshes.
- A single TensorMesh-backed quadrature primitive (`assembly.tensormesh_quadrature`)
  shared by assembly, rotation-constraint construction, and local-field recovery;
  the hand-written NumPy quadrature (`voigt.py`, `elements/`) has been retired.
- Sparse saddle-point solving for TensorMesh-backed constrained SG problems.
- Fully differentiable assembly and solve (PyTorch autograd via TensorMesh /
  torch-sla); `effective_stiffness` returns `Dbar` with gradients with respect to
  material stiffness and material parameters used by torch material builders.
  Assembly accepts torch material stiffness tensors only. SG node coordinates
  are treated as fixed mesh geometry, but geometry-derived quadrature data,
  constraints, and normalization factors stay in torch tensors in the core
  solve.
- PyTorch-backed `MSGResult` returned directly by `effective_stiffness`; JSON
  conversion happens at `to_dict`/CLI boundaries. The old `homogenize_msg`,
  `MSGTorchResult`, `AssemblyResult`, and NumPy assembly wrappers have been
  removed.
- SG-based Kirchhoff-Love plate `ABD` output.
- SG-based Euler-Bernoulli beam 4x4 stiffness output.
- Local Gauss-point strain and stress recovery.
- gmsh-based 2D SG square-pack fiber benchmark script with rules-of-mixture
  comparison.
- Example-level 1D plate MSG comparison against a classical laminate ABD
  reference.
- Unit tests for materials, algebra, constraints, assembly, homogeneous recovery,
  TensorMesh assembly validation, dehomogenization, config loading, end-to-end
  material autograd, structural-model material gradients, 1D/2D SG material
  gradients, dense-vs-sparse material-gradient equivalence, mixed element
  meshes, and the square-pack input generator.

## Next Useful Steps

1. Add full 3D material orientation transforms beyond axis permutations.
2. Add richer meshio material-region handling and mesh validation diagnostics.
3. Add reference benchmarks from SwiftComp/VABS/VAPAS/VAMUCH examples.
4. Add higher-order SG elements where TensorMesh already provides basis support,
   especially for curvature-dominated plate/beam convergence.
5. Add optional Reissner-Mindlin plate, Timoshenko beam, and shell
   macroscopic models when requested.
