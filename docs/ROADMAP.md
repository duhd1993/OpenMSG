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
- Sparse saddle-point solving for TensorMesh-backed constrained SG problems.
- SG-based Kirchhoff-Love plate `ABD` output and analytical laminate ABD output
  with ply angle transforms.
- SG-based Euler-Bernoulli beam 4x4 stiffness output.
- Local Gauss-point strain and stress recovery.
- Square-pack fiber benchmark script with rules-of-mixture comparison.
- Unit tests for materials, algebra, constraints, assembly, homogeneous recovery,
  TensorMesh assembly validation, dehomogenization, config loading, and the square-pack input
  generator.

## Next Useful Steps

1. Add full 3D material orientation transforms beyond axis permutations and
   laminate in-plane rotations.
2. Add richer meshio material-region handling and mesh validation diagnostics.
3. Add reference benchmarks from SwiftComp/VABS/VAPAS/VAMUCH examples.
4. Add higher-order SG elements where TensorMesh already provides basis support,
   especially for curvature-dominated plate/beam convergence.
5. Add optional Reissner-Mindlin plate, Timoshenko beam, and shell
   macroscopic models when requested.
