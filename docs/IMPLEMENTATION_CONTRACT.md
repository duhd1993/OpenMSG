# Implementation Contract

This file fixes names, tensor shapes, and signs. Numerical code should not
change these conventions.

## Voigt Convention

Use engineering shear strains:

```text
[e11, e22, e33, 2e23, 2e13, 2e12]
```

A material stiffness `C` maps this strain vector to:

```text
[s11, s22, s33, s23, s13, s12]
```

For isotropic materials, the shear diagonal entries are `mu`, not `2 * mu`.

## Matrix Names And Shapes

Let:

- `n_dof = 3 * n_nodes`
- `n_macro` is the number of generalized strains in the selected macroscopic
  model:
  - Cauchy continuum: 6
  - Kirchhoff-Love plate: 6
  - Euler-Bernoulli beam: 4

Then:

```text
E.shape     == (n_dof, n_dof)
H.shape     == (n_dof, n_macro)
D0.shape    == (n_macro, n_macro)
V0.shape    == (n_dof, n_macro)
Dbar.shape  == (n_macro, n_macro)
```

If constraints are used:

```text
G.shape == (n_constraints, n_dof)
```

The saddle-point matrix has shape:

```text
(n_dof + n_constraints, n_dof + n_constraints)
```

The reduced SG dimension changes which columns of the 3D shape-function
gradient are nonzero, but it does not decide `n_macro`; the macroscopic model
does.

## Sign Convention

Always solve:

```text
E V0 = -H
```

or:

```text
[E G.T] [V0    ] = [-H]
[G  0 ] [Lambda]   [ 0]
```

Then compute:

```text
Dbar = (D0 + V0.T @ H) / omega
```

Do not use `Dbar = D0 - V0.T @ H`.

## Homogeneous Material Test

For a homogeneous periodic SG with correct constraints:

```text
Dbar == C
```

within numerical tolerance. This is the most important physical sanity test.

For structural macroscopic models, compare against known section or laminate
stiffnesses:

- 1D thickness SG Kirchhoff-Love plate should converge to the classical
  laminate `ABD` matrix as the thickness mesh is refined.
- 2D cross-section SG Euler-Bernoulli beam should recover homogeneous axial and
  bending stiffness terms for a centered isotropic section.
