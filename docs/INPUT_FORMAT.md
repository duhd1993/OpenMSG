# OpenMSG Input Format

OpenMSG reads JSON configuration files. The format is deliberately explicit so
the same SG finite element core can target 3D Cauchy continuum,
Kirchhoff-Love plate, and Euler-Bernoulli beam macroscopic models.

## Minimal Example

```json
{
  "analysis": {
    "type": "msg_3d_cauchy",
    "constraints": [
      {"type": "periodic", "axes": ["x", "y", "z"]},
      {"type": "mean_zero"}
    ]
  },
  "materials": {
    "matrix": {"type": "isotropic", "E": 100.0, "nu": 0.25}
  },
  "mesh": {
    "nodes": [[0, 0, 0], [1, 0, 0], "..."],
    "elements": [
      {
        "type": "hex8",
        "connectivity": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "material": "matrix"
      }
    ]
  }
}
```

Run it with:

```bash
uv run python -m openmsg examples/homogeneous_3d.json
```

## Analysis Block

```json
{
  "type": "msg_3d_cauchy",
  "linear_solver": "auto",
  "constraints": [
    {"type": "periodic", "axes": ["x", "y", "z"]},
    {"type": "mean_zero"}
  ]
}
```

Supported MSG analysis types:

- `msg_3d_cauchy`: homogenize to a 3D Cauchy continuum. Output is a 6x6
  material stiffness matrix.
- `msg_kirchhoff_love_plate`: homogenize to a Kirchhoff-Love plate. Output is
  a 6x6 `ABD` matrix with generalized strain order
  `[e11, e22, 2e12, k11, k22, 2k12]`.
- `msg_euler_bernoulli_beam`: homogenize to an Euler-Bernoulli beam. Output is
  a 4x4 section stiffness matrix with generalized strain order
  `[e1, k1, k2, k3]`.

The SG mesh may be 3D, 2D, or 1D for any of these macroscopic models when the
geometry/material distribution is invariant by translation in the omitted
directions. Analytical classical laminate ABD references belong in examples or
tests, not in the JSON analysis API.

MSG finite element assembly uses TensorMesh. There is no alternate assembly
selector in the input format.

Material definitions in JSON are converted to torch stiffness tensors before
MSG assembly. The Python MSG assembly path expects each `material_stiffness`
value to be a `torch.Tensor`; NumPy stiffness arrays are not accepted by the
core solver.

Linear solvers:

- `auto`: use the TensorMesh / `torch-sla` sparse saddle-point solve.
- `dense`: force dense PyTorch block solving.
- `sparse`: force TensorMesh / `torch-sla` sparse saddle-point solving.

Constraint shorthand is also accepted:

```json
"constraints": ["periodic", "mean_zero"]
```

The `periodic` shorthand means all three axes.

If `constraints` is omitted, OpenMSG uses model-specific defaults:

- Cauchy continuum: periodic fluctuation constraints on all axes plus
  mean-zero.
- Kirchhoff-Love plate: periodic constraints on active in-plane SG axes, plus
  mean-zero and average fluctuation-rotation removal.
- Euler-Bernoulli beam: periodic constraints on an active axial SG axis, plus
  mean-zero and average twist removal.

Override `constraints` explicitly when a plate or beam SG needs a different
boundary/periodicity choice.

## MSG Macro Model Options

Plate and beam axes can be configured either directly in `analysis` or inside a
`macro_model` object:

```json
{
  "type": "msg_kirchhoff_love_plate",
  "thickness_axis": "z",
  "inplane_axes": ["x", "y"],
  "reference_point": [0, 0, 0]
}
```

```json
{
  "type": "msg_euler_bernoulli_beam",
  "axial_axis": "x",
  "cross_section_axes": ["y", "z"],
  "reference_point": [0, 0, 0]
}
```

If `reference_point` is omitted, the center of the mesh bounding box is used.
This point defines the plate reference surface or beam reference line for the
curvature modes and coupling terms.

The MSG normalization factor `omega` follows the papers:

- Cauchy: SG measure, so the output is an intensive material stiffness.
- Plate: product of active in-plane SG periods; for a 1D thickness SG,
  `omega = 1`.
- Beam: active axial SG period; for a 2D cross-section SG, `omega = 1`.

## Materials Block

Isotropic material:

```json
"steel": {"type": "isotropic", "E": 210000.0, "nu": 0.3}
```

Full stiffness matrix:

```json
"phase": {
  "type": "stiffness",
  "C": [[... 6 numbers ...], "... 6 rows total ..."]
}
```

Additional material symmetries:

```json
"cubic": {"type": "cubic", "C11": 250.0, "C12": 150.0, "C44": 90.0}
```

```json
"ud": {
  "type": "transversely_isotropic",
  "axis": "z",
  "E_l": 140.0,
  "E_t": 10.0,
  "nu_lt": 0.28,
  "nu_tt": 0.40,
  "G_lt": 5.0
}
```

```json
"ortho": {
  "type": "orthotropic",
  "E1": 30.0,
  "E2": 20.0,
  "E3": 10.0,
  "nu12": 0.25,
  "nu13": 0.20,
  "nu23": 0.18,
  "G12": 8.0,
  "G13": 6.0,
  "G23": 4.0
}
```

All stiffness matrices use Voigt order:

```text
[e11, e22, e33, 2e23, 2e13, 2e12]
```

Material stiffnesses are interpreted in their local material axes. Use element
block `orientation` to rotate a local material stiffness into the global SG
coordinate system.

## Explicit SG Mesh

The production-oriented input path is an explicit mesh with nodes and element
blocks. The top-level mesh does not carry an element type. Each block declares
its element `type`, a 2D `connectivity` matrix, and `material`. `material` may
be a single material name for the entire block or a list with one name per
element in that block. The supported first-order SG element types are:

- `hex8`: 8-node trilinear hexahedron.
- `tet4`: 4-node linear tetrahedron.
- `quad4`: 4-node bilinear quadrilateral for a 2D SG.
- `tri3`: 3-node linear triangle for a 2D SG.
- `line2`: 2-node linear line for a 1D SG.

For MSG analyses, all element types use a three-component fluctuation field
`w=[w1,w2,w3]`. Lower-dimensional elements simply restrict the fluctuation
field to vary along selected SG coordinates. The macroscopic output shape is
chosen by the analysis type, not by the SG dimension.

A single explicit mesh can combine element types of the same SG dimension.
Valid combinations include `hex8` with `tet4`, or `quad4` with `tri3`; mixing
1D, 2D, and 3D SG element dimensions in the same mesh is rejected.

```json
{
  "nodes": [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
  ],
  "elements": [
    {
      "type": "hex8",
      "connectivity": [[0, 1, 2, 3, 4, 5, 6, 7]],
      "material": "matrix"
    }
  ]
}
```

For heterogeneous blocks, replace the scalar `material` with a list whose
length equals the number of connectivity rows.

## Material Orientation

An element block may include `orientation`. If omitted, the identity orientation
is used. Orientation matrices follow the `local_to_global` convention: the
columns of the 3x3 matrix are the local material basis vectors expressed in
global SG coordinates.

Block-wide axis-angle orientation:

```json
{
  "type": "hex8",
  "connectivity": [[0, 1, 2, 3, 4, 5, 6, 7]],
  "material": "fiber",
  "orientation": {
    "type": "axis_angle",
    "axis": [0, 0, 1],
    "angle_degrees": 30.0
  }
}
```

Use exactly one of `angle_degrees` or `angle_radians`. A 3x3 matrix orientation
is also accepted:

```json
"orientation": {
  "type": "matrix",
  "local_to_global": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]
}
```

For per-element orientations, replace the single `orientation` object with a
list whose length equals the number of connectivity rows. The Python API also
accepts torch tensors inside orientation specs; gradients of `Dbar` can flow
back to axis-angle values or rotation matrices when they require gradients.

Hex8 node order is:

```text
0 (-,-,-), 1 (+,-,-), 2 (+,+,-), 3 (-,+,-),
4 (-,-,+), 5 (+,-,+), 6 (+,+,+), 7 (-,+,+)
```

Tet4 node order follows the standard positively oriented reference tetrahedron:

```text
0 (0,0,0), 1 (1,0,0), 2 (0,1,0), 3 (0,0,1)
```

Physical elements must have positive Jacobian determinant. If a tetrahedron is
read with reversed orientation, swap two nodes in that element.

A mixed 2D SG with Quad4 and Tri3 elements:

```json
{
  "active_axes": ["y", "z"],
  "nodes": [
    [0, 0],
    [1, 0],
    [2, 0],
    [0, 1],
    [1, 1],
    [2, 1]
  ],
  "elements": [
    {"type": "quad4", "connectivity": [[0, 1, 4, 3]], "material": "matrix"},
    {
      "type": "tri3",
      "connectivity": [[1, 2, 5], [1, 5, 4]],
      "material": "matrix"
    }
  ]
}
```

## Lower-Dimensional SG Meshes

Use `active_axes` to state which physical axes are represented by the SG
coordinates. Defaults follow common MSG conventions:

- `line2`: `["z"]`, i.e. a 1D SG through thickness/axis 3.
- `quad4` and `tri3`: `["y", "z"]`, i.e. a 2D SG over axes 2 and 3.
- `hex8` and `tet4`: `["x", "y", "z"]`.

A 1D homogeneous SG:

```json
{
  "active_axes": ["z"],
  "nodes": [[0], [1]],
  "elements": [
    {"type": "line2", "connectivity": [[0, 1]], "material": "matrix"}
  ]
}
```

A 2D SG in the `y-z` plane:

```json
{
  "active_axes": ["y", "z"],
  "nodes": [[0, 0], [1, 0], [1, 1], [0, 1]],
  "elements": [
    {"type": "quad4", "connectivity": [[0, 1, 2, 3]], "material": "matrix"}
  ]
}
```

When node coordinates have one or two columns, OpenMSG embeds them into the
listed `active_axes`. Three-column coordinates may also be supplied directly,
for example with constant `x` and varying `y,z`.

## External Mesh Files

External mesh files are read through `meshio`:

```json
"mesh": {
  "type": "meshio",
  "path": "cell.vtu",
  "cell_type": "hexahedron",
  "material_data": "gmsh:physical",
  "material_map": {
    "1": "matrix",
    "2": "fiber"
  }
}
```

Use `"cell_type": "tetra"` for Tet4 meshes, `"quad"` or `"triangle"` for 2D SG
meshes, and `"line"` for 1D SG meshes.

For mixed external meshes, use `cell_types`:

```json
"mesh": {
  "type": "meshio",
  "path": "mixed_2d.vtu",
  "cell_types": ["quad", "triangle"],
  "active_axes": ["y", "z"],
  "default_material": "matrix"
}
```

All selected `cell_types` must map to the same SG dimension. If multiple cell
types are selected, OpenMSG infers `quad4`, `tri3`, `hex8`, `tet4`, or `line2`
from the meshio cell type names.

Relative paths are resolved relative to the JSON file. If no material data is
available, `default_material` is used.

## Dehomogenization Request

A config can optionally store macroscopic strains for local field recovery:

```json
"dehomogenization": {
  "macro_strains": {
    "unit_z": [0, 0, 0.01, 0, 0, 0]
  }
}
```

Programmatic users can call `openmsg.dehomogenize.recover_gauss_fields` with
the torch-backed `MSGResult.V0` matrix and a macroscopic generalized strain
vector. The vector length follows the selected macroscopic model: 6 for Cauchy,
6 for Kirchhoff-Love plate, and 4 for Euler-Bernoulli beam.
