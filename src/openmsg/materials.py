"""Torch material stiffness helpers for MSG calculations."""

from __future__ import annotations

import torch

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def isotropic_stiffness(
    young: object,
    poisson: object,
    *,
    dtype: object | None = None,
    device: object | None = None,
) -> torch.Tensor:
    """Return a 6x6 isotropic stiffness tensor.

    The Voigt order is ``[e11, e22, e33, 2e23, 2e13, 2e12]``. Engineering
    shear strains are used, so the shear diagonal entries are the shear modulus
    ``mu``. ``young`` and ``poisson`` may be scalar tensors, in which case
    autograd history is preserved.
    """

    dtype, device = _dtype_device((young, poisson), dtype=dtype, device=device)
    young_t = _scalar_tensor(young, dtype=dtype, device=device)
    poisson_t = _scalar_tensor(poisson, dtype=dtype, device=device)
    _require_positive(young_t, "young")
    _require_between(poisson_t, "poisson", lower=-1.0, upper=0.5)

    lam = young_t * poisson_t / ((1.0 + poisson_t) * (1.0 - 2.0 * poisson_t))
    mu = young_t / (2.0 * (1.0 + poisson_t))
    zero = torch.zeros((), dtype=dtype, device=device)

    return torch.stack(
        [
            torch.stack([lam + 2.0 * mu, lam, lam, zero, zero, zero]),
            torch.stack([lam, lam + 2.0 * mu, lam, zero, zero, zero]),
            torch.stack([lam, lam, lam + 2.0 * mu, zero, zero, zero]),
            torch.stack([zero, zero, zero, mu, zero, zero]),
            torch.stack([zero, zero, zero, zero, mu, zero]),
            torch.stack([zero, zero, zero, zero, zero, mu]),
        ]
    )


def cubic_stiffness(
    c11: object,
    c12: object,
    c44: object,
    *,
    dtype: object | None = None,
    device: object | None = None,
) -> torch.Tensor:
    """Return a cubic material stiffness tensor."""

    dtype, device = _dtype_device((c11, c12, c44), dtype=dtype, device=device)
    c11_t = _scalar_tensor(c11, dtype=dtype, device=device)
    c12_t = _scalar_tensor(c12, dtype=dtype, device=device)
    c44_t = _scalar_tensor(c44, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    C = torch.stack(
        [
            torch.stack([c11_t, c12_t, c12_t, zero, zero, zero]),
            torch.stack([c12_t, c11_t, c12_t, zero, zero, zero]),
            torch.stack([c12_t, c12_t, c11_t, zero, zero, zero]),
            torch.stack([zero, zero, zero, c44_t, zero, zero]),
            torch.stack([zero, zero, zero, zero, c44_t, zero]),
            torch.stack([zero, zero, zero, zero, zero, c44_t]),
        ]
    )
    return as_stiffness_matrix(C)


def orthotropic_stiffness(
    *,
    E1: object,
    E2: object,
    E3: object,
    nu12: object,
    nu13: object,
    nu23: object,
    G12: object,
    G13: object,
    G23: object,
    dtype: object | None = None,
    device: object | None = None,
) -> torch.Tensor:
    """Return an orthotropic stiffness tensor from engineering constants."""

    values = (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)
    dtype, device = _dtype_device(values, dtype=dtype, device=device)
    E1_t = _scalar_tensor(E1, dtype=dtype, device=device)
    E2_t = _scalar_tensor(E2, dtype=dtype, device=device)
    E3_t = _scalar_tensor(E3, dtype=dtype, device=device)
    nu12_t = _scalar_tensor(nu12, dtype=dtype, device=device)
    nu13_t = _scalar_tensor(nu13, dtype=dtype, device=device)
    nu23_t = _scalar_tensor(nu23, dtype=dtype, device=device)
    G12_t = _scalar_tensor(G12, dtype=dtype, device=device)
    G13_t = _scalar_tensor(G13, dtype=dtype, device=device)
    G23_t = _scalar_tensor(G23, dtype=dtype, device=device)

    for name, value in {
        "E1": E1_t,
        "E2": E2_t,
        "E3": E3_t,
        "G12": G12_t,
        "G13": G13_t,
        "G23": G23_t,
    }.items():
        _require_positive(value, name)

    zero = torch.zeros((), dtype=dtype, device=device)
    S = torch.stack(
        [
            torch.stack([1.0 / E1_t, -nu12_t / E1_t, -nu13_t / E1_t, zero, zero, zero]),
            torch.stack([-nu12_t / E1_t, 1.0 / E2_t, -nu23_t / E2_t, zero, zero, zero]),
            torch.stack([-nu13_t / E1_t, -nu23_t / E2_t, 1.0 / E3_t, zero, zero, zero]),
            torch.stack([zero, zero, zero, 1.0 / G23_t, zero, zero]),
            torch.stack([zero, zero, zero, zero, 1.0 / G13_t, zero]),
            torch.stack([zero, zero, zero, zero, zero, 1.0 / G12_t]),
        ]
    )
    return as_stiffness_matrix(torch.linalg.inv(S))


def transversely_isotropic_stiffness(
    *,
    E_l: object,
    E_t: object,
    nu_lt: object,
    nu_tt: object,
    G_lt: object,
    axis: str = "z",
    dtype: object | None = None,
    device: object | None = None,
) -> torch.Tensor:
    """Return a transversely isotropic stiffness tensor.

    ``axis`` names the global longitudinal material axis. The transverse shear
    modulus is inferred as ``E_t / (2 * (1 + nu_tt))``.
    """

    dtype, device = _dtype_device((E_l, E_t, nu_lt, nu_tt, G_lt), dtype=dtype, device=device)
    E_t_t = _scalar_tensor(E_t, dtype=dtype, device=device)
    nu_tt_t = _scalar_tensor(nu_tt, dtype=dtype, device=device)
    G_tt = E_t_t / (2.0 * (1.0 + nu_tt_t))
    C_local_x = orthotropic_stiffness(
        E1=E_l,
        E2=E_t_t,
        E3=E_t_t,
        nu12=nu_lt,
        nu13=nu_lt,
        nu23=nu_tt_t,
        G12=G_lt,
        G13=G_lt,
        G23=G_tt,
        dtype=dtype,
        device=device,
    )
    local_to_global = _longitudinal_axis_permutation(axis)
    return rotate_stiffness_by_axis_permutation(C_local_x, local_to_global)


def as_stiffness_matrix(
    value: object,
    *,
    dtype: object | None = None,
    device: object | None = None,
) -> torch.Tensor:
    """Validate and return a symmetric 6x6 stiffness tensor."""

    if isinstance(value, torch.Tensor):
        C = value.to(dtype=dtype or value.dtype, device=device or value.device)
    else:
        C = torch.as_tensor(value, dtype=dtype or torch.float64, device=device)
    if C.shape != (6, 6):
        raise ValueError(f"stiffness matrix must have shape (6, 6), got {tuple(C.shape)}")
    if not torch.allclose(C, C.transpose(0, 1), rtol=1e-10, atol=1e-10):
        raise ValueError("stiffness matrix must be symmetric")
    return C


def stiffness_from_config(config: dict[str, object]) -> torch.Tensor:
    """Build a material stiffness tensor from a JSON material block."""

    kind = str(config.get("type", config.get("symmetry", "stiffness"))).lower()
    if kind == "isotropic":
        young = _required_float(config, "E", "young", "young_modulus")
        poisson = _required_float(config, "nu", "poisson", "poisson_ratio")
        return isotropic_stiffness(young=young, poisson=poisson)
    if kind == "cubic":
        return cubic_stiffness(
            c11=_required_float(config, "C11", "c11"),
            c12=_required_float(config, "C12", "c12"),
            c44=_required_float(config, "C44", "c44"),
        )
    if kind in {"orthotropic", "engineering_constants"}:
        return orthotropic_stiffness(
            E1=_required_float(config, "E1"),
            E2=_required_float(config, "E2"),
            E3=_required_float(config, "E3"),
            nu12=_required_float(config, "nu12"),
            nu13=_required_float(config, "nu13"),
            nu23=_required_float(config, "nu23"),
            G12=_required_float(config, "G12"),
            G13=_required_float(config, "G13"),
            G23=_required_float(config, "G23"),
        )
    if kind in {"transversely_isotropic", "transverse_isotropic"}:
        return transversely_isotropic_stiffness(
            E_l=_required_float(config, "E_l", "E_longitudinal", "EL", "E1"),
            E_t=_required_float(config, "E_t", "E_transverse", "ET", "E2"),
            nu_lt=_required_float(config, "nu_lt", "nu12"),
            nu_tt=_required_float(config, "nu_tt", "nu23"),
            G_lt=_required_float(config, "G_lt", "G12"),
            axis=str(config.get("axis", "z")),
        )
    if kind in {"stiffness", "anisotropic"}:
        if "C" not in config:
            raise ValueError("stiffness material requires a C matrix")
        return as_stiffness_matrix(config["C"])
    raise ValueError(f"unsupported material type {kind!r}")


def assert_positive_definite(C: object, *, name: str = "C") -> None:
    """Raise if a stiffness tensor is not symmetric positive definite."""

    matrix = as_stiffness_matrix(C)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if float(eigenvalues.min().detach().cpu()) <= 0.0:
        raise ValueError(f"{name} must be positive definite")


def rotate_stiffness_by_axis_permutation(C: object, local_to_global: tuple[int, int, int]) -> torch.Tensor:
    """Rotate a stiffness tensor by a pure axis permutation."""

    matrix = as_stiffness_matrix(C)
    if sorted(local_to_global) != [0, 1, 2]:
        raise ValueError("local_to_global must be a permutation of (0, 1, 2)")
    P = _voigt_permutation(local_to_global, dtype=matrix.dtype, device=matrix.device)
    return as_stiffness_matrix(P.transpose(0, 1) @ matrix @ P)


def engineering_constants_from_stiffness(C: object) -> dict[str, float]:
    """Return apparent engineering constants from a 6x6 stiffness tensor."""

    S = torch.linalg.inv(as_stiffness_matrix(C))
    values = {
        "E_x": 1.0 / S[0, 0],
        "E_y": 1.0 / S[1, 1],
        "E_z": 1.0 / S[2, 2],
        "G_yz": 1.0 / S[3, 3],
        "G_xz": 1.0 / S[4, 4],
        "G_xy": 1.0 / S[5, 5],
        "nu_xy": -S[1, 0] / S[0, 0],
        "nu_xz": -S[2, 0] / S[0, 0],
        "nu_yz": -S[2, 1] / S[1, 1],
    }
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _dtype_device(
    values: tuple[object, ...],
    *,
    dtype: object | None,
    device: object | None,
) -> tuple[object, object | None]:
    for value in values:
        if isinstance(value, torch.Tensor):
            return dtype or value.dtype, device or value.device
    return dtype or torch.float64, device


def _scalar_tensor(value: object, *, dtype: object, device: object | None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=dtype, device=device)
    else:
        tensor = torch.as_tensor(float(value), dtype=dtype, device=device)
    if tensor.ndim != 0:
        raise ValueError("material parameters must be scalar values")
    return tensor


def _require_positive(value: torch.Tensor, name: str) -> None:
    if float(value.detach().cpu()) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_between(value: torch.Tensor, name: str, *, lower: float, upper: float) -> None:
    scalar = float(value.detach().cpu())
    if not (lower < scalar < upper):
        raise ValueError(f"{name} must satisfy {lower:g} < {name} < {upper:g}")


def _required_float(config: dict[str, object], *names: str) -> float:
    for name in names:
        if name in config:
            return float(config[name])
    joined = ", ".join(names)
    raise ValueError(f"missing required material parameter: one of {joined}")


def _longitudinal_axis_permutation(axis: str) -> tuple[int, int, int]:
    key = axis.lower()
    if key == "x":
        return (0, 1, 2)
    if key == "y":
        return (1, 2, 0)
    if key == "z":
        return (2, 0, 1)
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def _voigt_permutation(
    local_to_global: tuple[int, int, int],
    *,
    dtype: object,
    device: object,
) -> torch.Tensor:
    pair_to_index = {tuple(sorted(pair)): idx for idx, pair in enumerate(VOIGT_PAIRS)}
    P = torch.zeros((6, 6), dtype=dtype, device=device)
    for local_index, local_pair in enumerate(VOIGT_PAIRS):
        global_pair = (local_to_global[local_pair[0]], local_to_global[local_pair[1]])
        global_index = pair_to_index[tuple(sorted(global_pair))]
        P[local_index, global_index] = 1.0
    return P
