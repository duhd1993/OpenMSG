"""Material stiffness helpers for MSG calculations."""

from __future__ import annotations

import numpy as np

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def isotropic_stiffness(young: float, poisson: float) -> np.ndarray:
    """Return a 6x6 isotropic stiffness matrix.

    The Voigt order is ``[e11, e22, e33, 2e23, 2e13, 2e12]``. Engineering
    shear strains are used, so the shear diagonal entries are the shear modulus
    ``mu``.
    """

    if young <= 0.0:
        raise ValueError("young must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must satisfy -1 < nu < 0.5")

    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))

    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = lam
    C[0, 0] += 2.0 * mu
    C[1, 1] += 2.0 * mu
    C[2, 2] += 2.0 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def cubic_stiffness(c11: float, c12: float, c44: float) -> np.ndarray:
    """Return a cubic material stiffness matrix."""

    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = c12
    C[0, 0] = c11
    C[1, 1] = c11
    C[2, 2] = c11
    C[3, 3] = c44
    C[4, 4] = c44
    C[5, 5] = c44
    return as_stiffness_matrix(C)


def orthotropic_stiffness(
    *,
    E1: float,
    E2: float,
    E3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    G12: float,
    G13: float,
    G23: float,
) -> np.ndarray:
    """Return an orthotropic stiffness matrix from engineering constants."""

    values = {
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "G12": G12,
        "G13": G13,
        "G23": G23,
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    S = np.zeros((6, 6), dtype=float)
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    return as_stiffness_matrix(np.linalg.inv(S))


def transversely_isotropic_stiffness(
    *,
    E_l: float,
    E_t: float,
    nu_lt: float,
    nu_tt: float,
    G_lt: float,
    axis: str = "z",
) -> np.ndarray:
    """Return a transversely isotropic stiffness matrix.

    ``axis`` names the global longitudinal material axis. The transverse shear
    modulus is inferred as ``E_t / (2 * (1 + nu_tt))``.
    """

    G_tt = E_t / (2.0 * (1.0 + nu_tt))
    C_local_x = orthotropic_stiffness(
        E1=E_l,
        E2=E_t,
        E3=E_t,
        nu12=nu_lt,
        nu13=nu_lt,
        nu23=nu_tt,
        G12=G_lt,
        G13=G_lt,
        G23=G_tt,
    )
    local_to_global = _longitudinal_axis_permutation(axis)
    return rotate_stiffness_by_axis_permutation(C_local_x, local_to_global)


def as_stiffness_matrix(value: object) -> np.ndarray:
    """Validate and return a symmetric 6x6 stiffness matrix."""

    C = np.asarray(value, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"stiffness matrix must have shape (6, 6), got {C.shape}")
    if not np.allclose(C, C.T, rtol=1e-10, atol=1e-10):
        raise ValueError("stiffness matrix must be symmetric")
    return C


def stiffness_from_config(config: dict[str, object]) -> np.ndarray:
    """Build a material stiffness matrix from a JSON material block."""

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


def assert_positive_definite(C: np.ndarray, *, name: str = "C") -> None:
    """Raise if a stiffness matrix is not symmetric positive definite."""

    C = as_stiffness_matrix(C)
    eigenvalues = np.linalg.eigvalsh(C)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(f"{name} must be positive definite")


def rotate_stiffness_by_axis_permutation(C: np.ndarray, local_to_global: tuple[int, int, int]) -> np.ndarray:
    """Rotate a stiffness matrix by a pure axis permutation."""

    C = as_stiffness_matrix(C)
    if sorted(local_to_global) != [0, 1, 2]:
        raise ValueError("local_to_global must be a permutation of (0, 1, 2)")
    P = _voigt_permutation(local_to_global)
    return as_stiffness_matrix(P.T @ C @ P)


def engineering_constants_from_stiffness(C: np.ndarray) -> dict[str, float]:
    """Return apparent engineering constants from a 6x6 stiffness matrix."""

    S = np.linalg.inv(as_stiffness_matrix(C))
    return {
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


def _voigt_permutation(local_to_global: tuple[int, int, int]) -> np.ndarray:
    pair_to_index = {tuple(sorted(pair)): idx for idx, pair in enumerate(VOIGT_PAIRS)}
    P = np.zeros((6, 6), dtype=float)
    for local_index, local_pair in enumerate(VOIGT_PAIRS):
        global_pair = (local_to_global[local_pair[0]], local_to_global[local_pair[1]])
        global_index = pair_to_index[tuple(sorted(global_pair))]
        P[local_index, global_index] = 1.0
    return P
