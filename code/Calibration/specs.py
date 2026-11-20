"""
Model specifications for the generic calibration engine in Calibration/calibration.py.

Each spec lists the model parameters (V0 is appended automatically as the last
calibration parameter), the initial guess, bounds, scaling, sampling ranges for
random starting points and model-specific sanity checks.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from Models.Heston import PARAMETER_NAMES as HESTON_PARAMS
from Models.Heston import HestonModel
from Models.SVJ import PARAMETER_NAMES as SVJ_PARAMS
from Models.SVJ import SVJModel
from Models.SVJJ import PARAMETER_NAMES as SVJJ_PARAMS
from Models.SVJJ import SVJJModel


@dataclass
class ModelSpec:
    name: str
    model_cls: type
    model_params: list  # constructor parameter names (without V0)
    initial_guess: np.ndarray  # order: model_params + [V0]
    lower: np.ndarray
    upper: np.ndarray
    x_scale: np.ndarray
    log_sample_ranges: dict  # name -> (lo, hi), sampled log-uniformly
    uniform_sample_ranges: dict  # name -> (lo, hi), sampled uniformly
    output_file: str
    structural_checks: Optional[Callable[[dict], None]] = None
    param_names: list = field(init=False)

    def __post_init__(self):
        self.param_names = list(self.model_params) + ["V0"]
        self.idx = {n: i for i, n in enumerate(self.param_names)}
        n = len(self.param_names)
        for a in ("initial_guess", "lower", "upper", "x_scale"):
            setattr(self, a, np.asarray(getattr(self, a), dtype=float))
            assert len(getattr(self, a)) == n, f"{self.name}: {a} has wrong length"
        sampled = set(self.log_sample_ranges) | set(self.uniform_sample_ranges)
        assert sampled == set(
            self.param_names
        ), f"{self.name}: sample ranges incomplete"

    def make_model(self, params, r, q):
        return self.model_cls(r=r, q=q, **dict(zip(self.model_params, params[:-1])))


# ---------------------------------------------------------
# Heston
# ---------------------------------------------------------


def heston_checks(p):
    feller = 2 * p["kappa"] * p["theta"] / p["sigma"] ** 2
    status = (
        "satisfied" if feller >= 1 else "violated (still valid for Fourier pricing)"
    )
    print(f"\nFeller ratio 2*kappa*theta/sigma^2 = {feller:.4f} -> {status}")
    print(f"Long-run variance: {p['theta']:.6f} (vol {np.sqrt(p['theta']):.2%})")
    print(f"Initial variance:  {p['V0']:.6f} (vol {np.sqrt(p['V0']):.2%})")


HESTON = ModelSpec(
    name="Heston",
    model_cls=HestonModel,
    model_params=HESTON_PARAMS,  # kappa, theta, sigma, rho
    initial_guess=[2.0, 0.04, 0.50, -0.50, 0.04],
    lower=[0.01, 1e-3, 1e-2, -0.999, 1e-4],
    upper=[20.0, 1.0, 3.0, 0.999, 1.0],
    x_scale=[2.0, 0.04, 0.5, 0.5, 0.04],
    log_sample_ranges={
        "kappa": (0.1, 10.0),
        "theta": (0.005, 0.20),
        "sigma": (0.05, 1.5),
        "V0": (0.005, 0.15),
    },
    uniform_sample_ranges={"rho": (-0.95, 0.5)},
    output_file="calibrated_heston_parameters.csv",
    structural_checks=heston_checks,
)


# ---------------------------------------------------------
# SVJJ
# ---------------------------------------------------------


def svjj_checks(p):
    """
    Assumes dV = kappa_v (vbar - V) dt + sigma_v sqrt(V) dW, jumps at intensity
    lmbda with J_v ~ Exp(mu_v) and J_y | J_v ~ N(mu_s + rho_j J_v, sigma^2).
    """
    feller = 2 * p["kappa_v"] * p["vbar"] / p["sigma_v"] ** 2
    status = (
        "satisfied"
        if feller > 1
        else "violated (fine for Fourier pricing; matters for simulation)"
    )
    print(f"\nFeller ratio 2*kappa*vbar/sigma_v^2 = {feller:.3f} -> {status}")
    long_run = p["vbar"] + p["lmbda"] * p["mu_v"] / p["kappa_v"]
    print(
        f"Long-run variance incl. jumps = {long_run:.5f} "
        f"(vol {np.sqrt(long_run):.2%}); diffusive vbar alone = {p['vbar']:.5f} "
        f"(vol {np.sqrt(p['vbar']):.2%})"
    )
    print(f"Initial variance V0 = {p['V0']:.5f} (vol {np.sqrt(p['V0']):.2%})")
    print(f"Mean log price jump = {p['mu_s'] + p['rho_j'] * p['mu_v']:+.4f}")


SVJJ = ModelSpec(
    name="SVJJ",
    model_cls=SVJJModel,
    # lmbda, rho, vbar, kappa_v, sigma_v, mu_s, sigma, rho_j, mu_v
    model_params=SVJJ_PARAMS,
    initial_guess=[0.50, -0.50, 0.04, 2.0, 0.50, -0.10, 0.15, -0.50, 0.10, 0.04],
    # rho_j <= 0 and mu_v > 0  =>  rho_j * mu_v <= 0 < 1 (restriction always holds)
    lower=[1e-4, -0.999, 1e-4, 1e-2, 1e-2, -1.0, 1e-3, -10.0, 1e-3, 1e-4],
    upper=[10.0, 0.999, 1.0, 20.0, 3.0, 1.0, 1.5, 0.0, 1.0, 1.0],
    x_scale=[1.0, 0.5, 0.04, 2.0, 0.5, 0.1, 0.2, 1.0, 0.05, 0.04],
    log_sample_ranges={
        "lmbda": (0.05, 3.0),
        "vbar": (0.01, 0.15),
        "kappa_v": (0.3, 10.0),
        "sigma_v": (0.1, 1.5),
        "sigma": (0.02, 0.6),
        "mu_v": (0.01, 0.5),
        "V0": (0.01, 0.15),
    },
    uniform_sample_ranges={
        "rho": (-0.9, 0.2),
        "mu_s": (-0.3, 0.1),
        "rho_j": (-3.0, -0.05),
    },
    output_file="calibrated_svjj_parameters.csv",
    structural_checks=svjj_checks,
)

# ---------------------------------------------------------
# SVJ
# ---------------------------------------------------------


def svj_checks(p):
    """
    Assumes dV = kappa_v (vbar - V) dt + sigma_v sqrt(V) dW, jumps at intensity
    lmbda with J_y ~ N(mu_s, sigma^2).
    """
    feller = 2 * p["kappa_v"] * p["vbar"] / p["sigma_v"] ** 2
    status = (
        "satisfied"
        if feller > 1
        else "violated (fine for Fourier pricing; matters for simulation)"
    )
    print(f"\nFeller ratio 2*kappa*vbar/sigma_v^2 = {feller:.3f} -> {status}")
    long_run = p["vbar"]
    print(f"Long-run variance = {long_run:.5f} " f"(vol {np.sqrt(long_run):.2%})")
    print(f"Initial variance V0 = {p['V0']:.5f} (vol {np.sqrt(p['V0']):.2%})")
    print(f"Mean log price jump = {p['mu_s']:+.4f}")


SVJ = ModelSpec(
    name="SVJ",
    model_cls=SVJModel,
    # lmbda, rho, vbar, kappa_v, sigma_v, mu_s, sigma
    model_params=SVJ_PARAMS,
    initial_guess=[0.50, -0.50, 0.04, 2.0, 0.50, -0.10, 0.15, 0.04],
    lower=[1e-4, -0.999, 1e-4, 1e-2, 1e-2, -1.0, 1e-3, 1e-4],
    upper=[10.0, 0.999, 1.0, 20.0, 3.0, 1.0, 1.5, 1.0],
    x_scale=[1.0, 0.5, 0.04, 2.0, 0.5, 0.1, 0.2, 0.04],
    log_sample_ranges={
        "lmbda": (0.05, 3.0),
        "vbar": (0.01, 0.15),
        "kappa_v": (0.3, 10.0),
        "sigma_v": (0.1, 1.5),
        "sigma": (0.02, 0.6),
        "V0": (0.01, 0.15),
    },
    uniform_sample_ranges={"rho": (-0.9, 0.2), "mu_s": (-0.3, 0.1)},
    output_file="calibrated_svj_parameters.csv",
    structural_checks=svj_checks,
)

SPECS = {"heston": HESTON, "svjj": SVJJ, "svj": SVJ}
