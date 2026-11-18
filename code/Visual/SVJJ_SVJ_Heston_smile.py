import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_handler import load_data
from Fourier import PRICERS
from Models.Heston import PARAMETER_NAMES as HESTON_PARAMETER_NAMES
from Models.Heston import HestonModel
from Models.SVJ import PARAMETER_NAMES as SVJ_PARAMETER_NAMES
from Models.SVJ import SVJModel
from Models.SVJJ import PARAMETER_NAMES as SVJJ_PARAMETER_NAMES
from Models.SVJJ import SVJJModel
from Visual.smile import select_market_maturity
from Visual.utils import (
    calculate_market_iv,
    create_model,
    load_calibrated_parameters,
    surface_iv_from_calls,
)

SVJJ_PARAMETER_FILE = "calibrated_svjj_parameters.csv"
SVJ_PARAMETER_FILE = "calibrated_svj_parameters.csv"
HESTON_PARAMETER_FILE = "calibrated_heston_parameters.csv"
# The calibrated initial variance V0 is stored as the last parameter.
SVJJ_CALIBRATION_PARAMETER_NAMES = list(SVJJ_PARAMETER_NAMES) + ["V0"]
SVJ_CALIBRATION_PARAMETER_NAMES = list(SVJ_PARAMETER_NAMES) + ["V0"]
HESTON_CALIBRATION_PARAMETER_NAMES = list(HESTON_PARAMETER_NAMES) + ["V0"]

METHOD = "lewis"
NUM_STRIKES = 100
# The calibrated SVJJ has a near-degenerate jump size (sigma ~ 0.003), so its
# characteristic function decays slowly; the default u_max = 400 leaves visible
# truncation ripples at short maturities.
U_MAX = 4000.0


def calculate_model_smile(Model, parameter_names, market_data, params, S0, q, r, T):
    """Model implied volatility for one maturity; params = model params + [V0]."""

    log_moneyness_market = np.log(market_data["Strike"] / S0)

    log_moneyness = np.linspace(
        log_moneyness_market.min(),
        log_moneyness_market.max(),
        NUM_STRIKES,
    )
    strikes = S0 * np.exp(log_moneyness)

    model = create_model(Model, parameter_names, params[:-1], r, q)

    Xt = np.array([np.log(S0), params[-1]])

    call_prices = PRICERS[METHOD](model, S0, strikes, T, Xt, u_max=U_MAX)

    # Invert OTM prices (puts below the forward) to avoid ITM round-off noise.
    implied_vols = surface_iv_from_calls(call_prices, S0, strikes, T, r, q)

    return log_moneyness, implied_vols


def plot_smile_comparison():
    """Plot SVJJ and Heston smiles against market implied volatilities."""

    data = load_data()

    market_data = data["cleaned_dataframe"].copy()
    S0 = data["underlying_close"]
    q = data["q"]
    r_by_maturity = {pd.Timestamp(k): v for k, v in data["r_by_maturity"].items()}

    svjj_params = load_calibrated_parameters(
        SVJJ_PARAMETER_FILE, SVJJ_CALIBRATION_PARAMETER_NAMES
    )
    svj_params = load_calibrated_parameters(
        SVJ_PARAMETER_FILE, SVJ_CALIBRATION_PARAMETER_NAMES
    )
    heston_params = load_calibrated_parameters(
        HESTON_PARAMETER_FILE, HESTON_CALIBRATION_PARAMETER_NAMES
    )

    exdate, T = select_market_maturity(market_data)

    if exdate not in r_by_maturity:
        raise ValueError(f"No interest rate available for {exdate.date()}.")

    r = r_by_maturity[exdate]

    selected_market = market_data[
        pd.to_datetime(market_data["exdate"]) == exdate
    ].copy()

    market_iv = calculate_market_iv(selected_market, S0, q, {exdate: r})

    svjj_lm, svjj_iv = calculate_model_smile(
        SVJJModel, SVJJ_PARAMETER_NAMES, selected_market, svjj_params, S0, q, r, T
    )
    svj_lm, svj_iv = calculate_model_smile(
        SVJModel, SVJ_PARAMETER_NAMES, selected_market, svj_params, S0, q, r, T
    )
    heston_lm, heston_iv = calculate_model_smile(
        HestonModel, HESTON_PARAMETER_NAMES, selected_market, heston_params, S0, q, r, T
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(svjj_lm, svjj_iv * 100, color="navy", linewidth=2, label="SVJJ model")
    ax.plot(svj_lm, svj_iv * 100, color="orange", linewidth=2, label="SVJJ model")
    ax.plot(
        heston_lm,
        heston_iv * 100,
        color="green",
        linewidth=2,
        linestyle="-.",
        label="Heston model",
    )

    options_to_plot = market_iv[market_iv["implied_vol"].notna()]
    ax.scatter(
        options_to_plot["log_moneyness"],
        options_to_plot["implied_vol"] * 100,
        color="red",
        s=15,
        alpha=0.5,
        label="Market options",
        zorder=3,
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="ATM")

    ax.set_xlabel(r"Log-moneyness $\log(K/S_0)$")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(
        f"SVJJ vs Heston Implied Volatility Smile - "
        f"{exdate.date()} (T = {T:.3f} years)"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_smile_comparison()
