import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_handler import load_data
from Fourier import PRICERS
from Models.Heston import PARAMETER_NAMES as HESTON_PARAMETER_NAMES
from Models.Heston import HestonModel
from Models.SVJJ import PARAMETER_NAMES as SVJJ_PARAMETER_NAMES
from Models.SVJJ import SVJJModel
from Visual.smile import select_market_maturity
from Visual.utils import create_model, load_calibrated_parameters

SVJJ_PARAMETER_FILE = "calibrated_svjj_parameters_bidask.csv"
HESTON_PARAMETER_FILE = "calibrated_heston_parameters_bidask.csv"
# The calibrated initial variance V0 is stored as the last parameter.
SVJJ_CALIBRATION_PARAMETER_NAMES = list(SVJJ_PARAMETER_NAMES) + ["V0"]
HESTON_CALIBRATION_PARAMETER_NAMES = list(HESTON_PARAMETER_NAMES) + ["V0"]

METHOD = "lewis"
NUM_STRIKES = 100
# The calibrated SVJJ has a near-degenerate jump size (sigma ~ 0.003), so its
# characteristic function decays slowly; the default u_max = 400 leaves visible
# truncation ripples at short maturities.
U_MAX = 4000.0


def calculate_model_price(Model, parameter_names, market_data, params, S0, q, r, T):
    """Model price for one maturity; params = model params + [V0]."""

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

    return log_moneyness, call_prices


def plot_price_comparison():
    """Plot SVJJ and Heston prices against the market prices with markets for bids and asks."""

    data = load_data()

    market_data = data["cleaned_dataframe"].copy()
    S0 = data["underlying_close"]
    q = data["q"]
    r_by_maturity = {pd.Timestamp(k): v for k, v in data["r_by_maturity"].items()}
    # Calculate market log-moneyness.
    market_data["log_moneyness"] = np.log(market_data["Strike"] / S0)

    svjj_params = load_calibrated_parameters(
        SVJJ_PARAMETER_FILE, SVJJ_CALIBRATION_PARAMETER_NAMES
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

    market_prices = selected_market["Mid"].values
    market_bids = selected_market["best_bid"].values
    market_asks = selected_market["best_offer"].values

    svjj_lm, svjj_price = calculate_model_price(
        SVJJModel, SVJJ_PARAMETER_NAMES, selected_market, svjj_params, S0, q, r, T
    )
    heston_lm, heston_price = calculate_model_price(
        HestonModel, HESTON_PARAMETER_NAMES, selected_market, heston_params, S0, q, r, T
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(svjj_lm, svjj_price, color="navy", linewidth=2, label="SVJJ model")
    ax.plot(
        heston_lm,
        heston_price,
        color="green",
        linewidth=2,
        linestyle="-.",
        label="Heston model",
    )

    ax.scatter(
        selected_market["log_moneyness"],
        market_prices,
        color="black",
        s=15,
        alpha=0.5,
        label="Market mid",
        zorder=3,
    )

    ax.scatter(
        selected_market["log_moneyness"],
        market_bids,
        color="red",
        s=15,
        alpha=0.5,
        label="Market bid",
        zorder=3,
    )

    ax.scatter(
        selected_market["log_moneyness"],
        market_asks,
        color="orange",
        s=15,
        alpha=0.5,
        label="Market ask",
        zorder=3,
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="ATM")

    ax.set_xlabel(r"Log-moneyness $\log(K/S_0)$")
    ax.set_ylabel("Price")
    ax.set_title(f"SVJJ vs Heston Price fit- " f"{exdate.date()} (T = {T:.3f} years)")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_price_comparison()
