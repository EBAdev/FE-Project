import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_handler import load_data
from Fourier import PRICERS
from Models.SVJJ import PARAMETER_NAMES, SVJJModel
from option_pricing import implied_call_price, implied_volatility_call

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PARAMETER_FILE = "calibrated_svjj_parameters.csv"
METHOD = "lewis"

NUM_STRIKES = 60
NUM_MATURITIES = 30

MIN_LOG_MONEYNESS = -1.5
MAX_LOG_MONEYNESS = 0.4


# ---------------------------------------------------------
# Load calibrated parameters
# ---------------------------------------------------------


def load_calibrated_parameters():
    """Load calibrated SVJJ parameters from CSV."""

    parameters = pd.read_csv(
        PARAMETER_FILE,
        index_col=0,
    )["value"]

    return parameters.loc[PARAMETER_NAMES].to_numpy(dtype=float)


# ---------------------------------------------------------
# Model construction
# ---------------------------------------------------------


def create_model(params, r, q):
    parameters = dict(zip(PARAMETER_NAMES, params))

    return SVJJModel(
        r=r,
        q=q,
        **parameters,
    )


# ---------------------------------------------------------
# Calculate model implied volatility surface
# ---------------------------------------------------------


def calculate_model_surface(
    params,
    S0,
    q,
    r_by_maturity,
    market_data,
):
    """
    Calculate model implied volatilities over a grid of
    log-moneyness and maturities.
    """

    # Maturity grid.
    min_tau = market_data["time_to_maturity"].min()
    max_tau = market_data["time_to_maturity"].max()

    maturities = np.linspace(
        min_tau,
        max_tau,
        NUM_MATURITIES,
    )

    # Log-moneyness grid.
    log_moneyness = np.linspace(
        MIN_LOG_MONEYNESS,
        MAX_LOG_MONEYNESS,
        NUM_STRIKES,
    )

    iv_surface = np.full(
        (len(maturities), len(log_moneyness)),
        np.nan,
    )

    # Interpolate interest rates against maturity.
    observed_rates = []

    for exdate, r in r_by_maturity.items():

        observed_tau = market_data.loc[
            pd.to_datetime(market_data["exdate"]) == pd.Timestamp(exdate),
            "time_to_maturity",
        ]

        if len(observed_tau):
            observed_rates.append((float(observed_tau.iloc[0]), float(r)))

    observed_rates.sort()

    rate_times = np.array([x[0] for x in observed_rates])
    rate_values = np.array([x[1] for x in observed_rates])

    for i, T in enumerate(maturities):

        r = np.interp(T, rate_times, rate_values)

        model = create_model(params, r, q)

        Xt = np.array(
            [
                np.log(S0),
                params[-1],
            ]
        )

        # Convert log-moneyness to strikes.
        strikes = S0 * np.exp(log_moneyness)

        # Price all strikes for this maturity.
        call_prices = PRICERS[METHOD](
            model,
            S0,
            strikes,
            T,
            Xt,
        )

        for j, K in enumerate(strikes):

            iv_surface[i, j] = implied_volatility_call(
                call_prices[j],
                S0,
                K,
                T,
                r,
                q,
            )

    return maturities, log_moneyness, iv_surface


# ---------------------------------------------------------
# Market implied volatility observations
# ---------------------------------------------------------


def calculate_market_iv(market_data, S0, q, r_by_maturity):
    """Calculate implied volatilities for market call and put prices."""

    market_data = market_data.dropna(subset=["Mid"]).copy()

    market_ivs = []

    for _, row in market_data.iterrows():

        exdate = pd.Timestamp(row["exdate"])
        T = float(row["time_to_maturity"])
        K = float(row["Strike"])
        r = r_by_maturity[exdate]

        price = float(row["Mid"])

        if row["cp_flag"] == "P":
            price = implied_call_price(price_underlying=S0, P=price, K=K, T=T, r=r, q=q)

        iv = implied_volatility_call(
            price,
            S0,
            K,
            T,
            r,
            q,
        )

        market_ivs.append(iv)

    market_data["implied_vol"] = market_ivs

    # Calculate market log-moneyness.
    market_data["log_moneyness"] = np.log(market_data["Strike"] / S0)

    return market_data


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------


def plot_surface(
    maturities,
    log_moneyness,
    iv_surface,
    market_data,
):
    """Plot model implied volatility surface and market observations."""

    from mpl_toolkits.mplot3d import Axes3D

    K_grid, T_grid = np.meshgrid(
        log_moneyness,
        maturities,
    )

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        T_grid,
        K_grid,
        iv_surface * 100,
        cmap="viridis",
        alpha=0.85,
        linewidth=0,
        antialiased=True,
    )

    # Overlay market observations.
    valid = market_data["implied_vol"].notna()
    market = market_data.loc[valid]

    ax.scatter(
        market["time_to_maturity"],
        market["log_moneyness"],
        market["implied_vol"] * 100,
        color="red",
        s=15,
        alpha=0.8,
        label="Market implied volatility",
    )

    ax.set_xlabel("Time to maturity (years)", labelpad=12)
    ax.set_ylabel(r"Log-moneyness $\log(K/S_0)$", labelpad=12)
    ax.set_zlabel("Implied volatility (%)", labelpad=12)
    ax.set_title("SVJJ Implied Volatility Surface")

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.6,
        aspect=15,
        label="Implied volatility (%)",
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load market data.
    data = load_data()

    market_data = data["cleaned_dataframe"].copy()
    S0 = data["underlying_close"]
    q = data["q"]

    r_by_maturity = {pd.Timestamp(k): v for k, v in data["r_by_maturity"].items()}

    # Load calibrated SVJJ parameters.
    params = load_calibrated_parameters()

    print("Loaded calibrated parameters:")
    for name, value in zip(PARAMETER_NAMES, params):
        print(f"{name:12s}: {value:.8f}")

    # Calculate model surface.
    maturities, log_moneyness, iv_surface = calculate_model_surface(
        params,
        S0,
        q,
        r_by_maturity,
        market_data,
    )

    # Calculate market implied volatilities.
    market_iv = calculate_market_iv(
        market_data,
        S0,
        q,
        r_by_maturity,
    )

    # Plot.
    plot_surface(
        maturities,
        log_moneyness,
        iv_surface,
        market_iv,
    )
