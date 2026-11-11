import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_handler import load_data
from Visual.utils import calculate_market_iv, load_calibrated_parameters


def plot_surface(file_path, parameter_names, model_surface_function, model_name):
    """Plot model implied volatility surface and market observations."""

    # Load market data.
    data = load_data()

    market_data = data["cleaned_dataframe"]
    S0 = data["underlying_close"]
    q = data["q"]
    r_by_maturity = {pd.Timestamp(k): v for k, v in data["r_by_maturity"].items()}

    params = load_calibrated_parameters(file_path, parameter_names)

    maturities, log_moneyness, iv_surface = model_surface_function(
        params,
        S0,
        q,
        r_by_maturity,
        market_data,
    )

    # Calculate market implied volatilities.
    market_data = calculate_market_iv(
        market_data,
        S0,
        q,
        r_by_maturity,
    )

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
    calls = market_data[
        (market_data["cp_flag"] == "C") & market_data["implied_vol"].notna()
    ]

    ax.scatter(
        calls["time_to_maturity"],
        calls["log_moneyness"],
        calls["implied_vol"] * 100,
        color="red",
        s=15,
        alpha=0.8,
        label="Market implied volatility (calls)",
    )

    puts = market_data[
        (market_data["cp_flag"] == "P") & market_data["implied_vol"].notna()
    ]
    ax.scatter(
        puts["time_to_maturity"],
        puts["log_moneyness"],
        puts["implied_vol"] * 100,
        color="darkorange",
        s=15,
        alpha=0.8,
        label="Market implied volatility (puts)",
    )

    ax.set_xlabel("Time to maturity (years)", labelpad=12)
    ax.set_ylabel(r"Log-moneyness $\log(K/S_0)$", labelpad=12)
    ax.set_zlabel("Implied volatility (%)", labelpad=12)
    ax.set_title(f"{model_name} Implied Volatility Surface")

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
