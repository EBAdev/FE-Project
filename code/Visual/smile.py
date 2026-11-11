import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_handler import load_data
from Visual.utils import calculate_market_iv, load_calibrated_parameters


def select_market_maturity(market_data):
    """Let the user select an available market maturity."""

    available = (
        market_data.dropna(subset=["Mid", "exdate", "time_to_maturity"])
        .groupby("exdate")["time_to_maturity"]
        .median()
        .reset_index()
    )

    available["exdate"] = pd.to_datetime(available["exdate"])
    available = available.sort_values("time_to_maturity").reset_index(drop=True)

    if available.empty:
        raise ValueError("No market maturities available.")

    print("\nAvailable market maturities:")
    print("-" * 55)

    for i, row in available.iterrows():
        exdate = row["exdate"].date()
        T = row["time_to_maturity"]

        num_options = len(
            market_data[pd.to_datetime(market_data["exdate"]) == row["exdate"]]
        )

        print(
            f"{i + 1:3d}. Expiration: {exdate} | "
            f"Maturity: {T:.4f} years | "
            f"Options: {num_options}"
        )

    print("-" * 55)

    while True:
        try:
            choice = int(input("\nSelect maturity (enter number): "))

            if 1 <= choice <= len(available):
                break

            print(f"Please enter a number between 1 and {len(available)}.")

        except ValueError:
            print("Invalid input. Please enter an integer.")

    selected = available.iloc[choice - 1]

    exdate = selected["exdate"]
    T = float(selected["time_to_maturity"])

    print(f"\nSelected expiration: {exdate.date()}")
    print(f"Selected maturity:   {T:.6f} years")

    return exdate, T


def plot_smile(file_path, parameter_names, model_smile_function, model_name):
    """Plot model implied volatility smile and market observations."""

    # Load market data.
    data = load_data()

    market_data = data["cleaned_dataframe"].copy()
    S0 = data["underlying_close"]
    q = data["q"]

    r_by_maturity = {pd.Timestamp(k): v for k, v in data["r_by_maturity"].items()}

    # Load calibrated parameters.
    params = load_calibrated_parameters(file_path, parameter_names)

    # Select an actual market maturity.
    exdate, T = select_market_maturity(market_data)

    # Extract market observations for selected maturity.
    selected_market = market_data[
        pd.to_datetime(market_data["exdate"]) == exdate
    ].copy()

    if exdate not in r_by_maturity:
        raise ValueError(f"No interest rate available for {exdate.date()}.")

    r = r_by_maturity[exdate]

    # Calculate market implied volatilities.
    market_iv = calculate_market_iv(selected_market, S0, q, {exdate: r})

    # Calculate model implied volatility smile.
    log_moneyness, model_iv = model_smile_function(
        selected_market,
        params,
        S0,
        q,
        r,
        T,
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    # Model smile.
    ax.plot(
        log_moneyness,
        model_iv * 100,
        color="navy",
        linewidth=2,
        label=f"{model_name} model",
    )

    # Market calls.
    calls = market_iv[(market_iv["cp_flag"] == "C") & market_iv["implied_vol"].notna()]

    ax.scatter(
        calls["log_moneyness"],
        calls["implied_vol"] * 100,
        color="red",
        s=35,
        alpha=0.8,
        label="Market calls",
        zorder=3,
    )

    # Market puts.
    puts = market_iv[(market_iv["cp_flag"] == "P") & market_iv["implied_vol"].notna()]

    ax.scatter(
        puts["log_moneyness"],
        puts["implied_vol"] * 100,
        color="darkorange",
        s=35,
        alpha=0.8,
        label="Market puts",
        zorder=3,
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="ATM",
    )

    ax.set_xlabel(r"Log-moneyness $\log(K/S_0)$")
    ax.set_ylabel("Implied volatility (%)")

    ax.set_title(
        f"{model_name} Implied Volatility Smile - "
        f"{exdate.date()} (T = {T:.3f} years)"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()
