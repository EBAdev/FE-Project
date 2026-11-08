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


NUM_STRIKES = 100

MIN_LOG_MONEYNESS = -0.50
MAX_LOG_MONEYNESS = 0.25


# ---------------------------------------------------------
# Load calibrated parameters
# ---------------------------------------------------------


def load_calibrated_parameters():
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
# Select market maturity
# ---------------------------------------------------------


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


# ---------------------------------------------------------
# Market implied volatilities
# ---------------------------------------------------------


def calculate_market_iv(market_data, S0, q, r):

    market_data = market_data.dropna(subset=["Mid"]).copy()

    market_ivs = []

    for _, row in market_data.iterrows():

        T = float(row["time_to_maturity"])
        K = float(row["Strike"])
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

    market_data["log_moneyness"] = np.log(market_data["Strike"] / S0)

    return market_data


# ---------------------------------------------------------
# Calculate model implied volatility smile
# ---------------------------------------------------------


def calculate_model_smile(
    market_data,
    params,
    S0,
    q,
    r,
    T,
):
    """Calculate model implied volatility for one maturity."""

    # Use the observed market log-moneyness range.
    market_data["log_moneyness"] = np.log(market_data["Strike"] / S0)
    min_log_moneyness = market_data["log_moneyness"].min()
    max_log_moneyness = market_data["log_moneyness"].max()

    log_moneyness = np.linspace(
        min_log_moneyness,
        max_log_moneyness,
        NUM_STRIKES,
    )

    strikes = S0 * np.exp(log_moneyness)

    model = create_model(params, r, q)

    Xt = np.array(
        [
            np.log(S0),
            params[-1],
        ]
    )

    # Price all strikes in one call.
    call_prices = PRICERS[METHOD](
        model,
        S0,
        strikes,
        T,
        Xt,
    )

    implied_vols = np.array(
        [
            implied_volatility_call(
                price,
                S0,
                K,
                T,
                r,
                q,
            )
            for price, K in zip(call_prices, strikes)
        ]
    )

    return log_moneyness, implied_vols


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------


def plot_smile(
    log_moneyness,
    model_iv,
    market_data,
    T,
    exdate,
):
    """Plot model implied volatility smile and market observations."""

    fig, ax = plt.subplots(figsize=(11, 7))

    # Model smile.
    ax.plot(
        log_moneyness,
        model_iv * 100,
        color="navy",
        linewidth=2,
        label="SVJJ model",
    )

    # Market calls.
    calls = market_data[
        (market_data["cp_flag"] == "C") & market_data["implied_vol"].notna()
    ]

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
    puts = market_data[
        (market_data["cp_flag"] == "P") & market_data["implied_vol"].notna()
    ]

    ax.scatter(
        puts["log_moneyness"],
        puts["implied_vol"] * 100,
        color="darkorange",
        s=35,
        alpha=0.8,
        marker="^",
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
        f"SVJJ Implied Volatility Smile - " f"{exdate.date()} (T = {T:.3f} years)"
    )

    ax.grid(alpha=0.25)
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

    # Load calibrated parameters.
    params = load_calibrated_parameters()

    print("Loaded calibrated parameters:")
    for name, value in zip(PARAMETER_NAMES, params):
        print(f"{name:12s}: {value:.8f}")

    # Select an actual market maturity.
    exdate, T = select_market_maturity(market_data)

    # Extract market observations for selected maturity.
    selected_market = market_data[
        pd.to_datetime(market_data["exdate"]) == exdate
    ].copy()

    if exdate not in r_by_maturity:
        raise ValueError(f"No interest rate available for {exdate.date()}.")

    r = r_by_maturity[exdate]

    print(f"Interest rate:       {r:.6f}")
    print(f"Market observations: {len(selected_market)}")

    # Calculate market implied volatilities.
    market_iv = calculate_market_iv(
        selected_market,
        S0,
        q,
        r,
    )

    # Calculate model implied volatility smile.
    log_moneyness, model_iv = calculate_model_smile(
        selected_market,
        params,
        S0,
        q,
        r,
        T,
    )

    # Plot comparison.
    plot_smile(
        log_moneyness,
        model_iv,
        market_iv,
        T,
        exdate,
    )
