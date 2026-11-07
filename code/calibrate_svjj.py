import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr
from scipy.stats import norm

from data_handler import load_data
from Fourier import carr_madan_call_prices, lewis_call_prices, probability_call_prices
from Models import SVJJModel

# ---------------------------------------------------------
# SVJJ parameters
# ---------------------------------------------------------

PARAMETER_NAMES = [
    "lmbda",
    "rho",
    "vbar",
    "kappa_v",
    "sigma_v",
    "mu_s",
    "sigma",
    "rho_j",
    "mu_v",
]

INITIAL_GUESS = np.array(
    [
        0.1,  # lmbda
        -0.7,  # rho
        0.04,  # vbar
        2.0,  # kappa_v
        0.3,  # sigma_v
        -0.1,  # mu_s
        0.15,  # sigma
        0.1,  # rho_j
        0.2,  # mu_v
    ]
)

# Broad parameter bounds.
LOWER_BOUNDS = np.array(
    [
        0.001,  # lmbda
        -0.99,  # rho
        0.001,  # vbar
        0.05,  # kappa_v
        0.01,  # sigma_v
        -1.0,  # mu_s
        0.001,  # sigma
        -0.99,  # rho_j
        0.001,  # mu_v
    ]
)

UPPER_BOUNDS = np.array(
    [
        3.0,  # lmbda
        0.99,  # rho
        0.5,  # vbar
        10.0,  # kappa_v
        3.0,  # sigma_v
        1.0,  # mu_s
        1.0,  # sigma
        0.99,  # rho_j
        2.0,  # mu_v
    ]
)


# ---------------------------------------------------------
# Model pricing
# ---------------------------------------------------------


def create_model(params, r, q):
    """
    Construct an SVJJ model from the parameter vector.
    """

    parameters = dict(zip(PARAMETER_NAMES, params))

    return SVJJModel(
        r=r,
        q=q,
        **parameters,
    )


def price_options(params, groups, S0, q):
    """
    Price all market options. `groups` holds, per maturity, the option
    indices, strikes, call mask and rate (see build_groups).
    Returns model prices in the original option order.
    """

    n = sum(len(g["idx"]) for g in groups)
    prices = np.empty(n)

    for g in groups:
        model = create_model(params, g["r"], q)
        Xt = np.array([np.log(S0), V0])
        T = g["T"]

        calls = PRICERS[METHOD](model, S0, g["K"], T, Xt)
        puts = calls - S0 * np.exp(-q * T) + g["K"] * np.exp(-g["r"] * T)

        prices[g["idx"]] = np.where(g["is_call"], calls, puts)

    return prices


def build_groups(market_data, r_by_maturity):
    """
    Group options by maturity so the characteristic function is evaluated
    once per maturity. Indices refer to the row order of market_data.
    """

    exdates = pd.to_datetime(market_data["exdate"]).to_numpy()
    strikes = market_data["Strike"].to_numpy(dtype=float)
    maturities = market_data["time_to_maturity"].to_numpy(dtype=float)
    is_call = (market_data["cp_flag"] == "C").to_numpy()

    groups = []
    for exdate in np.unique(exdates):
        idx = np.flatnonzero(exdates == exdate)
        groups.append(
            {
                "idx": idx,
                "K": strikes[idx],
                "T": float(maturities[idx[0]]),
                "is_call": is_call[idx],
                "r": r_by_maturity[pd.Timestamp(exdate)],
            }
        )

    return groups


# ---------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------

V0 = 0.04  # Initial variance

PRICERS = {
    "carr_madan": carr_madan_call_prices,
    "lewis": lewis_call_prices,
    "probability": probability_call_prices,
}
METHOD = "lewis"

IV_LOWER = 1e-4
IV_UPPER = 5.0
IV_FAILURE_PENALTY = 1.0


def black_scholes_price(sigma, S0, K, T, r, q, is_call):
    """
    Black-Scholes price with continuous dividend yield q.
    """

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)

    if is_call:
        return S0 * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)

    return K * df_r * norm.cdf(-d2) - S0 * df_q * norm.cdf(-d1)


def implied_volatility(price, S0, K, T, r, q, is_call):
    """
    Invert Black-Scholes for the implied volatility.
    Returns NaN if the price is outside the no-arbitrage range.
    """

    if not np.isfinite(price):
        return np.nan

    def objective(sigma):
        return black_scholes_price(sigma, S0, K, T, r, q, is_call) - price

    try:
        if objective(IV_LOWER) > 0 or objective(IV_UPPER) < 0:
            return np.nan
        return brentq(objective, IV_LOWER, IV_UPPER, xtol=1e-10)
    except (ValueError, FloatingPointError):
        return np.nan


def implied_volatility_vec(prices, S0, K, T, r, q, is_call, n_iter=60):
    """
    Vectorised implied volatility by bisection on [IV_LOWER, IV_UPPER].
    Returns NaN where the price is outside the attainable range.
    """

    sqrt_T = np.sqrt(T)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)

    def bs(sigma):
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        call = S0 * df_q * ndtr(d1) - K * df_r * ndtr(d2)
        put = K * df_r * ndtr(-d2) - S0 * df_q * ndtr(-d1)
        return np.where(is_call, call, put)

    lo = np.full(len(prices), IV_LOWER)
    hi = np.full(len(prices), IV_UPPER)
    valid = np.isfinite(prices) & (bs(lo) <= prices) & (bs(hi) >= prices)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_high = bs(mid) > prices
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)

    return np.where(valid, 0.5 * (lo + hi), np.nan)


def add_market_implied_vols(market_data, S0, q, r_by_maturity):
    """
    Add a 'market_iv' column (from mid prices) and drop options whose
    implied volatility cannot be computed.
    """

    r_by_maturity = {pd.Timestamp(k): v for k, v in r_by_maturity.items()}

    ivs = [
        implied_volatility(
            price=row["Mid"],
            S0=S0,
            K=float(row["Strike"]),
            T=float(row["time_to_maturity"]),
            r=r_by_maturity[pd.Timestamp(row["exdate"])],
            q=q,
            is_call=row["cp_flag"] == "C",
        )
        for _, row in market_data.iterrows()
    ]

    market_data = market_data.copy()
    market_data["market_iv"] = ivs

    n_before = len(market_data)
    market_data = market_data.dropna(subset=["market_iv"])
    if len(market_data) < n_before:
        print(
            f"Dropped {n_before - len(market_data)} options "
            "without a valid market implied volatility."
        )

    return market_data


# ---------------------------------------------------------
# Calibration objective
# ---------------------------------------------------------


def calibration_residuals(
    params,
    market_data,
    S0,
    q,
    r_by_maturity,
    groups=None,
):
    """
    Return implied volatility residuals (model IV - market IV).
    market_data must contain a 'market_iv' column.
    """

    if groups is None:
        groups = build_groups(market_data, r_by_maturity)

    model_prices = price_options(params, groups, S0, q)

    market_iv = market_data["market_iv"].to_numpy(dtype=float)
    strikes = market_data["Strike"].to_numpy(dtype=float)
    maturities = market_data["time_to_maturity"].to_numpy(dtype=float)
    is_call = (market_data["cp_flag"] == "C").to_numpy()
    rates = np.empty(len(market_data))
    for g in groups:
        rates[g["idx"]] = g["r"]

    model_iv = implied_volatility_vec(
        model_prices, S0, strikes, maturities, rates, q, is_call
    )

    return np.where(np.isnan(model_iv), IV_FAILURE_PENALTY, model_iv - market_iv)


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------


def calibrate_svjj(
    market_data,
    S0,
    q,
    r_by_maturity,
    initial_guess=INITIAL_GUESS,
):
    """
    Calibrate SVJJ parameters by fitting the model implied volatility
    surface to the market implied volatilities.
    """

    # Normalize maturity keys
    r_by_maturity = {pd.Timestamp(k): v for k, v in r_by_maturity.items()}

    market_data = add_market_implied_vols(market_data, S0, q, r_by_maturity)

    market_data = market_data.reset_index(drop=True)
    groups = build_groups(market_data, r_by_maturity)

    print(f"Calibrating to {len(market_data)} options...")
    print(f"Underlying price: {S0:.4f}")
    print(f"Dividend yield: {q:.6f}")
    print(f"Pricing method: {METHOD}")

    result = least_squares(
        fun=calibration_residuals,
        x0=initial_guess,
        args=(
            market_data,
            S0,
            q,
            r_by_maturity,
            groups,
        ),
        method="trf",
        loss="linear",
        f_scale=0.01,
        x_scale="jac",
        max_nfev=1000,
        verbose=2,
    )

    if not result.success:
        print("Warning:", result.message)

    calibrated_parameters = dict(zip(PARAMETER_NAMES, result.x))

    return calibrated_parameters, result


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    # Number of options per maturity and option type.
    # For example, 10 calls and 10 puts per maturity.
    NUM_OPTIONS = 10

    # Load market data
    data = load_data()
    df = data["cleaned_dataframe"]
    q = data["q"]
    r_by_maturity = data["r_by_maturity"]
    S0 = data["underlying_close"]

    print(f"Inferred dividend yield: {q:.6f}")
    for maturity, r in r_by_maturity.items():
        print(f"Inferred discount rate for {maturity.date()}: {r:.6f}")

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------

    parameters, result = calibrate_svjj(
        market_data=df,
        S0=S0,
        q=q,
        r_by_maturity=r_by_maturity,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\nCalibrated SVJJ parameters")
    print("--------------------------")

    for name, value in parameters.items():
        print(f"{name:12s}: {value:.8f}")

    print("\nCalibration statistics")
    print("---------------------")
    print(f"Success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Cost: {result.cost:.8f}")
    print(f"Optimality: {result.optimality:.8f}")

    # Calculate final implied volatility errors
    residuals = calibration_residuals(
        result.x,
        add_market_implied_vols(df, S0, q, r_by_maturity).reset_index(drop=True),
        S0,
        q,
        r_by_maturity,
    )

    print(f"Mean absolute IV error: {np.mean(np.abs(residuals)):.6f}")

    # Save calibrated parameters
    pd.Series(parameters).to_csv(
        "calibrated_svjj_parameters.csv",
        header=["value"],
    )
