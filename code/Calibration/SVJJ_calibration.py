import numpy as np
import pandas as pd
from data_handler import load_data
from Fourier import PRICERS
from Models.SVJJ import PARAMETER_NAMES, SVJJModel
from scipy.optimize import least_squares

# ---------------------------------------------------------
# SVJJ parameters
# ---------------------------------------------------------

V0 = (13 / 100) ** 2  # 0.0169

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
# Pricing configuration
# ---------------------------------------------------------

METHOD = "lewis"
PRICE_FLOOR = 1.0

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


def price_options(params, groups, price_underlying, q):
    """
    Price all market options. `groups` holds, per maturity, the option
    indices, strikes, call mask and rate (see build_groups).
    Returns model prices in the original option order.
    """

    n = sum(len(g["idx"]) for g in groups)
    prices = np.empty(n)

    for g in groups:
        r = g["r"]
        tau = g["tau"]
        K = g["K"]
        model = create_model(params, r, q)
        # The state vector is (log(S), V) where V is the variance process.
        Yt = np.array([np.log(price_underlying), V0])

        calls = PRICERS[METHOD](model, price_underlying, K, tau, Yt)
        puts = PRICERS["put_call_parity"](calls, price_underlying, K, tau, r, q)

        prices[g["idx"]] = np.where(g["is_call"], calls, puts)

    return prices


def build_groups(market_data, r_by_maturity):
    """
    Group options by maturity so the characteristic function is evaluated
    once per maturity. Indices refer to the row order of market_data.
    """

    exdates = pd.to_datetime(market_data["exdate"]).to_numpy()
    strikes = market_data["Strike"].to_numpy(dtype=float)
    taus = market_data["time_to_maturity"].to_numpy(dtype=float)
    is_call = (market_data["cp_flag"] == "C").to_numpy()

    groups = []
    for exdate in np.unique(exdates):
        idx = np.flatnonzero(exdates == exdate)
        groups.append(
            {
                "idx": idx,
                "K": strikes[idx],
                "tau": float(taus[idx[0]]),
                "is_call": is_call[idx],
                "r": r_by_maturity[pd.Timestamp(exdate)],
            }
        )

    return groups


# ---------------------------------------------------------
# Calibration objective
# ---------------------------------------------------------


def calibration_residuals(
    params,
    market_data,
    price_underlying,
    q,
    r_by_maturity,
    groups=None,
):
    """
    Return price residuals (model price - market mid price), scaled by
    max(market price, PRICE_FLOOR) so cheap and expensive options both count.
    """

    if groups is None:
        groups = build_groups(market_data, r_by_maturity)

    model_prices = price_options(params, groups, price_underlying, q)
    market_prices = market_data["Mid"].to_numpy(dtype=float)

    return (model_prices - market_prices) / np.maximum(market_prices, PRICE_FLOOR)


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------


def calibrate_svjj(
    market_data,
    price_underlying,
    q,
    r_by_maturity,
    initial_guess=INITIAL_GUESS,
):
    """
    Calibrate SVJJ parameters by fitting model prices to market mid prices.
    """

    # Normalize maturity keys
    r_by_maturity = {pd.Timestamp(k): v for k, v in r_by_maturity.items()}

    market_data = market_data.dropna(subset=["Mid"]).reset_index(drop=True)
    groups = build_groups(market_data, r_by_maturity)

    print(f"Calibrating to {len(market_data)} options...")
    print(f"Underlying price: {price_underlying:.4f}")
    print(f"Dividend yield: {q:.6f}")
    print(f"Pricing method: {METHOD}")

    result = least_squares(
        fun=calibration_residuals,
        x0=initial_guess,
        args=(
            market_data,
            price_underlying,
            q,
            r_by_maturity,
            groups,
        ),
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method="trf",
        loss="soft_l1",
        f_scale=0.05,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        x_scale="jac",
        max_nfev=300,
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
        price_underlying=S0,
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

    # Calculate final price errors
    residuals = calibration_residuals(
        result.x,
        df.dropna(subset=["Mid"]).reset_index(drop=True),
        S0,
        q,
        r_by_maturity,
    )

    print(
        f"Mean absolute relative price error: {np.mean(np.abs(residuals)):.6f}\nRMSE: {np.sqrt(np.mean(residuals**2)):.6f}"
    )

    # Save calibrated parameters
    pd.Series(parameters).to_csv(
        "calibrated_svjj_parameters.csv",
        header=["value"],
    )
