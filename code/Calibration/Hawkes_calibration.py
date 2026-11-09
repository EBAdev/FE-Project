import numpy as np
import pandas as pd
from data_handler import load_data
from Fourier import PRICERS
from Models.Hawkes import PARAMETER_NAMES, HawkesModel
from scipy.optimize import least_squares

# ---------------------------------------------------------
# Hawkes parameters
# ---------------------------------------------------------


INITIAL_GUESS = np.array(
    [
        0.1,  # gamma
        0.2,  # sigma
        2.0,  # delta
        1.0,  # lambdabar
        -0.1,  # epsilon
    ]
)

LOWER_BOUNDS = np.array(
    [
        0.01,  # gamma
        0.01,  # sigma
        0.01,  # delta
        0.01,  # lambdabar
        -2.0,  # epsilon
    ]
)

UPPER_BOUNDS = np.array(
    [
        2.0,  # gamma
        1.0,  # sigma
        20.0,  # delta
        10.0,  # lambdabar
        -0.001,  # epsilon
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
    Construct a Hawkes model from the parameter vector.
    """

    parameters = dict(zip(PARAMETER_NAMES, params))

    return HawkesModel(r=r, q=q, **parameters)


# Identical to SVJJ
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

        # Initialize Hawkes intensity at its stationary mean.
        lambda0 = (
            model.delta * model.lambdabar / (model.delta - model.epsilon * model.gamma)
        )
        # the initial state vector for the Hawkes model is [log(S0), lambda0]
        Yt = np.array([np.log(price_underlying), lambda0])

        calls = PRICERS[METHOD](model, price_underlying, K, tau, Yt)
        puts = PRICERS["put_call_parity"](calls, price_underlying, K, tau, r, q)

        prices[g["idx"]] = np.where(g["is_call"], calls, puts)

    return prices


# identical to SVJJ
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


# identical to SVJJ
def calibration_residuals(
    params,
    market_data,
    price_underlying,
    q,
    r_by_maturity,
    groups=None,
):
    """
    Return scaled price residuals:
        (model price - market mid) / max(market mid, PRICE_FLOOR)
    This ensures cheap and expensive options are weighted equally in the calibration objective.
    """

    if groups is None:
        groups = build_groups(market_data, r_by_maturity)

    model_prices = price_options(
        params,
        groups,
        price_underlying,
        q,
    )

    market_prices = market_data["Mid"].to_numpy(dtype=float)

    return (model_prices - market_prices) / np.maximum(market_prices, PRICE_FLOOR)


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------


def calibrate_hawkes(
    market_data,
    price_underlying,
    q,
    r_by_maturity,
    initial_guess=INITIAL_GUESS,
):
    """
    Calibrate Hawkes parameters by fitting model prices
    to market mid prices.
    """

    # Normalize maturity keys.
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
        loss="soft_l1",  # outlier robust loss function
        f_scale=0.05,
        ftol=1e-12,  # tolerance for termination by the change of the cost function.
        xtol=1e-12,  # tolerance for termination by the change of the solution.
        gtol=1e-12,  # tolerance for termination by the norm of the gradient.
        x_scale="jac",  # scale the variables according to the Jacobian.
        max_nfev=300,  # maximum number of function evaluations.
        verbose=2,  # print progress messages during optimization.
    )

    if not result.success:
        print("Warning:", result.message)

    calibrated_parameters = dict(zip(PARAMETER_NAMES, result.x))

    return calibrated_parameters, result


if __name__ == "__main__":

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    # Number of options per maturity and option type.
    NUM_OPTIONS = 10

    # Load market data.
    data = load_data()

    df = data["cleaned_dataframe"]
    q = data["q"]
    r_by_maturity = data["r_by_maturity"]
    S0 = data["underlying_close"]

    print(f"Inferred dividend yield: {q:.6f}")

    for maturity, r in r_by_maturity.items():
        print(f"Inferred discount rate for " f"{maturity.date()}: {r:.6f}")

    # -----------------------------------------------------
    # Calibration
    # -----------------------------------------------------

    parameters, result = calibrate_hawkes(
        market_data=df,
        price_underlying=S0,
        q=q,
        r_by_maturity=r_by_maturity,
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\nCalibrated Hawkes parameters")
    print("----------------------------")

    for name, value in parameters.items():
        print(f"{name:12s}: {value:.8f}")

    print("\nCalibration statistics")
    print("----------------------")

    print(f"Success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Cost: {result.cost:.8f}")
    print(f"Optimality: {result.optimality:.8f}")

    # Calculate final price errors.
    clean_df = df.dropna(subset=["Mid"]).reset_index(drop=True)

    residuals = calibration_residuals(
        result.x,
        clean_df,
        S0,
        q,
        r_by_maturity,
    )

    print(f"Mean absolute relative price error: " f"{np.mean(np.abs(residuals)):.6f}")

    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")

    # -----------------------------------------------------
    # Save calibrated parameters
    # -----------------------------------------------------

    pd.Series(parameters).to_csv(
        "calibrated_hawkes_parameters.csv",
        header=["value"],
    )
