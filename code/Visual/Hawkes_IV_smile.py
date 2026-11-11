import numpy as np

from Fourier import PRICERS
from Models.Hawkes import PARAMETER_NAMES, HawkesModel
from option_pricing import implied_volatility_call
from Visual.smile import plot_smile
from Visual.utils import create_model

PARAMETER_FILE = "calibrated_hawkes_parameters.csv"
METHOD = "lewis"


NUM_STRIKES = 100

MIN_LOG_MONEYNESS = -0.50
MAX_LOG_MONEYNESS = 0.25


def calculate_hawkes_smile(
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

    model = create_model(HawkesModel, PARAMETER_NAMES, params, r, q)

    # Initialize Hawkes intensity at its stationary mean.
    lambda0 = model.stationary_lambda()
    # the initial state vector for the Hawkes model is [log(S0), lambda0]
    Yt = np.array([np.log(S0), lambda0])

    # Price all strikes in one call.
    call_prices = PRICERS[METHOD](model, S0, strikes, T, Yt)

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


if __name__ == "__main__":

    # Plot.
    plot_smile(PARAMETER_FILE, PARAMETER_NAMES, calculate_hawkes_smile, "Hawkes")
