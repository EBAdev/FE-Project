import numpy as np
import pandas as pd

from Fourier import PRICERS
from Models.SVJJ import PARAMETER_NAMES, SVJJModel
from option_pricing import implied_volatility_call
from Visual.surface import plot_surface
from Visual.utils import create_model

PARAMETER_FILE = "calibrated_svjj_parameters.csv"
METHOD = "lewis"

V0 = (13 / 100) ** 2  # 0.0169

NUM_STRIKES = 100
NUM_MATURITIES = 50

MIN_LOG_MONEYNESS = -2.0
MAX_LOG_MONEYNESS = 0.4


def calculate_svjj_surface(
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

        model = create_model(SVJJModel, PARAMETER_NAMES, params, r, q)

        Xt = np.array(
            [
                np.log(S0),
                V0,
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


if __name__ == "__main__":
    # Plot.
    plot_surface(PARAMETER_FILE, PARAMETER_NAMES, calculate_svjj_surface, "SVJJ")
