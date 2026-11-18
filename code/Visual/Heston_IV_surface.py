import numpy as np
import pandas as pd

from Fourier import PRICERS
from Models.Heston import PARAMETER_NAMES, HestonModel
from Visual.surface import plot_surface
from Visual.utils import surface_iv_from_calls

PARAMETER_FILE = "calibrated_heston_parameters.csv"
CALIBRATION_PARAMETER_NAMES = list(PARAMETER_NAMES) + ["V0"]

METHOD = "lewis"

NUM_STRIKES = 200
NUM_MATURITIES = 100

MIN_LOG_MONEYNESS = -2.0
MAX_LOG_MONEYNESS = 0.4


def calculate_heston_surface(
    params,
    S0,
    q,
    r_by_maturity,
    market_data,
):
    """
    Calculate Heston model implied volatilities over a grid of
    log-moneyness and maturities.

    Parameter ordering:
        [kappa, theta, sigma, rho, V0]
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

    # Convert log-moneyness to strikes.
    strikes = S0 * np.exp(log_moneyness)

    for i, T in enumerate(maturities):

        r = np.interp(T, rate_times, rate_values)

        # Construct Heston model using the updated interface.
        model = HestonModel(r=r, q=q, **dict(zip(PARAMETER_NAMES, params)))

        Xt = np.array(
            [
                np.log(S0),
                params[-1],  # V0 is the last parameter
            ]
        )

        # Price all strikes for this maturity.
        call_prices = PRICERS[METHOD](
            model,
            S0,
            strikes,
            T,
            Xt,
        )

        iv_surface[i] = surface_iv_from_calls(call_prices, S0, strikes, T, r, q)

    return maturities, log_moneyness, iv_surface


if __name__ == "__main__":

    plot_surface(
        PARAMETER_FILE,
        CALIBRATION_PARAMETER_NAMES,
        calculate_heston_surface,
        "Heston",
    )
