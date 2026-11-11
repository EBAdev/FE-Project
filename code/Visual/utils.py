import numpy as np
import pandas as pd

from option_pricing import implied_call_price, implied_volatility_call


def load_calibrated_parameters(file_path, parameter_names):
    """Load calibrated SVJJ parameters from CSV."""

    parameters = pd.read_csv(
        file_path,
        index_col=0,
    )["value"]

    return parameters.loc[parameter_names].to_numpy(dtype=float)


def create_model(Model, parameter_names, params, r, q):
    parameters = dict(zip(parameter_names, params))

    return Model(
        r=r,
        q=q,
        **parameters,
    )


def calculate_market_iv(market_data, price_underlying, q, r_by_maturity):
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
            price = implied_call_price(price_underlying, P=price, K=K, T=T, r=r, q=q)

        iv = implied_volatility_call(
            price,
            price_underlying,
            K,
            T,
            r,
            q,
        )

        market_ivs.append(iv)

    market_data["implied_vol"] = market_ivs

    # Calculate market log-moneyness.
    market_data["log_moneyness"] = np.log(market_data["Strike"] / price_underlying)

    return market_data
