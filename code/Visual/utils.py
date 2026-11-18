import numpy as np
import pandas as pd

from option_pricing import (
    implied_call_price,
    implied_put_price,
    implied_volatility_call,
)


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


def surface_iv_from_calls(call_prices, S0, strikes, T, r, q, min_price=1e-6):
    """
    Implied vols for a row of model call prices, robust to Fourier noise.

    Out-of-the-money options are inverted (puts below the forward, calls above)
    since deep ITM prices are dominated by intrinsic value. Points whose
    time value is at the pricing noise level are returned as NaN.
    """
    ivs = np.full(len(strikes), np.nan)
    forward = S0 * np.exp((r - q) * T)

    for j, K in enumerate(strikes):
        price = call_prices[j]
        if not np.isfinite(price):
            continue

        if K < forward:
            put = implied_put_price(S0, price, K, T, r, q)
            if put < min_price:
                continue
            price = implied_call_price(S0, put, K, T, r, q)
        elif price < min_price:
            continue

        iv = implied_volatility_call(price, S0, K, T, r, q)
        if np.isfinite(iv) and iv > 1e-4:
            ivs[j] = iv

    return ivs
