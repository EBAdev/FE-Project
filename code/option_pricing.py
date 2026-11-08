import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def implied_put_price(S, C, K, T, r, q):
    """Convert call prices to put prices using put-call parity."""
    return C + K * np.exp(-r * T) - S * np.exp(-q * T)


def implied_call_price(S, P, K, T, r, q):
    """Convert put prices to call prices using put-call parity."""
    return P + S * np.exp(-q * T) - K * np.exp(-r * T)


def put_call_parity(
    prices: np.ndarray,
    underlying_price: float,
    strikes: np.ndarray,
    tau: float,
    r: float,
    q: float,
) -> np.ndarray:
    """Convert call prices to put prices using put-call parity."""
    return implied_put_price(underlying_price, prices, strikes, tau, r, q)


def BS_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_put(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_volatility_call(price, S0, K, T, r, q):
    """Calculate Black-Scholes implied volatility from a call price."""
    intrinsic = max(
        S0 * np.exp(-q * T) - K * np.exp(-r * T),
        0.0,
    )
    upper_bound = S0 * np.exp(-q * T)

    if not np.isfinite(price):
        return np.nan

    if price < intrinsic - 1e-8 or price > upper_bound + 1e-8:
        return np.nan

    price = np.clip(price, intrinsic, upper_bound)

    if price - intrinsic < 1e-10:
        return 0.0

    try:
        return brentq(
            lambda sigma: BS_call(S0, K, T, r, q, sigma) - price,
            1e-6,
            5.0,
            xtol=1e-8,
        )
    except ValueError:
        return np.nan


def calculate_implied_volatility(
    price_underlying, option_mid, strike, time_to_maturity, r, q, option_type
):
    """Calculate implied volatility for a call or put option price."""
    if option_type == "C":
        call_price = option_mid
    else:
        call_price = implied_call_price(
            price_underlying,
            option_mid,
            strike,
            time_to_maturity,
            r,
            q,
        )

    return implied_volatility_call(
        call_price,
        price_underlying,
        strike,
        time_to_maturity,
        r,
        q,
    )
