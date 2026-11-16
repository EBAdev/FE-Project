import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def implied_put_price(S, C, K, T, r, q):
    """Convert call prices to put prices using put-call parity."""
    return C + K * np.exp(-r * T) - S * np.exp(-q * T)


def implied_call_price(S, P, K, T, r, q):
    """Convert put prices to call prices using put-call parity."""
    return P + S * np.exp(-q * T) - K * np.exp(-r * T)


def put_call_parity(prices, underlying_price, strikes, tau, r, q):
    """Convert call prices to put prices using put-call parity."""
    return implied_put_price(underlying_price, prices, strikes, tau, r, q)


def BS_call(S, K, T, r, q, sigma):
    """Vectorized Black-Scholes call price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_put(S, K, T, r, q, sigma):
    """Vectorized Black-Scholes put price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def BS_vega(S, K, T, r, q, sigma):
    """Vectorized Black-Scholes vega."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def BS_price(S, K, T, r, q, sigma, is_call):
    """Return call or put prices, supporting scalar and array inputs."""
    return np.where(
        is_call,
        BS_call(S, K, T, r, q, sigma),
        BS_put(S, K, T, r, q, sigma),
    )


def implied_volatility_call(price, S0, K, T, r, q):
    """Calculate Black-Scholes implied volatility from a call price."""
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
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


def implied_volatility(
    price, S, K, T, r, q, is_call, lower_vol=1e-4, upper_vol=5.0, n_iter=80
):
    """
    Vectorized Black-Scholes implied volatility using bisection.

    Returns implied volatilities and a mask identifying valid prices.
    """
    price = np.asarray(price, dtype=float)

    lower = np.full_like(price, lower_vol)
    upper = np.full_like(price, upper_vol)

    p_lower = BS_price(S, K, T, r, q, lower, is_call)
    p_upper = BS_price(S, K, T, r, q, upper, is_call)

    valid = (
        np.isfinite(price)
        & np.isfinite(p_lower)
        & np.isfinite(p_upper)
        & (price > p_lower)
        & (price < p_upper)
    )

    for _ in range(n_iter):
        mid = 0.5 * (lower + upper)
        p_mid = BS_price(S, K, T, r, q, mid, is_call)

        too_high = p_mid > price
        upper = np.where(too_high, mid, upper)
        lower = np.where(too_high, lower, mid)

    return 0.5 * (lower + upper), valid


def calculate_implied_volatility(
    price_underlying, option_mid, strike, time_to_maturity, r, q, option_type
):
    """Calculate implied volatility for an individual call or put."""
    if option_type == "C":
        call_price = option_mid
    else:
        call_price = implied_call_price(
            price_underlying, option_mid, strike, time_to_maturity, r, q
        )

    return implied_volatility_call(
        call_price, price_underlying, strike, time_to_maturity, r, q
    )
