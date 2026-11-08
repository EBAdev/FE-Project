from typing import Literal

import numpy as np
from scipy.integrate import quad

OptionType = Literal["call", "put"]


class FourierPricer:
    """
    Fourier pricing of European options using an arbitrary model
    providing a log-price characteristic function.

    Expected model interface:
        model.characteristic_function(u, t, T, Xt)

    The characteristic function must be:
        E_Q[exp(i*u*log(S_T)) | X_t = Xt]

    The model must expose constant attributes r and q.
    """

    def __init__(
        self,
        model,
        S0: float,
        K: float,
        T: float,
        Xt: np.ndarray,
        t: float = 0.0,
        option_type: OptionType | None = None,
        *,
        epsabs: float = 1e-9,
        epsrel: float = 1e-9,
        limit: int = 250,
    ):
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive.")
        if T < t:
            raise ValueError("Maturity T must be greater than or equal to t.")
        if option_type not in (None, "call", "put"):
            raise ValueError("Invalid option type.")

        self.model = model
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.t = float(t)
        self.tau = self.T - self.t
        self.Xt = np.asarray(Xt, dtype=float)

        self.r = model.r
        self.q = model.q

        self.option_type = option_type
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit

        self.k = np.log(self.K)

    def characteristic_function(self, u):
        """Conditional characteristic function of log(S_T)."""
        return self.model.characteristic_function(u, self.t, self.T, self.Xt)

    def _integrate(self, integrand):
        value, _ = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            limit=self.limit,
        )
        return float(value)

    def _intrinsic_value(self, option_type: OptionType) -> float:
        if option_type == "call":
            return max(self.S0 - self.K, 0.0)
        return max(self.K - self.S0, 0.0)

    # ---------------------------------------------------------
    # Probability-integral representation
    # ---------------------------------------------------------

    def pi1_integrand(self, u: float) -> float:
        numerator = np.exp(-1j * u * self.k) * self.characteristic_function(u - 1j)
        expected_stock = self.S0 * np.exp((self.r - self.q) * self.tau)
        return float(np.real(numerator / (1j * u * expected_stock)))

    def pi2_integrand(self, u: float) -> float:
        numerator = np.exp(-1j * u * self.k) * self.characteristic_function(u)
        return float(np.real(numerator / (1j * u)))

    def pi1(self) -> float:
        return 0.5 + self._integrate(self.pi1_integrand) / np.pi

    def pi2(self) -> float:
        return 0.5 + self._integrate(self.pi2_integrand) / np.pi

    def probability_call_price(self) -> float:
        return float(
            self.S0 * np.exp(-self.q * self.tau) * self.pi1()
            - self.K * np.exp(-self.r * self.tau) * self.pi2()
        )

    # ---------------------------------------------------------
    # Carr-Madan representation
    # ---------------------------------------------------------

    def carr_madan_integrand(self, u: float, alpha: float) -> float:
        shifted_u = u - 1j * (alpha + 1.0)

        denominator = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u

        psi = (
            np.exp(-self.r * self.tau)
            * self.characteristic_function(shifted_u)
            / denominator
        )

        return float(np.real(np.exp(-1j * u * self.k) * psi))

    def carr_madan_call_price(self, alpha: float = 1.5) -> float:
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        integral = self._integrate(lambda u: self.carr_madan_integrand(u, alpha))

        return float(np.exp(-alpha * self.k) * integral / np.pi)

    # ---------------------------------------------------------
    # Lewis representation
    # ---------------------------------------------------------

    def lewis_integrand(self, u: float) -> float:
        """
        Lewis representation using the characteristic function
        of log(S_T / F), where F = S0*exp((r-q)*tau).
        """
        forward = self.S0 * np.exp((self.r - self.q) * self.tau)

        z = u - 0.5j

        # Characteristic function of log(S_T / F).
        phi = np.exp(-1j * z * np.log(forward)) * self.characteristic_function(z)

        phase = np.exp(-1j * u * np.log(self.K / forward))

        return float(np.real(phase * phi) / (u**2 + 0.25))

    def lewis_call_price(self) -> float:
        forward = self.S0 * np.exp((self.r - self.q) * self.tau)

        integral = self._integrate(self.lewis_integrand)

        return float(
            self.S0 * np.exp(-self.q * self.tau)
            - np.sqrt(forward * self.K) * np.exp(-self.r * self.tau) * integral / np.pi
        )

    # ---------------------------------------------------------
    # Public pricing interface
    # ---------------------------------------------------------

    def call_price(self, method: str = "lewis") -> float:
        if self.tau == 0:
            return self._intrinsic_value("call")

        methods = {
            "lewis": self.lewis_call_price,
            "carr_madan": self.carr_madan_call_price,
            "probability": self.probability_call_price,
        }

        if method not in methods:
            raise ValueError(f"Unknown pricing method: {method}")

        return float(methods[method]())

    def put_price(self, method: str = "lewis") -> float:
        if self.tau == 0:
            return self._intrinsic_value("put")

        # Put-call parity with a continuous dividend yield.
        return float(
            self.call_price(method)
            - self.S0 * np.exp(-self.q * self.tau)
            + self.K * np.exp(-self.r * self.tau)
        )

    def price(self, method: str = "lewis"):
        if self.option_type == "call":
            return self.call_price(method)

        if self.option_type == "put":
            return self.put_price(method)

        return {
            "call": self.call_price(method),
            "put": self.put_price(method),
        }


def _integration_grid(
    u_max: float = 400.0,
    panel_width: float = 10.0,
    nodes_per_panel: int = 16,
):
    """Composite Gauss-Legendre nodes and weights on [0, u_max]."""
    x, w = np.polynomial.legendre.leggauss(nodes_per_panel)
    # Finer panels near 0 resolve the 1/(u^2 + 1/4)-type peaks.
    edges = np.concatenate(
        [
            [0.0, 0.5, 1.0, 2.0, 4.0, 7.0],
            np.arange(10.0, u_max + panel_width, panel_width),
        ]
    )
    edges = edges[edges <= u_max]
    left, right = edges[:-1], edges[1:]
    u = (0.5 * (right - left)[:, None] * (x[None, :] + 1.0) + left[:, None]).ravel()
    weights = (0.5 * (right - left)[:, None] * w[None, :]).ravel()
    return u, weights


def carr_madan_call_prices(
    model,
    S0: float,
    strikes: np.ndarray,
    T: float,
    Xt: np.ndarray,
    alpha: float = 1.5,
    **grid_kwargs,
) -> np.ndarray:
    """
    Carr-Madan call prices for many strikes with a single maturity.

    The characteristic function does not depend on the strike, so it is
    evaluated once on a composite Gauss-Legendre grid and reused for every
    strike. The model is assumed to be priced at t = 0.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    log_k = np.log(np.asarray(strikes, dtype=float))
    u, weights = _integration_grid(**grid_kwargs)

    shifted_u = u - 1j * (alpha + 1.0)
    denominator = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
    psi = (
        np.exp(-model.r * T)
        * model.characteristic_function(shifted_u, 0.0, T, Xt)
        / denominator
    )

    phase = np.exp(-1j * np.outer(log_k, u))
    integral = np.real(phase * psi[None, :]) @ weights

    return np.exp(-alpha * log_k) * integral / np.pi


def lewis_call_prices(
    model,
    S0: float,
    strikes: np.ndarray,
    T: float,
    Xt: np.ndarray,
    **grid_kwargs,
) -> np.ndarray:
    """Lewis call prices for many strikes with a single maturity.
    Args:
        model: A model object with a characteristic_function method.
        S0: Current price of the underlying asset.
        strikes: Array of strike prices.
        T: Option maturity
        Xt: State variables at time t=0.
        grid_kwargs: Additional keyword arguments for the integration grid.
    Returns:
        Array of call option prices corresponding to the input strikes.
    """
    strikes = np.asarray(strikes, dtype=float)
    forward = S0 * np.exp((model.r - model.q) * T)
    u, weights = _integration_grid(**grid_kwargs)

    z = u - 0.5j
    phi = np.exp(-1j * z * np.log(forward)) * model.characteristic_function(
        z, 0.0, T, Xt
    )

    phase = np.exp(-1j * np.outer(np.log(strikes / forward), u))
    integral = np.real(phase * phi[None, :]) @ (weights / (u**2 + 0.25))

    return (
        S0 * np.exp(-model.q * T)
        - np.sqrt(forward * strikes) * np.exp(-model.r * T) * integral / np.pi
    )


def probability_call_prices(
    model,
    S0: float,
    strikes: np.ndarray,
    T: float,
    Xt: np.ndarray,
    **grid_kwargs,
) -> np.ndarray:
    """Probability-representation (P1/P2) call prices for many strikes."""
    strikes = np.asarray(strikes, dtype=float)
    forward = S0 * np.exp((model.r - model.q) * T)
    u, weights = _integration_grid(**grid_kwargs)

    phase = np.exp(-1j * np.outer(np.log(strikes), u))
    f1 = model.characteristic_function(u - 1j, 0.0, T, Xt) / (1j * u * forward)
    f2 = model.characteristic_function(u, 0.0, T, Xt) / (1j * u)

    pi1 = 0.5 + (np.real(phase * f1[None, :]) @ weights) / np.pi
    pi2 = 0.5 + (np.real(phase * f2[None, :]) @ weights) / np.pi

    return S0 * np.exp(-model.q * T) * pi1 - strikes * np.exp(-model.r * T) * pi2


def put_call_parity(
    prices: np.ndarray,
    underlying_price: float,
    strikes: np.ndarray,
    tau: float,
    r: float,
    q: float,
) -> np.ndarray:
    """Convert call prices to put prices using put-call parity,
    Args:
        prices: Call option prices.
        underlying_price: Current price of the underlying asset.
        strikes: Strike prices.
        tau: Time to maturity.
        r: Risk-free interest rate.
        q: Dividend yield.
    """
    return prices - underlying_price * np.exp(-q * tau) + strikes * np.exp(-r * tau)


PRICERS = {
    "carr_madan": carr_madan_call_prices,
    "lewis": lewis_call_prices,
    "probability": probability_call_prices,
    "put_call_parity": put_call_parity,
}


if __name__ == "__main__":

    # price a five options using all three methods and compare the results
    from Models.SVJJ import SVJJModel

    # Model parameters
    params = {
        "kappa_v": 2.0,
        "vbar": 0.04,
        "sigma_v": 0.3,
        "mu_s": 0.05,
        "sigma": 0.2,
        "rho": -0.5,
        "rho_j": 0.1,
        "mu_v": 0.03,
        "r": 0.01,
        "q": 0.02,
        "lmbda": 0.1,
    }

    model = SVJJModel(**params)

    S0 = 100.0
    strikes = np.array([90, 95, 100, 105, 110])

    T = 0.5
    Xt = np.array([np.log(S0), model.vbar])

    for method in ["carr_madan", "lewis", "probability"]:
        call_prices = PRICERS[method](model, S0, strikes, T, Xt)
        put_prices = put_call_parity(call_prices, S0, strikes, T, model.r, model.q)

        print(f"Method: {method}")
        print("Strike\tCall Price\tPut Price")
        for K, C, P in zip(strikes, call_prices, put_prices):
            print(f"{K}\t{C:.4f}\t\t{P:.4f}")
        print()
