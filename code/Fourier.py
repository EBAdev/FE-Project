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
