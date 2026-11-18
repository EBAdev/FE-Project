import numpy as np

PARAMETER_NAMES = [
    "kappa",
    "theta",
    "sigma",
    "rho",
]


class HestonModel:
    """
    Heston stochastic volatility model.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the variance process.
    theta : float
        Long-run variance.
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between the asset and variance Brownian motions.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    """

    def __init__(self, kappa, theta, sigma, rho, r, q):
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if not -1 <= rho <= 1:
            raise ValueError("rho must lie in [-1, 1].")

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.q = q

    def characteristic_function(self, u, t, T, Xt):
        """
        Characteristic function of log(S_T), conditional on X_t.

        Xt = [log(S_t), v_t]
        """
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        r = self.r
        q = self.q

        Yt, Vt = Xt
        tau = T - t

        if tau < 0:
            raise ValueError("T must be greater than or equal to t.")

        if tau == 0:
            return np.exp(1j * u * Yt)

        iu = 1j * u

        b = kappa - rho * sigma * iu

        d = np.sqrt(b**2 + sigma**2 * (u**2 + iu))

        # Select the square-root branch with non-negative real part.
        d = np.where(np.real(d) < 0, -d, d)

        g = (b - d) / (b + d)

        exp_dt = np.exp(-d * tau)

        C = (
            kappa
            * theta
            / sigma**2
            * ((b - d) * tau - 2 * np.log((1 - g * exp_dt) / (1 - g)))
        )

        D = (b - d) / sigma**2 * (1 - exp_dt) / (1 - g * exp_dt)

        return np.exp(iu * (Yt + (r - q) * tau) + C + D * Vt)


if __name__ == "__main__":

    # Example parameters for testing
    S0 = 100.0
    V0 = 0.04
    r = 0.05
    q = 0.02
    T = 1.0  # 1 year to maturity

    params = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.2,
        "rho": -0.5,
    }

    model = HestonModel(r=r, q=q, **params)

    print("\nHeston characteristic function diagnostics")
    print("----------------------------------------")

    Xt = np.array([np.log(S0), V0])

    phi_zero = model.characteristic_function(0.0, 0.0, T, Xt)

    expected_martingale = S0 * np.exp((r - q) * T)

    phi_minus_i = model.characteristic_function(-1j, 0.0, T, Xt)

    print(f"Characteristic function at u=0: {phi_zero}")
    print(f"Expected martingale value: {expected_martingale}")
    print(f"Characteristic function at u=-i: {phi_minus_i}")
