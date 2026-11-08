import numpy as np
from Models.AffineModel import AffineModel

PARAMETER_NAMES = [
    "lmbda",
    "rho",
    "vbar",
    "kappa_v",
    "sigma_v",
    "mu_s",
    "sigma",
    "rho_j",
    "mu_v",
]


class SVJJModel(AffineModel):
    """
    Stochastic Volatility Jump-Diffusion (SVJJ) model.

    We assume that the price and vol process have  simultaneous jumps, with arrival intensity lambda >=0.
    * The jumps in vol are iid. exponentially with mean mu_v >=0.
    * Conditionally on a jump in vol = y, the jumps in price are iid. normally distributed with mean mu_S + rho_jy and variance sigma^2.

    Note that for slideset 9
    kappa_v = kappa, theta = vbar, eta = sigma_v.
    """

    def __init__(
        self, r, q, lmbda, rho, vbar, kappa_v, sigma_v, mu_s, sigma, rho_j, mu_v
    ):
        self.r = r
        self.q = q
        self.lmbda = lmbda
        self.rho = rho
        self.vbar = vbar
        self.kappa_v = kappa_v
        self.sigma_v = sigma_v
        self.mu_s = mu_s
        self.sigma = sigma
        self.rho_j = rho_j
        self.mu_v = mu_v
        self.ParameterNames = PARAMETER_NAMES
        # slideset 9 page 8
        self.mubar = np.exp(mu_s + 0.5 * sigma**2) / (1 - rho_j * mu_v) - 1

        rho0 = r
        rho1 = np.array([0.0, 0.0])

        K0 = np.array([r - q - lmbda * self.mubar, kappa_v * vbar])
        K1 = np.array(
            [
                [0.0, -0.5],
                [0.0, -kappa_v],
            ]
        )

        I0 = lmbda
        I1 = np.array([0.0, 0.0])

        H0 = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        H1 = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, rho * sigma_v], [rho * sigma_v, sigma_v**2]],
            ]
        )

        super().__init__(K0, K1, H0, H1, I0, I1, rho0, rho1)

    def a(self, u):
        return 1j * u + u**2

    def b(self, u):
        return self.sigma_v * self.rho * 1j * u - self.kappa_v

    def c(self, u):
        return 1 - self.rho_j * self.mu_v * 1j * u

    def gamma(self, b, a):
        return np.sqrt(b**2 + a * self.sigma_v**2)

    def d(self, a, b, c, gamma, tau):

        frac1 = ((gamma - b) * tau) / ((gamma - b) * c + a * self.mu_v)
        f2_top = (
            2
            * self.mu_v
            * a
            * np.log(
                1
                - (((gamma + b) * c - a * self.mu_v) / (2 * gamma * c))
                * (1 - np.exp(-gamma * tau))
            )
        )
        f2_bottom = (gamma * c) ** 2 - (b * c - a * self.mu_v) ** 2

        return frac1 - f2_top / f2_bottom

    def A0(self, a, b, gamma, tau):

        t1 = ((gamma + b) / self.sigma_v**2) * tau
        t2 = (2 / self.sigma_v**2) * np.log(
            1 - (gamma + b) / (2 * gamma) * (1 - np.exp(-gamma * tau))
        )

        return -self.kappa_v * self.vbar * (t1 + t2)

    def f(self, d, u):
        fact = np.exp(1j * u * self.mu_s - 0.5 * u**2 * self.sigma**2)
        return fact * d

    def A(self, A0, f, u, tau):
        t1 = self.lmbda * tau * (1 + self.mubar * 1j * u)
        drift = 1j * u * (self.r - self.q) * tau
        return A0 - t1 + self.lmbda * f + drift

    def B(self, a, b, gamma, tau):

        top = a * (1 - np.exp(-gamma * tau))
        bottom = 2 * gamma - (gamma + b) * (1 - np.exp(-gamma * tau))
        return -(top / bottom)

    def characteristic_function(self, u, t, T, Yt):
        Xt = Yt[0]
        Vt = Yt[1]

        tau = T - t
        a = self.a(u)
        b = self.b(u)
        c = self.c(u)
        gamma = self.gamma(b, a)
        d = self.d(a, b, c, gamma, tau)
        A0 = self.A0(a, b, gamma, tau)
        f = self.f(d, u)
        A = self.A(A0, f, u, tau)
        B = self.B(a, b, gamma, tau)

        return np.exp(1j * u * Xt) * np.exp(A + B * Vt)
