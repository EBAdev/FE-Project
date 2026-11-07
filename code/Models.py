import numpy as np


class AffineModel:
    """
    Standard affine model introduced by Duffie, Pan and Singleton (2000).
    """

    def __init__(
        self,
        K0: np.ndarray,
        K1: np.ndarray,
        H0: np.ndarray,
        H1: np.ndarray,
        I0: float,
        I1: np.ndarray,
        rho0: float,
        rho1: np.ndarray,
    ):
        """
        Initialize the affine model with the given parameters.
        Args:
            K0 (R^n): the constant term in the drift of the state variable
            K1 (R^{n x n}): the linear term in the drift of the state variable
            H0 (R^{n x n}): the constant term in the diffusion of the state variable
            H1 (R^{n x n x n}): the linear term in the diffusion
            I0 (R): the constant term in the jump intensity
            I1 (R^n): the linear term in the jump intensity
            rho0 (R): the constant term in the short rate
            rho1 (R^n): the linear term in the short rate
        """
        self.K0 = K0
        self.K1 = K1
        self.H0 = H0
        self.H1 = H1
        self.I0 = I0
        self.I1 = I1
        self.rho0 = rho0
        self.rho1 = rho1

    def Mu(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the drift of the state variable at a given state x.
        Args:
            x (R^n): the current state of the system
        Returns:
            R^n: the drift of the state variable
        """
        return self.K0 + self.K1 @ x

    def inst_cov_mat(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the instantaneous covariance matrix of the state variable at a given state x.
        Args:
            x (R^n): the current state of the system
        Returns:
            R^{n x n}: the instantaneous covariance matrix of the state variable
        """
        return self.H0 + np.einsum("ijk,k->ij", self.H1, x)

    def Lambda(self, x: np.ndarray) -> float:
        """
        Compute the jump intensity at a given state x.
        Args:
            x (R^n): the current state of the system
        Returns:
            R: the jump intensity lambda(x) = I0 + I1 · x
        """
        return self.I0 + np.dot(self.I1, x)

    def R(self, x: np.ndarray) -> float:
        """
        Compute the short rate at a given state x.
        Args:
            x (R^n): the current state of the system
        Returns:
            R: the short rate R(x) = rho0 + rho1 · x
        """
        return self.rho0 + np.dot(self.rho1, x)

    def characteristic_function(self, u, T, x0):
        """
        Returns E[exp(i*u*Y_T) | X_0=x0].
        """
        raise NotImplementedError(
            "Characteristic function not implemented for this model."
        )


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

    def characteristic_function(self, u, t, T, Xt):
        Yt = Xt[0]
        Vt = Xt[1]

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

        return np.exp(1j * u * Yt) * np.exp(A + B * Vt)


class HawkesModel(AffineModel):
    """
    Hawkes process model for jump intensity.
    """

    def __init__(self, r, q, gamma, sigma, delta, lambdabar, epsilon):
        """
        Initialize the Hawkes model with the given parameters.
        """
        mubar = gamma / (1 + gamma) - 1
        rho0 = r
        rho1 = np.array([0.0, 0.0])

        K0 = np.array([r - q - 0.5 * sigma**2, delta * lambdabar])
        K1 = np.array([[0.0, -mubar], [0.0, -delta]])

        I0 = 0.0
        I1 = np.array([0.0, 1.0])

        H0 = np.array([[sigma**2, 0.0], [0.0, 0.0]])
        H1 = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        super().__init__(K0, K1, H0, H1, I0, I1, rho0, rho1)

    def characteristic_function(self, u, T, x0):
        """
        Returns E[exp(i*u*Y_T) | X_0=x0].
        """
        pass
