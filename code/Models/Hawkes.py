import numpy as np
from scipy.integrate import solve_ivp

from Models.AffineModel import AffineModel

PARAMETER_NAMES = [
    "gamma",
    "sigma",
    "delta",
    "lambdabar",
    "epsilon",
]


class HawkesModel(AffineModel):
    """
    Affine Hawkes model with negatively exponentially distributed
    log-price jumps and self-exciting jump intensity.

    State vector:
        X_t = [Y_t, lambda_t]

    Jump vector:
        Delta X_t = [J, epsilon * J]

    where J is negative exponentially distributed with rate gamma.
    """

    def __init__(
        self,
        r,
        q,
        gamma,
        sigma,
        delta,
        lambdabar,
        epsilon,
    ):
        self.r = r
        self.q = q
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.lambdabar = lambdabar
        self.epsilon = epsilon

        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if delta <= 0:
            raise ValueError("delta must be positive.")
        if lambdabar <= 0:
            raise ValueError("lambdabar must be positive.")
        if epsilon >= 0:
            raise ValueError("epsilon must be negative.")

        # Compensator for the log-price jumps:
        # J ~ -Exp(gamma), hence E[exp(J) - 1] = -1/(1+gamma).
        self.mubar = -1.0 / (1.0 + gamma)

        # Affine drift coefficients
        K0 = np.array(
            [
                r - q - 0.5 * sigma**2,
                delta * lambdabar,
            ]
        )

        K1 = np.array(
            [
                [0.0, -self.mubar],
                [0.0, -delta],
            ]
        )

        # Jump intensity: lambda_t
        I0 = 0.0
        I1 = np.array([0.0, 1.0])

        # Diffusion covariance
        H0 = np.array(
            [
                [sigma**2, 0.0],
                [0.0, 0.0],
            ]
        )

        H1 = np.zeros((2, 2, 2))

        # No killing rate for the characteristic function
        rho0 = 0.0
        rho1 = np.zeros(2)

        super().__init__(K0, K1, H0, H1, I0, I1, rho0, rho1)

    def jump_transform(self, beta):
        """
        Jump transform:

            theta(beta) = E[exp(beta_1 * J + beta_2 * epsilon * J)]

                       = gamma / (gamma + beta_1 + epsilon * beta_2)

        where J ~ -Exp(gamma).

        beta can have shape (2,) or (..., 2).
        """
        beta = np.asarray(beta, dtype=complex)

        denominator = self.gamma + beta[..., 0] + self.epsilon * beta[..., 1]

        return self.gamma / denominator

    def _riccati_system(self, tau, y, n):
        """
        Riccati system in time-to-maturity coordinates.

        State vector y is arranged as:

            y = [alpha, beta_Y, beta_lambda]

        Each component contains n Fourier arguments.
        """
        alpha, beta1, beta2 = y.reshape(3, n)

        beta = np.stack((beta1, beta2), axis=-1)

        theta = self.jump_transform(beta)

        # d beta_Y / d tau = 0
        dbeta1 = np.zeros(n, dtype=complex)

        # d beta_lambda / d tau
        dbeta2 = -self.mubar * beta1 - self.delta * beta2 + theta - 1.0

        # d alpha / d tau
        dalpha = (
            (self.r - self.q - 0.5 * self.sigma**2) * beta1
            + self.delta * self.lambdabar * beta2
            + 0.5 * self.sigma**2 * beta1**2
        )

        return np.concatenate((dalpha, dbeta1, dbeta2))

    def characteristic_function(
        self,
        u,
        t,
        T,
        Xt,
        rtol=1e-8,
        atol=1e-10,
    ):
        """
        Characteristic function of log-price:

            E[exp(i*u*Y_T) | X_t]

        Supports scalar and vector-valued Fourier arguments.

        Xt = [Yt, lambdat].
        """
        u = np.asarray(u, dtype=complex)
        scalar_input = u.ndim == 0
        u = np.atleast_1d(u).ravel()

        Xt = np.asarray(Xt, dtype=float)

        if Xt.shape != (2,):
            raise ValueError("Xt must contain [Yt, lambdat].")

        if T < t:
            raise ValueError("T must be greater than or equal to t.")

        if T == t:
            result = np.exp(1j * u * Xt[0])
            return result[0] if scalar_input else result

        n = len(u)

        # Initial conditions at tau = 0:
        # alpha(0) = 0
        # beta_Y(0) = i*u
        # beta_lambda(0) = 0
        y0 = np.concatenate(
            (
                np.zeros(n, dtype=complex),
                1j * u,
                np.zeros(n, dtype=complex),
            )
        )

        solution = solve_ivp(
            lambda tau, y: self._riccati_system(tau, y, n),
            t_span=(0.0, T - t),
            y0=y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )

        if not solution.success:
            raise RuntimeError(f"Riccati ODE failed: {solution.message}")

        alpha, beta1, beta2 = solution.y[:, -1].reshape(3, n)

        result = np.exp(alpha + beta1 * Xt[0] + beta2 * Xt[1])

        return result[0] if scalar_input else result

    def stationary_lambda(self):
        """
        Stationary jump intensity:

            lambda_bar = delta * lambdabar / (delta - epsilon * gamma)
        """
        return self.delta * self.lambdabar / (self.delta - self.epsilon * self.gamma)


if __name__ == "__main__":
    model = HawkesModel(
        r=0.04,
        q=0.0,
        gamma=1.0,
        sigma=0.15,
        delta=1.1,
        lambdabar=0.5,
        epsilon=-0.1,
    )

    Xt = np.array([np.log(100.0), 0.5])

    # Scalar input
    phi_scalar = model.characteristic_function(0.5, 0.0, 1.0, Xt)

    # Vector input
    u = np.linspace(-50, 50, 100)
    phi_vector = model.characteristic_function(u, 0.0, 1.0, Xt)

    # Characteristic function at zero
    phi_zero = model.characteristic_function(0.0, 0.0, 1.0, Xt)

    print("Scalar:", phi_scalar)
    print("Vector shape:", phi_vector.shape)
    print("Phi(0):", phi_zero)

    # Conjugate symmetry
    u = np.linspace(0.1, 50, 100)

    phi_pos = model.characteristic_function(u, 0.0, 1.0, Xt)
    phi_neg = model.characteristic_function(-u, 0.0, 1.0, Xt)

    print("Conjugate symmetry error:", np.max(np.abs(phi_neg - np.conj(phi_pos))))

    print("Maximum characteristic function magnitude:", np.max(np.abs(phi_vector)))
