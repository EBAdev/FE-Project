import numpy as np
from Models.AffineModel import AffineModel
from scipy.integrate import solve_ivp

PARAMETER_NAMES = [
    "gamma",
    "sigma",
    "delta",
    "lambdabar",
    "epsilon",
]


class HawkesModel(AffineModel):
    """
    Affine Hawkes model with exponentially distributed negative
    log-price jumps and self-exciting jump intensity.

    State:
        Y_t = [X_t, lambda_t]

    where:
        X_t       : log-price
        lambda_t  : Hawkes intensity

    Jump distribution:
        J ~ -Exp(gamma)

    Intensity jump:
        Delta lambda = epsilon
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

        # Compensator for the log-price jumps
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

        # No standalone constant killing rate for the
        # characteristic function of the log-price.
        rho0 = 0.0
        rho1 = np.zeros(2)

        super().__init__(K0, K1, H0, H1, I0, I1, rho0, rho1)

    def jump_transform(self, beta):
        return np.exp(self.epsilon * beta[1]) * self.gamma / (self.gamma + beta[0])

    def _riccati_system(self, tau, y, n):
        """
        Vectorized Riccati equations for n Fourier arguments.

        y contains:
            alpha[0:n]
            beta_1[0:n]
            beta_2[0:n]
        """
        y = y.reshape(3, n)

        # alpha = y[0]
        beta = y[1:]

        theta = self.jump_transform(beta)

        dbeta = self.K1.T @ beta + self.I1[:, None] * (theta - 1.0)

        dalpha = self.K0 @ beta + 0.5 * self.sigma**2 * beta[0] ** 2

        return np.concatenate((dalpha, dbeta[0], dbeta[1]))

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

        # Terminal conditions at tau = 0
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


if __name__ == "__main__":
    # Test the characteristic function of the Hawkes model
    model = HawkesModel(
        r=0.04,
        q=0.0,
        gamma=1.0,
        sigma=0.15,
        delta=1.1,
        lambdabar=0.5,
        epsilon=0.1,
    )

    Yt = np.array([np.log(100.0), 0.5])

    # Test scalar input
    phi_scalar = model.characteristic_function(0.5, 0.0, 1.0, Yt)

    # Test vector input
    u = np.linspace(-50, 50, 100)
    phi_vector = model.characteristic_function(u, 0.0, 1.0, Yt)

    # Characteristic function at zero
    phi_zero = model.characteristic_function(0.0, 0.0, 1.0, Yt)

    print("Scalar:", phi_scalar)
    print("Vector shape:", phi_vector.shape)
    print("Phi(0):", phi_zero)

    u = np.linspace(0.1, 50, 100)

    phi_pos = model.characteristic_function(u, 0.0, 1.0, Yt)
    phi_neg = model.characteristic_function(-u, 0.0, 1.0, Yt)

    print(np.max(np.abs(phi_neg - np.conj(phi_pos))))
    print(np.max(np.abs(phi_vector)))
