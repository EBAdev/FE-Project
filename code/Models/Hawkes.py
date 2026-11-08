import numpy as np

from Models.AffineModel import AffineModel


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
