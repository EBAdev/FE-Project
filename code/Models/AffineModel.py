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
