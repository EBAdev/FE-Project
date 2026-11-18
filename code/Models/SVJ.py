from Models.SVJJ import SVJJModel

PARAMETER_NAMES = [
    "lmbda",
    "rho",
    "vbar",
    "kappa_v",
    "sigma_v",
    "mu_s",
    "sigma",
]


class SVJModel(SVJJModel):
    """Stochastic volatility with jumps (SVJ) model.

    This is a special case of the SVJJ model with no jump in volatility.
    """

    def __init__(self, r, q, lmbda, rho, vbar, kappa_v, sigma_v, mu_s, sigma):
        """
        Initialize the SVJ model with the given parameters.

        """
        super().__init__(
            r, q, lmbda, rho, vbar, kappa_v, sigma_v, mu_s, sigma, 0.0, 0.0
        )

        self.ParameterNames = [
            "lmbda",
            "rho",
            "vbar",
            "kappa_v",
            "sigma_v",
            "mu_s",
            "sigma",
        ]
