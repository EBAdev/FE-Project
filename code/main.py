import numpy as np
from Fourier import FourierPricer
from Models import SVJJModel

# Market inputs
S0 = 100.0
K = 100.0
T = 1.0
v0 = 0.04

# SVJJ parameters (illustrative)
model = SVJJModel(
    r=0.04,
    q=0.0,
    lmbda=0.1,
    rho=-0.7,
    vbar=0.04,
    kappa_v=2.0,
    sigma_v=0.3,
    mu_s=-0.1,
    sigma=0.15,
    rho_j=0.1,
    mu_v=0.2,
)

# Initial affine state: (log(S0), V0)
Xt = np.array([np.log(S0), v0])

option = FourierPricer(
    model=model,
    S0=S0,
    K=K,
    T=T,
    Xt=Xt,
)

print("Lewis:", option.price(method="lewis"))
print("Carr-Madan:", option.price(method="carr_madan"))
print("Probability:", option.price(method="probability"))
