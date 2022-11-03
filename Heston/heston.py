import numpy as np
from scipy.integrate import quad


def char_function(s0, v0, vbar, kappa, zeta, r, rho, T, w):
    """
    Determines characteristic function of log(S(T))
    Reference: Crisóstomo, 2014
    """
    alpha = -((w**2) / 2.0) - ((1j * w) / 2.0)
    beta = kappa - rho * zeta * 1j * w
    gamma = (zeta**2) / 2.0
    h = np.sqrt(beta**2 - 4 * alpha * gamma)
    rplus = (beta + h) / (zeta**2)
    rmin = (beta - h) / (zeta**2)
    g = rmin / rplus
    c = kappa * (
        rmin * T - (2 / zeta**2) * np.log((1 - g * np.exp(-h * T)) / (1 - g))
    )
    d = rmin * ((1 - np.exp(-h * T)) / (1 - g * np.exp(-h * T)))
    return np.exp(c * vbar + d * v0 + 1j * w * np.log(s0 * np.exp(r * T)))


def heston_euro_call(s0, K, T, v0, vbar, kappa, zeta, r, rho):
    """
    Calculates the price of a European call option for
    Heston model analytically using integration
    """

    def cf(w):
        return char_function(s0, v0, vbar, kappa, zeta, r, rho, T, w)

    def integrand_1(w):
        return np.real((np.exp(-1j * w * np.log(K)) * cf(w - 1j)) / (1j * w * cf(-1j)))

    integral_1 = quad(integrand_1, 0, np.inf)
    pi_1 = 0.5 + integral_1[0] / np.pi

    def integrand_2(w):
        return np.real((np.exp(-1j * w * np.log(K)) * cf(w)) / (1j * w))

    integral_2 = quad(integrand_2, 0, np.inf)
    pi_2 = 0.5 + integral_2[0] / np.pi
    return s0 * pi_1 - K * np.exp(-r * T) * pi_2


def heston_euro_call_param(param, maturity, moneyness):
    """
    Calculates the price of a European call option for
    Heston model analytically using integration
    """
    strike = moneyness * param["s0"]

    def cf(w):
        return char_function(
            param["s0"],
            param["v0"],
            param["vbar"],
            param["kappa"],
            param["zeta"],
            param["r"],
            param["rho"],
            maturity,
            w,
        )

    def integrand_1(w):
        return np.real(
            (np.exp(-1j * w * np.log(strike)) * cf(w - 1j)) / (1j * w * cf(-1j))
        )

    integral_1 = quad(integrand_1, 0, np.inf)
    pi_1 = 0.5 + integral_1[0] / np.pi

    def integrand_2(w):
        return np.real((np.exp(-1j * w * np.log(strike)) * cf(w)) / (1j * w))

    integral_2 = quad(integrand_2, 0, np.inf)
    pi_2 = 0.5 + integral_2[0] / np.pi
    return param["s0"] * pi_1 - strike * np.exp(-param["r"] * maturity) * pi_2
