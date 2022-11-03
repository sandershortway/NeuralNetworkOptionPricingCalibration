import numpy as np
from scipy.stats import norm

# Calculates price of European call via Black-Scholes Formula
def euro_call(s0, K, T, r, sigma):
    d_1 = (np.log(s0 / K) + T * (r + (sigma**2) / 2)) / (sigma * np.sqrt(T))
    d_2 = (np.log(s0 / K) + T * (r - (sigma**2) / 2)) / (sigma * np.sqrt(T))
    return s0 * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
