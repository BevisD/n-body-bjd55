import numpy as np
from scipy.special import binom, comb
from timeit import timeit
import matplotlib.pyplot as plt
np.random.seed(0)

P = 35
Z = -0.5
A = np.random.random(P + 1)


def M2M_old(p, z, a):
    b = np.zeros_like(a)
    b[0] = a[0]
    for L in range(1, p+1):
        b[L] = - a[0] * z**L/L
        for k in range(1, L+1):
            b[L] += a[k] * z**(L-k) * comb(L-1, k-1, exact=True)
    return b


def M2M_new(p, z, a):
    b = np.zeros_like(a)
    b[0] = a[0]
    L = np.arange(1, p+1, dtype=int)
    k = np.arange(1, p+1, dtype=int)

    b[1:] = -a[0] * z**L/L
    binom_terms = binom(L-1, k[:, np.newaxis]-1)

    a_terms = a[1:, np.newaxis]
    z_terms = z**(L - k[:, np.newaxis])
    b[1:] += np.sum(binom_terms * a_terms * z_terms, axis=0)
    return b


def M2L_old(p, z, b):
    c = np.zeros_like(b)
    c[0] = b[0] * np.log(-z)
    for k in range(1, p+1):
        c[0] += (-1)**k * b[k]/z**k

    for L in range(1, p+1):
        c[L] = -b[0]/(L * z**L)
        for k in range(1, p+1):
            c[L] += 1/z**L * (-1)**k * b[k]/z**k * binom(L+k-1, k-1)

    return c

def M2L_new(p, z, b):
    c = np.zeros_like(b)
    c[0] = b[0] * np.log(-z)
    k = np.arange(1, p+1)
    L = np.arange(1, p+1)
    signs = np.ones(p)
    signs[::2] = -1

    c[0] += np.sum(signs * b[1:] / z**k)

    binom_factors = binom(L[:, np.newaxis] + k - 1, k - 1)
    c[1:] = -b[0] / (L * z**L)
    c[1:] += 1/z**L * np.sum(binom_factors * signs * b[1:] / z**k, axis=1)
    return c


def L2L_old(p, z, c):
    d = np.zeros_like(c)
    for L in range(0, p+1):
        for k in range(L, p+1):
            d[L] += c[k] * binom(k, L) * z**(k-L)
    return d


def L2L_new(p, z, c):
    L = np.arange(p+1)
    k = np.arange(p+1)
    d = np.sum(c * binom(k, L[:, np.newaxis]) *
               z ** (k-L[:, np.newaxis]), axis=1)
    return d


assert np.all(np.isclose(L2L_old(P, Z, A), L2L_new(P, Z, A)))

time_old = timeit("L2L_old(P, Z, A)", globals=globals(), number=1000)
time_new = timeit("L2L_new(P, Z, A)", globals=globals(), number=1000)

print(f"OLD: {time_old:.3f}")
print(f"NEW: {time_new:.3f}")
