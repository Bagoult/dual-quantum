"""
Implements dual-complex numbers.

Author: Nathan Houyet
"""

import numpy as np


CX = PolynomialRing(CC, 'X')
X = CX.gen()
DC = CX.quotient('X*X', 'ϵ')  # Dual-Complex commutative ring
ϵ = DC.gen()  # Dual imaginary unit


def real(w: DC) -> CC:
    """
    Gets the **real** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `a`
    """
    z, t = w
    return real(z)


def comp(w: DC) -> CC:
    """
    Gets the **complex** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `b`
    """
    z, t = w
    return imag(z)


def dual(w: DC) -> CC:
    """
    Gets the **dual** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `c`
    """
    z, t = w
    return real(t)


def dual_comp(w: DC) -> CC:
    """
    Gets the **dual-complex** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c = d*i`

        Returns `b`
    """
    z, t = w
    return imag(t)


@np.vectorize
def cnj(w: DC) -> DC:
    z, t = w
    return conjugate(z) + conjugate(t)*ϵ


def sqrt(w):
    z, t = DC(w)
    return z**.5 + ϵ * t / (2 * z**.5)


def dagger(A):
    return np.transpose(cnj(A))


def inner(ψ, ϕ):
    return dagger(ψ) @ ϕ


def norm(ψ):
    return sqrt(inner(ψ, ψ))


def measure_prob(M, ψ):
    """
    Probability of outcome associated with measurement operator M to happen
    when measuring ψ
    """
    return inner(M @ ψ, M @ ψ)


def completeness(Ms):
    """
    Returns the sum of M^\dagger M for M in Ms.
    Used to check if a list of matrix respects completeness relation.
    """
    return sum(dagger(M) @ M for M in Ms)
