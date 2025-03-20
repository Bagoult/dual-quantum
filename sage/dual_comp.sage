"""
Implements dual-complex numbers.

Author: Nathan Houyet
"""

import numpy as np
from typing import Union, Iterable


CX = PolynomialRing(CC, 'X')
X = CX.gen()
DC = CX.quotient('X*X', 'ϵ')  # Dual-Complex commutative ring
ϵ = eps = epsilon = DC.gen()  # Dual imaginary unit


@np.vectorize
def Re(w: Union[DC, CC]) -> CC:
    """
    Gets the **real** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `a`
    """
    z, t = DC(w)
    return real(z)


@np.vectorize
def ImI(w: Union[DC, CC]) -> CC:
    """
    Gets the **complex, non-dual** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `b`
    """
    z, t = DC(w)
    return imag(z)


@np.vectorize
def ImE(w: Union[DC, CC]) -> CC:
    """
    Gets the **dual, non-complex** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c + d*i`

        Returns `c`
    """
    z, t = DC(w)
    return real(t)


@np.vectorize
def ImIE(w: Union[DC, CC]) -> CC:
    """
    Gets the **dual-complex** part of a dual-complex number.

        Parameters:
            w (DC): a dual-complex of the form `w = z + t*ϵ`
                where `z = a + b*i`
                      `t = c = d*i`

        Returns `d`
    """
    z, t = DC(w)
    return imag(t)


@np.vectorize
def Du(w: Union[DC, CC]) -> CC:
    """
    Dual part of w.
    Let w = z + tϵ. Then Du(w) = t.
    """
    z, t = DC(w)
    return t


@np.vectorize
def Co(w: Union[DC, CC]) -> CC:
    """
    Non-dual part of w.
    Let w = z + tϵ. Then Co(w) = z.
    """
    z, t = DC(w)
    return z


@np.vectorize
def is_dual(w: Union[DC, CC]) -> bool:
    """
    True iff w ∉ CC
    """
    z, t = DC(w)
    return t != 0


@np.vectorize
def cnj(w: Union[DC, CC]) -> DC:
    """
    Returns w*, the complex conjugate of a dual-complex of w.
    """
    z, t = DC(w)
    return conjugate(z) + conjugate(t)*ϵ


@np.vectorize
def sqrt(w: Union[DC, CC]) -> DC:
    """
    The square root of a dual-complex w is defined iff Co(w) != 0.
    """
    z, t = DC(w)
    return z**.5 + ϵ * t / (2 * z**.5)


def dagger(A):
    """
    Transpose and conjugate of A to get A^\\dagger.
    """
    return np.transpose(cnj(A))


def inner(ψ, ϕ):
    """
    The inner product of ψ and ϕ is defined as ∑_i ψ_i* ϕ_i.
    """
    return dagger(ψ) @ ϕ


def outer(ψ, ϕ):
    """
    Outer product of vectors ψ and ϕ.
    """
    return np.outer(ψ, dagger(ϕ))


def norm(ψ):
    """
    The norm of ψ is defined as the square root of the inner product of ψ and
    itself.
    """
    return sqrt(inner(ψ, ψ))


def measure_prob(M, ψ):
    """
    Probability of outcome associated with measurement operator M to happen
    when measuring ψ
    """
    return inner(M @ ψ, M @ ψ)


def completeness(*Ms):
    """
    Returns the sum of M^\\dagger M for M in Ms.
    Used to check if a list of matrix respects completeness relation.
    """
    return sum(dagger(M) @ M for M in Ms)


def evolution(U, ρ):
    """
    Let U represent the evolution of the system and ρ be a density operator
    representing the state of the said system before evolution.
    evolution(U, ρ) = U ρ U^\\dagger is the state of the system after
    evolution.
    """
    return U @ ρ @ dagger(U)


def measurement(ρ, *Ms):
    """
    Matrices Ms form a measurement operators collection applied to density
    operator ρ. measurement(ρ, *Ms) is the density operator after measurement.
    """
    return sum(M @ ρ @ dagger(M) for M in Ms)


if __name__ == "__main__":
    Psi = np.array([1 + 2 * ϵ, 1 - ϵ, 1 - ϵ])
    ψ = Psi / norm(Psi)
    one, two, three = (np.eye(1, 3, i) for i in range(3))
    A = outer(one, one)
    B = outer(two, two) + outer(three, three)
    M1 = A + ϵ * B
    M2 = B + ϵ * A

    # M1, M2 is a valid measurement operators collection
    assert np.all(completeness(M1, M2) - np.identity(3) == 0)

    p1 = measure_prob(M1, ψ)
    p2 = measure_prob(M2, ψ)
    ψ_1 = M1 @ ψ
    ψ_2 = M2 @ ψ

    ρ = outer(ψ, ψ)
    ρ_res = measurement(ρ, M1, M2)

    assert np.all(ρ_res - outer(ψ_1, ψ_1) - outer(ψ_2, ψ_2) == 0)
