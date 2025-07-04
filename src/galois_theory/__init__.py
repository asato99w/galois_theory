"""
ガロア理論ライブラリ

このライブラリは、ガロア理論の主要概念を実装します：
- 環（Ring）と体（Field）
- 多項式環（PolynomialRing）
- 体の拡大（Field Extensions）
- 群論（Group Theory）
- ガロア群（Galois Groups）

使用例:
>>> from galois_theory import RationalField, PolynomialRing, SimpleExtension
>>> Q = RationalField()
>>> poly_ring = PolynomialRing(Q, "x")
>>> x_squared_minus_2 = poly_ring.from_coefficients([-2, 0, 1])
>>> Q_sqrt2 = SimpleExtension(Q, x_squared_minus_2, "sqrt2")
>>> print(Q_sqrt2.degree())
2
"""

from .ring import Ring, RingElement, IntegerRing
from .field import Field, FieldElement, RationalField, FiniteField
from .polynomials import Polynomial, PolynomialRing, PolynomialElement, PolynomialException
from .field_extensions import (
    FieldExtension, SimpleExtension, ExtensionElement, SplittingField,
    MinimalPolynomial, AlgebraicElement, FieldExtensionException
)
from .group_theory import (
    Group, GroupElement, GroupException,
    CyclicGroup, CyclicGroupElement,
    SymmetricGroup, Permutation,
    DihedralGroup, DihedralElement,
    GaloisGroup, GaloisGroupElement,
    GroupAction, GroupHomomorphism, GroupIsomorphism,
    Subgroup, FiniteGroup, FieldAutomorphism
)

__version__ = "0.1.0"
__author__ = "Galois Theory Project"

__all__ = [
    # Ring and Field
    "Ring", "RingElement", "IntegerRing",
    "Field", "FieldElement", "RationalField", "FiniteField",
    
    # Polynomials
    "Polynomial", "PolynomialRing", "PolynomialElement", "PolynomialException",
    
    # Field Extensions
    "FieldExtension", "SimpleExtension", "ExtensionElement", "SplittingField",
    "MinimalPolynomial", "AlgebraicElement", "FieldExtensionException",
    
    # Group Theory
    "Group", "GroupElement", "GroupException",
    "CyclicGroup", "CyclicGroupElement",
    "SymmetricGroup", "Permutation",
    "DihedralGroup", "DihedralElement",
    "GaloisGroup", "GaloisGroupElement",
    "GroupAction", "GroupHomomorphism", "GroupIsomorphism",
    "Subgroup", "FiniteGroup", "FieldAutomorphism",
]
