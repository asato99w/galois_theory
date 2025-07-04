"""
ガロア理論ライブラリ

このライブラリは、5次方程式の解の存在判定をガロア理論を用いて行うためのツールを提供します。
"""

from .ring import Ring, RingElement, IntegerRing, RationalRing
from .field import Field, FieldElement, RationalField, FiniteField
from .polynomials import Polynomial, PolynomialRing, PolynomialElement, PolynomialException

__version__ = "0.1.0"
__all__ = [
    "Ring", "RingElement", "IntegerRing", "RationalRing",
    "Field", "FieldElement", "RationalField", "FiniteField",
    "Polynomial", "PolynomialRing", "PolynomialElement", "PolynomialException"
]
