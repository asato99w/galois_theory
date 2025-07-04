"""
ガロア理論ライブラリ

このパッケージは、ガロア理論の基本的な概念を実装します。
環、体、多項式環、体拡大、ガロア群などの数学的構造を提供します。
"""

from .ring import IntegerRing, RingElement
from .field import RationalField, FiniteField, FieldElement
from .polynomials import Polynomial, PolynomialRing, PolynomialElement
from .field_extensions import (
    FieldExtension, SimpleExtension, ExtensionElement,
    AlgebraicElement, MinimalPolynomial, SplittingField,
    FieldExtensionException
)

__version__ = "0.1.0"
__all__ = [
    # Ring classes
    "IntegerRing", "RingElement",
    
    # Field classes
    "RationalField", "FiniteField", "FieldElement",
    
    # Polynomial classes
    "Polynomial", "PolynomialRing", "PolynomialElement",
    
    # Field extension classes
    "FieldExtension", "SimpleExtension", "ExtensionElement",
    "AlgebraicElement", "MinimalPolynomial", "SplittingField",
    "FieldExtensionException",
]
