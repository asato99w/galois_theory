"""
環（Ring）オブジェクトのテスト

このファイルは、環の基本的な性質と操作をテストします。
ガロア理論における環の概念を実装するためのテストファーストアプローチです。
"""

import pytest
from fractions import Fraction
from typing import Any, Union

from galois_theory.ring import Ring, RingElement, IntegerRing, RationalRing


class TestRingElement:
    """環の要素のテスト"""

    def test_ring_element_creation(self) -> None:
        """環の要素の作成テスト"""
        ring = IntegerRing()
        elem = RingElement(5, ring)
        
        assert elem.value == 5
        assert elem.ring == ring

    def test_ring_element_equality(self) -> None:
        """環の要素の等価性テスト"""
        ring = IntegerRing()
        elem1 = RingElement(5, ring)
        elem2 = RingElement(5, ring)
        elem3 = RingElement(3, ring)
        
        assert elem1 == elem2
        assert elem1 != elem3

    def test_ring_element_addition(self) -> None:
        """環の要素の加法テスト"""
        ring = IntegerRing()
        elem1 = RingElement(3, ring)
        elem2 = RingElement(5, ring)
        
        result = elem1 + elem2
        assert result.value == 8
        assert result.ring == ring

    def test_ring_element_multiplication(self) -> None:
        """環の要素の乗法テスト"""
        ring = IntegerRing()
        elem1 = RingElement(3, ring)
        elem2 = RingElement(5, ring)
        
        result = elem1 * elem2
        assert result.value == 15
        assert result.ring == ring

    def test_ring_element_additive_inverse(self) -> None:
        """環の要素の加法逆元テスト"""
        ring = IntegerRing()
        elem = RingElement(5, ring)
        
        inverse = -elem
        assert inverse.value == -5
        assert inverse.ring == ring

    def test_ring_element_subtraction(self) -> None:
        """環の要素の減法テスト"""
        ring = IntegerRing()
        elem1 = RingElement(8, ring)
        elem2 = RingElement(3, ring)
        
        result = elem1 - elem2
        assert result.value == 5
        assert result.ring == ring

    def test_ring_element_different_rings_error(self) -> None:
        """異なる環の要素同士の演算エラーテスト"""
        int_ring = IntegerRing()
        rat_ring = RationalRing()
        
        elem1 = RingElement(5, int_ring)
        elem2 = RingElement(Fraction(3, 2), rat_ring)
        
        with pytest.raises(ValueError, match="異なる環の要素同士の演算はできません"):
            elem1 + elem2


class TestRing:
    """環の基本的な性質のテスト"""

    def test_ring_zero_element(self) -> None:
        """環の零元テスト"""
        ring = IntegerRing()
        zero = ring.zero()
        
        assert zero.value == 0
        assert zero.ring == ring

    def test_ring_one_element(self) -> None:
        """環の単位元テスト"""
        ring = IntegerRing()
        one = ring.one()
        
        assert one.value == 1
        assert one.ring == ring

    def test_ring_contains_element(self) -> None:
        """環が要素を含むかのテスト"""
        ring = IntegerRing()
        
        assert ring.contains(5)
        assert ring.contains(-3)
        assert ring.contains(0)

    def test_ring_create_element(self) -> None:
        """環の要素作成テスト"""
        ring = IntegerRing()
        elem = ring.element(7)
        
        assert isinstance(elem, RingElement)
        assert elem.value == 7
        assert elem.ring == ring


class TestIntegerRing:
    """整数環のテスト"""

    def test_integer_ring_creation(self) -> None:
        """整数環の作成テスト"""
        ring = IntegerRing()
        assert ring.name == "整数環 Z"

    def test_integer_ring_contains(self) -> None:
        """整数環の要素判定テスト"""
        ring = IntegerRing()
        
        assert ring.contains(5)
        assert ring.contains(-3)
        assert ring.contains(0)
        assert not ring.contains(Fraction(1, 2))
        assert not ring.contains(3.14)

    def test_integer_ring_operations(self) -> None:
        """整数環の演算テスト"""
        ring = IntegerRing()
        
        a = ring.element(6)
        b = ring.element(4)
        
        # 加法
        assert (a + b).value == 10
        
        # 乗法
        assert (a * b).value == 24
        
        # 減法
        assert (a - b).value == 2

    def test_integer_ring_additive_identity(self) -> None:
        """整数環の加法単位元テスト"""
        ring = IntegerRing()
        zero = ring.zero()
        elem = ring.element(5)
        
        assert (elem + zero) == elem
        assert (zero + elem) == elem

    def test_integer_ring_multiplicative_identity(self) -> None:
        """整数環の乗法単位元テスト"""
        ring = IntegerRing()
        one = ring.one()
        elem = ring.element(5)
        
        assert (elem * one) == elem
        assert (one * elem) == elem

    def test_integer_ring_additive_inverse(self) -> None:
        """整数環の加法逆元テスト"""
        ring = IntegerRing()
        elem = ring.element(5)
        zero = ring.zero()
        
        assert (elem + (-elem)) == zero
        assert ((-elem) + elem) == zero


class TestRationalRing:
    """有理数環のテスト"""

    def test_rational_ring_creation(self) -> None:
        """有理数環の作成テスト"""
        ring = RationalRing()
        assert ring.name == "有理数環 Q"

    def test_rational_ring_contains(self) -> None:
        """有理数環の要素判定テスト"""
        ring = RationalRing()
        
        assert ring.contains(Fraction(1, 2))
        assert ring.contains(Fraction(3, 1))
        assert ring.contains(Fraction(-5, 7))
        assert ring.contains(5)  # 整数も有理数
        assert not ring.contains(3.14159)  # 浮動小数点数は除外

    def test_rational_ring_operations(self) -> None:
        """有理数環の演算テスト"""
        ring = RationalRing()
        
        a = ring.element(Fraction(2, 3))
        b = ring.element(Fraction(1, 4))
        
        # 加法: 2/3 + 1/4 = 8/12 + 3/12 = 11/12
        result_add = a + b
        assert result_add.value == Fraction(11, 12)
        
        # 乗法: 2/3 * 1/4 = 2/12 = 1/6
        result_mul = a * b
        assert result_mul.value == Fraction(1, 6)
        
        # 減法: 2/3 - 1/4 = 8/12 - 3/12 = 5/12
        result_sub = a - b
        assert result_sub.value == Fraction(5, 12)

    def test_rational_ring_integer_coercion(self) -> None:
        """有理数環での整数の強制変換テスト"""
        ring = RationalRing()
        
        # 整数を有理数として扱う
        elem = ring.element(5)
        assert elem.value == Fraction(5, 1)
        
        # 整数と有理数の演算
        int_elem = ring.element(3)
        frac_elem = ring.element(Fraction(1, 2))
        
        result = int_elem + frac_elem
        assert result.value == Fraction(7, 2)


class TestRingAxioms:
    """環の公理のテスト"""

    @pytest.fixture
    def integer_ring(self) -> IntegerRing:
        """整数環のフィクスチャ"""
        return IntegerRing()

    def test_additive_associativity(self, integer_ring: IntegerRing) -> None:
        """加法の結合律テスト: (a + b) + c = a + (b + c)"""
        a = integer_ring.element(2)
        b = integer_ring.element(3)
        c = integer_ring.element(5)
        
        left = (a + b) + c
        right = a + (b + c)
        
        assert left == right

    def test_additive_commutativity(self, integer_ring: IntegerRing) -> None:
        """加法の交換律テスト: a + b = b + a"""
        a = integer_ring.element(7)
        b = integer_ring.element(11)
        
        assert (a + b) == (b + a)

    def test_multiplicative_associativity(self, integer_ring: IntegerRing) -> None:
        """乗法の結合律テスト: (a * b) * c = a * (b * c)"""
        a = integer_ring.element(2)
        b = integer_ring.element(3)
        c = integer_ring.element(5)
        
        left = (a * b) * c
        right = a * (b * c)
        
        assert left == right

    def test_distributivity(self, integer_ring: IntegerRing) -> None:
        """分配律テスト: a * (b + c) = a * b + a * c"""
        a = integer_ring.element(3)
        b = integer_ring.element(4)
        c = integer_ring.element(5)
        
        left = a * (b + c)
        right = (a * b) + (a * c)
        
        assert left == right

    def test_additive_identity_existence(self, integer_ring: IntegerRing) -> None:
        """加法単位元の存在テスト"""
        zero = integer_ring.zero()
        elem = integer_ring.element(42)
        
        assert (elem + zero) == elem
        assert (zero + elem) == elem

    def test_multiplicative_identity_existence(self, integer_ring: IntegerRing) -> None:
        """乗法単位元の存在テスト"""
        one = integer_ring.one()
        elem = integer_ring.element(42)
        
        assert (elem * one) == elem
        assert (one * elem) == elem

    def test_additive_inverse_existence(self, integer_ring: IntegerRing) -> None:
        """加法逆元の存在テスト"""
        elem = integer_ring.element(42)
        zero = integer_ring.zero()
        
        inverse = -elem
        assert (elem + inverse) == zero
        assert (inverse + elem) == zero 