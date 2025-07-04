"""
多項式環（Polynomial Ring）のテスト

このファイルは、多項式環の基本的な性質と操作をテストします。
ガロア理論における多項式環の概念を実装するためのテストファーストアプローチです。

多項式環 R[x] は、環 R 上の多項式からなる環です。
"""

import pytest
from fractions import Fraction
from typing import Any, List

from galois_theory.ring import IntegerRing, RationalRing
from galois_theory.field import RationalField
from galois_theory.polynomials import (
    Polynomial, PolynomialRing, 
    PolynomialElement, PolynomialException
)


class TestPolynomial:
    """多項式の基本的な性質のテスト"""

    def test_polynomial_creation_with_coefficients(self) -> None:
        """係数リストから多項式の作成テスト"""
        # f(x) = 3x^2 + 2x + 1 の場合、係数は [1, 2, 3] (定数項から高次へ)
        coeffs = [1, 2, 3]
        field = RationalField()
        poly = Polynomial(coeffs, field)
        
        assert poly.coefficients == [Fraction(1), Fraction(2), Fraction(3)]
        assert poly.degree() == 2
        assert poly.base_ring == field

    def test_polynomial_creation_empty_coefficients(self) -> None:
        """空の係数リストから零多項式作成テスト"""
        field = RationalField()
        poly = Polynomial([], field)
        
        assert poly.coefficients == [Fraction(0)]
        assert poly.degree() == 0
        assert poly.is_zero()

    def test_polynomial_creation_zero_coefficients(self) -> None:
        """零係数のみの多項式作成テスト"""
        field = RationalField()
        poly = Polynomial([0, 0, 0], field)
        
        assert poly.coefficients == [Fraction(0)]
        assert poly.degree() == 0
        assert poly.is_zero()

    def test_polynomial_leading_coefficient_stripping(self) -> None:
        """先頭の零係数の除去テスト"""
        field = RationalField()
        # [1, 2, 0, 0] -> [1, 2] (先頭の零を削除)
        poly = Polynomial([1, 2, 0, 0], field)
        
        assert poly.coefficients == [Fraction(1), Fraction(2)]
        assert poly.degree() == 1

    def test_polynomial_degree_calculation(self) -> None:
        """多項式の次数計算テスト"""
        field = RationalField()
        
        poly1 = Polynomial([1, 2, 3], field)  # 3x^2 + 2x + 1
        assert poly1.degree() == 2
        
        poly2 = Polynomial([5], field)  # 5
        assert poly2.degree() == 0
        
        poly3 = Polynomial([0], field)  # 0
        assert poly3.degree() == 0

    def test_polynomial_equality(self) -> None:
        """多項式の等価性テスト"""
        field = RationalField()
        poly1 = Polynomial([1, 2, 3], field)
        poly2 = Polynomial([1, 2, 3], field)
        poly3 = Polynomial([1, 2, 4], field)
        
        assert poly1 == poly2
        assert poly1 != poly3

    def test_polynomial_string_representation(self) -> None:
        """多項式の文字列表現テスト"""
        field = RationalField()
        
        # 3x^2 + 2x + 1
        poly1 = Polynomial([1, 2, 3], field)
        expected1 = "3*x^2 + 2*x + 1"
        assert str(poly1) == expected1
        
        # 5
        poly2 = Polynomial([5], field)
        assert str(poly2) == "5"
        
        # 0
        poly3 = Polynomial([0], field)
        assert str(poly3) == "0"
        
        # x^3 - 2x + 1 (x^2の係数が0)
        poly4 = Polynomial([1, -2, 0, 1], field)
        expected4 = "x^3 - 2*x + 1"
        assert str(poly4) == expected4

    def test_polynomial_evaluation(self) -> None:
        """多項式の値の計算テスト"""
        field = RationalField()
        # f(x) = x^2 + 2x + 1
        poly = Polynomial([1, 2, 1], field)
        
        # f(0) = 1
        assert poly.evaluate(field.element(0)) == field.element(1)
        
        # f(1) = 4
        assert poly.evaluate(field.element(1)) == field.element(4)
        
        # f(2) = 9
        assert poly.evaluate(field.element(2)) == field.element(9)

    def test_polynomial_is_zero(self) -> None:
        """零多項式判定テスト"""
        field = RationalField()
        
        zero_poly = Polynomial([0], field)
        assert zero_poly.is_zero()
        
        non_zero_poly = Polynomial([1, 0, 0], field)
        assert not non_zero_poly.is_zero()

    def test_polynomial_is_monic(self) -> None:
        """モニック多項式判定テスト"""
        field = RationalField()
        
        # x^2 + 2x + 1 (最高次係数が1)
        monic_poly = Polynomial([1, 2, 1], field)
        assert monic_poly.is_monic()
        
        # 2x^2 + 2x + 1 (最高次係数が2)
        non_monic_poly = Polynomial([1, 2, 2], field)
        assert not non_monic_poly.is_monic()


class TestPolynomialArithmetic:
    """多項式の算術演算のテスト"""

    def test_polynomial_addition(self) -> None:
        """多項式の加法テスト"""
        field = RationalField()
        
        # (x^2 + 2x + 1) + (x^2 + 3x + 2) = 2x^2 + 5x + 3
        poly1 = Polynomial([1, 2, 1], field)
        poly2 = Polynomial([2, 3, 1], field)
        result = poly1 + poly2
        
        expected = Polynomial([3, 5, 2], field)
        assert result == expected

    def test_polynomial_addition_different_degrees(self) -> None:
        """異なる次数の多項式の加法テスト"""
        field = RationalField()
        
        # (x^3 + x + 1) + (2x + 3) = x^3 + 3x + 4
        poly1 = Polynomial([1, 1, 0, 1], field)  # x^3 + x + 1
        poly2 = Polynomial([3, 2], field)        # 2x + 3
        result = poly1 + poly2
        
        expected = Polynomial([4, 3, 0, 1], field)
        assert result == expected

    def test_polynomial_subtraction(self) -> None:
        """多項式の減法テスト"""
        field = RationalField()
        
        # (2x^2 + 5x + 3) - (x^2 + 3x + 2) = x^2 + 2x + 1
        poly1 = Polynomial([3, 5, 2], field)
        poly2 = Polynomial([2, 3, 1], field)
        result = poly1 - poly2
        
        expected = Polynomial([1, 2, 1], field)
        assert result == expected

    def test_polynomial_scalar_multiplication(self) -> None:
        """多項式のスカラー倍テスト"""
        field = RationalField()
        
        # 3 * (x^2 + 2x + 1) = 3x^2 + 6x + 3
        poly = Polynomial([1, 2, 1], field)
        scalar = field.element(3)
        result = poly.scalar_multiply(scalar)
        
        expected = Polynomial([3, 6, 3], field)
        assert result == expected

    def test_polynomial_multiplication(self) -> None:
        """多項式の乗法テスト"""
        field = RationalField()
        
        # (x + 1) * (x + 2) = x^2 + 3x + 2
        poly1 = Polynomial([1, 1], field)  # x + 1
        poly2 = Polynomial([2, 1], field)  # x + 2
        result = poly1 * poly2
        
        expected = Polynomial([2, 3, 1], field)
        assert result == expected

    def test_polynomial_multiplication_complex(self) -> None:
        """複雑な多項式の乗法テスト"""
        field = RationalField()
        
        # (x^2 + x + 1) * (x + 1) = x^3 + 2x^2 + 2x + 1
        poly1 = Polynomial([1, 1, 1], field)  # x^2 + x + 1
        poly2 = Polynomial([1, 1], field)     # x + 1
        result = poly1 * poly2
        
        expected = Polynomial([1, 2, 2, 1], field)
        assert result == expected

    def test_polynomial_power(self) -> None:
        """多項式の冪乗テスト"""
        field = RationalField()
        
        # (x + 1)^2 = x^2 + 2x + 1
        poly = Polynomial([1, 1], field)  # x + 1
        result = poly ** 2
        
        expected = Polynomial([1, 2, 1], field)
        assert result == expected

    def test_polynomial_power_zero(self) -> None:
        """多項式の0乗テスト"""
        field = RationalField()
        
        poly = Polynomial([1, 2, 3], field)  # 3x^2 + 2x + 1
        result = poly ** 0
        
        expected = Polynomial([1], field)  # 1
        assert result == expected


class TestPolynomialDivision:
    """多項式の除法のテスト"""

    def test_polynomial_division_exact(self) -> None:
        """割り切れる多項式の除法テスト"""
        field = RationalField()
        
        # (x^2 + 3x + 2) ÷ (x + 1) = x + 2, remainder = 0
        # x^2 + 3x + 2 = (x + 1)(x + 2)
        dividend = Polynomial([2, 3, 1], field)  # x^2 + 3x + 2
        divisor = Polynomial([1, 1], field)      # x + 1
        
        quotient, remainder = dividend.divide(divisor)
        
        expected_quotient = Polynomial([2, 1], field)  # x + 2
        expected_remainder = Polynomial([0], field)    # 0
        
        assert quotient == expected_quotient
        assert remainder == expected_remainder

    def test_polynomial_division_with_remainder(self) -> None:
        """余りのある多項式の除法テスト"""
        field = RationalField()
        
        # (x^2 + x + 1) ÷ (x + 1) = x, remainder = 1
        dividend = Polynomial([1, 1, 1], field)  # x^2 + x + 1
        divisor = Polynomial([1, 1], field)      # x + 1
        
        quotient, remainder = dividend.divide(divisor)
        
        expected_quotient = Polynomial([0, 1], field)  # x
        expected_remainder = Polynomial([1], field)    # 1
        
        assert quotient == expected_quotient
        assert remainder == expected_remainder

    def test_polynomial_division_by_zero_error(self) -> None:
        """零多項式による除法のエラーテスト"""
        field = RationalField()
        
        dividend = Polynomial([1, 2, 3], field)
        zero_divisor = Polynomial([0], field)
        
        with pytest.raises(PolynomialException, match="零多項式による除法はできません"):
            dividend.divide(zero_divisor)

    def test_polynomial_gcd(self) -> None:
        """多項式の最大公約数テスト"""
        field = RationalField()
        
        # gcd(x^2 - 1, x^2 + 2x + 1) = x + 1
        # x^2 - 1 = (x-1)(x+1), x^2 + 2x + 1 = (x+1)^2
        poly1 = Polynomial([-1, 0, 1], field)   # x^2 - 1
        poly2 = Polynomial([1, 2, 1], field)    # x^2 + 2x + 1
        
        gcd_result = poly1.gcd(poly2)
        expected = Polynomial([1, 1], field)    # x + 1
        
        # GCDはモニック多項式として正規化される
        assert gcd_result.is_monic()
        assert gcd_result == expected


class TestPolynomialRing:
    """多項式環のテスト"""

    def test_polynomial_ring_creation(self) -> None:
        """多項式環の作成テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        assert poly_ring.base_ring == base_field
        assert poly_ring.variable == "x"
        assert poly_ring.name == "有理数体 Q[x]"

    def test_polynomial_ring_zero(self) -> None:
        """多項式環の零元テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        zero = poly_ring.zero()
        assert isinstance(zero, PolynomialElement)
        assert zero.polynomial.is_zero()

    def test_polynomial_ring_one(self) -> None:
        """多項式環の単位元テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        one = poly_ring.one()
        assert isinstance(one, PolynomialElement)
        assert one.polynomial.coefficients == [Fraction(1)]

    def test_polynomial_ring_variable(self) -> None:
        """多項式環の変数テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        x = poly_ring.x()
        assert isinstance(x, PolynomialElement)
        assert x.polynomial.coefficients == [Fraction(0), Fraction(1)]  # x

    def test_polynomial_ring_constant(self) -> None:
        """多項式環の定数多項式作成テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        const = poly_ring.constant(5)
        assert isinstance(const, PolynomialElement)
        assert const.polynomial.coefficients == [Fraction(5)]

    def test_polynomial_ring_from_coefficients(self) -> None:
        """係数から多項式環要素の作成テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 3x^2 + 2x + 1
        poly_elem = poly_ring.from_coefficients([1, 2, 3])
        assert isinstance(poly_elem, PolynomialElement)
        assert poly_elem.polynomial.coefficients == [Fraction(1), Fraction(2), Fraction(3)]


class TestPolynomialElement:
    """多項式環の要素のテスト"""

    def test_polynomial_element_creation(self) -> None:
        """多項式環要素の作成テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        polynomial = Polynomial([1, 2, 3], base_field)
        
        poly_elem = PolynomialElement(polynomial, poly_ring)
        assert poly_elem.polynomial == polynomial
        assert poly_elem.ring == poly_ring

    def test_polynomial_element_addition(self) -> None:
        """多項式環要素の加法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        poly1 = poly_ring.from_coefficients([1, 2, 1])  # x^2 + 2x + 1
        poly2 = poly_ring.from_coefficients([2, 3, 1])  # x^2 + 3x + 2
        
        result = poly1 + poly2
        expected_coeffs = [Fraction(3), Fraction(5), Fraction(2)]  # 2x^2 + 5x + 3
        assert result.polynomial.coefficients == expected_coeffs

    def test_polynomial_element_multiplication(self) -> None:
        """多項式環要素の乗法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        poly1 = poly_ring.from_coefficients([1, 1])  # x + 1
        poly2 = poly_ring.from_coefficients([2, 1])  # x + 2
        
        result = poly1 * poly2
        expected_coeffs = [Fraction(2), Fraction(3), Fraction(1)]  # x^2 + 3x + 2
        assert result.polynomial.coefficients == expected_coeffs

    def test_polynomial_element_different_rings_error(self) -> None:
        """異なる多項式環の要素同士の演算エラーテスト"""
        field1 = RationalField()
        field2 = RationalField()  # 同じ型だが異なるインスタンス
        
        ring1 = PolynomialRing(field1, "x")
        ring2 = PolynomialRing(field2, "y")
        
        poly1 = ring1.from_coefficients([1, 1])
        poly2 = ring2.from_coefficients([1, 1])
        
        with pytest.raises(ValueError, match="異なる多項式環の要素同士の演算はできません"):
            poly1 + poly2


class TestPolynomialSpecialMethods:
    """多項式の特殊メソッドのテスト"""

    def test_polynomial_derivative(self) -> None:
        """多項式の微分テスト"""
        field = RationalField()
        
        # f(x) = x^3 + 2x^2 + 3x + 4
        # f'(x) = 3x^2 + 4x + 3
        poly = Polynomial([4, 3, 2, 1], field)
        derivative = poly.derivative()
        
        expected = Polynomial([3, 4, 3], field)
        assert derivative == expected

    def test_polynomial_derivative_constant(self) -> None:
        """定数多項式の微分テスト"""
        field = RationalField()
        
        # f(x) = 5, f'(x) = 0
        poly = Polynomial([5], field)
        derivative = poly.derivative()
        
        expected = Polynomial([0], field)
        assert derivative == expected

    def test_polynomial_composition(self) -> None:
        """多項式の合成テスト"""
        field = RationalField()
        
        # f(x) = x^2 + 1, g(x) = x + 1
        # f(g(x)) = (x + 1)^2 + 1 = x^2 + 2x + 2
        f = Polynomial([1, 0, 1], field)  # x^2 + 1
        g = Polynomial([1, 1], field)     # x + 1
        
        composition = f.compose(g)
        expected = Polynomial([2, 2, 1], field)  # x^2 + 2x + 2
        assert composition == expected

    def test_polynomial_roots_finding(self) -> None:
        """多項式の根の発見テスト（簡単な場合）"""
        field = RationalField()
        
        # f(x) = x^2 - 1 = (x-1)(x+1)
        # 根は x = 1, -1
        poly = Polynomial([-1, 0, 1], field)
        
        # x = 1 が根であることを確認
        assert poly.evaluate(field.element(1)) == field.element(0)
        
        # x = -1 が根であることを確認
        assert poly.evaluate(field.element(-1)) == field.element(0)


class TestPolynomialExceptions:
    """多項式の例外処理テスト"""

    def test_polynomial_invalid_base_ring_error(self) -> None:
        """無効な基底環でのエラーテスト"""
        # 無効な基底環を仮定（実際の実装では適切な型チェックが必要）
        pass

    def test_polynomial_degree_error_for_zero(self) -> None:
        """零多項式の次数エラーテスト（実装依存）"""
        # 零多項式の次数を-∞とするか0とするかは実装依存
        # このテストは実装の方針に従って調整する
        pass


class TestPolynomialIrreducibility:
    """多項式の既約性判定のテスト"""

    def test_linear_polynomial_irreducible(self) -> None:
        """1次多項式は常に既約であることのテスト"""
        field = RationalField()
        
        # x + 1
        poly1 = Polynomial([1, 1], field)
        assert poly1.is_irreducible()
        
        # 2x + 3
        poly2 = Polynomial([3, 2], field)
        assert poly2.is_irreducible()
        
        # x - 5
        poly3 = Polynomial([-5, 1], field)
        assert poly3.is_irreducible()

    def test_constant_polynomial_not_irreducible(self) -> None:
        """定数多項式は既約ではないことのテスト"""
        field = RationalField()
        
        # 定数多項式 5
        poly1 = Polynomial([5], field)
        assert not poly1.is_irreducible()
        
        # 零多項式
        poly2 = Polynomial([0], field)
        assert not poly2.is_irreducible()

    def test_irreducible_quadratic_polynomial(self) -> None:
        """既約な2次多項式のテスト"""
        field = RationalField()
        
        # x^2 + 1 (有理数体上で既約)
        poly1 = Polynomial([1, 0, 1], field)
        assert poly1.is_irreducible()
        
        # x^2 + x + 1 (有理数体上で既約)
        poly2 = Polynomial([1, 1, 1], field)
        assert poly2.is_irreducible()
        
        # x^2 + 2 (有理数体上で既約)
        poly3 = Polynomial([2, 0, 1], field)
        assert poly3.is_irreducible()

    def test_reducible_quadratic_polynomial(self) -> None:
        """可約な2次多項式のテスト"""
        field = RationalField()
        
        # x^2 - 1 = (x-1)(x+1)
        poly1 = Polynomial([-1, 0, 1], field)
        assert not poly1.is_irreducible()
        
        # x^2 + 2x + 1 = (x+1)^2
        poly2 = Polynomial([1, 2, 1], field)
        assert not poly2.is_irreducible()
        
        # x^2 - 4 = (x-2)(x+2)
        poly3 = Polynomial([-4, 0, 1], field)
        assert not poly3.is_irreducible()

    def test_irreducible_cubic_polynomial(self) -> None:
        """既約な3次多項式のテスト"""
        field = RationalField()
        
        # x^3 + 2x + 1 (有理数体上で既約)
        poly1 = Polynomial([1, 2, 0, 1], field)
        assert poly1.is_irreducible()
        
        # x^3 + x^2 + x + 1 の因数分解を確認
        # これは実は可約: (x+1)(x^2+1) だが、有理数体上でx^2+1は既約
        # なので全体としては可約
        poly2 = Polynomial([1, 1, 1, 1], field)
        assert not poly2.is_irreducible()

    def test_reducible_cubic_polynomial(self) -> None:
        """可約な3次多項式のテスト"""
        field = RationalField()
        
        # x^3 - 1 = (x-1)(x^2+x+1)
        poly1 = Polynomial([-1, 0, 0, 1], field)
        assert not poly1.is_irreducible()
        
        # x^3 + x^2 - 2x = x(x^2 + x - 2) = x(x+2)(x-1)
        poly2 = Polynomial([0, -2, 1, 1], field)
        assert not poly2.is_irreducible()

    def test_irreducible_with_rational_root_test(self) -> None:
        """有理根定理を使った既約性判定のテスト"""
        field = RationalField()
        
        # x^3 + x + 1 (有理根を持たず、3次なので既約)
        poly1 = Polynomial([1, 1, 0, 1], field)
        assert poly1.is_irreducible()
        
        # x^4 + x + 1 (既約性の判定が必要)
        poly2 = Polynomial([1, 1, 0, 0, 1], field)
        # この多項式は実際には既約
        assert poly2.is_irreducible()

    def test_eisenstein_criterion_irreducible(self) -> None:
        """アイゼンシュタインの既約判定法のテスト"""
        field = RationalField()
        
        # x^2 + 2x + 2 (p=2でアイゼンシュタインの条件を満たす)
        poly1 = Polynomial([2, 2, 1], field)
        assert poly1.is_irreducible()
        
        # x^3 + 3x + 3 (p=3でアイゼンシュタインの条件を満たす)
        poly2 = Polynomial([3, 3, 0, 1], field)
        assert poly2.is_irreducible()

    def test_factorization_based_irreducibility(self) -> None:
        """因数分解に基づく既約性判定のテスト"""
        field = RationalField()
        
        # 既に因数分解されている多項式を構築
        # (x + 1)(x + 2) = x^2 + 3x + 2
        factor1 = Polynomial([1, 1], field)  # x + 1
        factor2 = Polynomial([2, 1], field)  # x + 2
        product = factor1 * factor2
        
        # 積は可約であるべき
        assert not product.is_irreducible()
        
        # 個々の因子は既約であるべき
        assert factor1.is_irreducible()
        assert factor2.is_irreducible()

    def test_content_and_primitive_part(self) -> None:
        """内容と原始部分のテスト（既約性判定の前処理）"""
        field = RationalField()
        
        # 2x^2 + 4x + 2 = 2(x^2 + 2x + 1) = 2(x+1)^2
        poly = Polynomial([2, 4, 2], field)
        
        # 内容を取得
        content = poly.content()
        assert content == Fraction(2)
        
        # 原始部分を取得
        primitive = poly.primitive_part()
        expected_primitive = Polynomial([1, 2, 1], field)  # x^2 + 2x + 1
        assert primitive == expected_primitive
        
        # 原始部分は可約（(x+1)^2）
        assert not primitive.is_irreducible()

    def test_degree_bounds_for_irreducibility(self) -> None:
        """次数による既約性の境界テスト"""
        field = RationalField()
        
        # 次数0の多項式（定数）は既約ではない
        poly0 = Polynomial([5], field)
        assert not poly0.is_irreducible()
        
        # 次数1の多項式は常に既約
        poly1 = Polynomial([3, 2], field)
        assert poly1.is_irreducible()
        
        # 次数2以上では実際の因数分解が必要
        poly2 = Polynomial([1, 0, 1], field)  # x^2 + 1
        assert poly2.is_irreducible()

    def test_square_free_and_irreducibility(self) -> None:
        """平方因子を持たない多項式と既約性のテスト"""
        field = RationalField()
        
        # x^2 + 2x + 1 = (x+1)^2 (平方因子を持つ)
        poly_with_square = Polynomial([1, 2, 1], field)
        assert not poly_with_square.is_square_free()
        assert not poly_with_square.is_irreducible()
        
        # x^2 + x + 1 (平方因子を持たない)
        poly_square_free = Polynomial([1, 1, 1], field)
        assert poly_square_free.is_square_free()
        assert poly_square_free.is_irreducible()

    def test_simple_finite_field_irreducibility(self) -> None:
        """有限体上での簡単な既約性判定テスト"""
        from galois_theory.field import FiniteField
        
        # F_2 上で
        field = FiniteField(2)
        
        # x + 1 は既約（1次多項式）
        poly1 = Polynomial([1, 1], field)
        assert poly1.is_irreducible()
        
        # x^2 + x + 1 は F_2 上で既約
        poly2 = Polynomial([1, 1, 1], field)
        assert poly2.is_irreducible()
        
        # x^2 + 1 = (x+1)^2 は F_2 上で可約
        poly3 = Polynomial([1, 0, 1], field)
        assert not poly3.is_irreducible()

    def test_finite_field_f3_irreducibility(self) -> None:
        """F_3 上での既約性判定テスト"""
        from galois_theory.field import FiniteField
        
        field = FiniteField(3)
        
        # x + 1 は既約（1次多項式）
        poly1 = Polynomial([1, 1], field)
        assert poly1.is_irreducible()
        
        # x^2 + 1 は F_3 上で既約（根を持たない）
        poly2 = Polynomial([1, 0, 1], field)
        assert poly2.is_irreducible()
        
        # x^2 - 1 = x^2 + 2 は F_3 上で可約（x=1, x=2で根を持つ）
        poly3 = Polynomial([2, 0, 1], field)  # x^2 + 2 (mod 3)
        assert not poly3.is_irreducible()

    def test_finite_field_f5_irreducibility(self) -> None:
        """F_5 上での既約性判定テスト"""
        from galois_theory.field import FiniteField
        
        field = FiniteField(5)
        
        # x^2 + 2 は F_5 上で既約
        poly1 = Polynomial([2, 0, 1], field)
        assert poly1.is_irreducible()
        
        # x^2 - 1 = x^2 + 4 は F_5 上で可約（x=1, x=4で根を持つ）
        poly2 = Polynomial([4, 0, 1], field)  # x^2 + 4 ≡ x^2 - 1 (mod 5)
        assert not poly2.is_irreducible()

    def test_finite_field_cubic_irreducibility(self) -> None:
        """有限体上での3次多項式の既約性判定テスト"""
        from galois_theory.field import FiniteField
        
        # F_2 上で
        field2 = FiniteField(2)
        
        # x^3 + x + 1 は F_2 上で既約
        poly1 = Polynomial([1, 1, 0, 1], field2)
        assert poly1.is_irreducible()
        
        # x^3 + 1 = (x+1)(x^2+x+1) は F_2 上で可約
        poly2 = Polynomial([1, 0, 0, 1], field2)
        assert not poly2.is_irreducible() 