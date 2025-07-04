"""
体の拡大（Field Extensions）のテスト

このファイルは、体の拡大の基本的な性質と操作をテストします。
ガロア理論における体拡大の概念を実装するためのテストファーストアプローチです。

体の拡大 K/F は、体 F を含む体 K のことです。
"""

import pytest
from fractions import Fraction
from typing import Any, List

from galois_theory.field import RationalField, FiniteField
from galois_theory.polynomials import Polynomial, PolynomialRing
from galois_theory.field_extensions import (
    FieldExtension, ExtensionElement, SimpleExtension,
    AlgebraicElement, MinimalPolynomial, SplittingField,
    FieldExtensionException
)


class TestFieldExtensionBasics:
    """体拡大の基本的な性質のテスト"""

    def test_simple_extension_creation(self) -> None:
        """単純拡大の作成テスト"""
        # Q(√2) = Q[x]/(x^2 - 2)
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 - 2 (√2の最小多項式)
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        assert extension.base_field == base_field
        assert extension.minimal_polynomial == minimal_poly.polynomial
        assert extension.generator_name == "sqrt2"
        assert extension.name == "有理数体 Q(sqrt2)"

    def test_extension_degree_calculation(self) -> None:
        """拡大次数の計算テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2): [Q(√2):Q] = 2
        minimal_poly_sqrt2 = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        ext_sqrt2 = SimpleExtension(base_field, minimal_poly_sqrt2, "sqrt2")
        assert ext_sqrt2.degree() == 2
        
        # Q(∛2): [Q(∛2):Q] = 3
        minimal_poly_cbrt2 = poly_ring.from_coefficients([-2, 0, 0, 1])  # x^3 - 2
        ext_cbrt2 = SimpleExtension(base_field, minimal_poly_cbrt2, "cbrt2")
        assert ext_cbrt2.degree() == 3

    def test_extension_element_creation(self) -> None:
        """拡大体の要素作成テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2)
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # 3 + 2√2 を表現
        coeffs = [3, 2]  # 3 + 2*sqrt2
        element = ExtensionElement(coeffs, extension)
        
        assert element.coefficients == [Fraction(3), Fraction(2)]
        assert element.extension == extension

    def test_extension_element_string_representation(self) -> None:
        """拡大体要素の文字列表現テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # 3 + 2√2
        element1 = ExtensionElement([3, 2], extension)
        assert str(element1) == "3 + 2*sqrt2"
        
        # 5
        element2 = ExtensionElement([5], extension)
        assert str(element2) == "5"
        
        # √2
        element3 = ExtensionElement([0, 1], extension)
        assert str(element3) == "sqrt2"
        
        # -1 + 3√2
        element4 = ExtensionElement([-1, 3], extension)
        assert str(element4) == "-1 + 3*sqrt2"

    def test_base_field_embedding(self) -> None:
        """基底体の埋め込みテスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # 有理数 3/2 の埋め込み
        rational_element = base_field.element(Fraction(3, 2))
        embedded = extension.embed_base_element(rational_element)
        
        assert embedded.coefficients == [Fraction(3, 2), Fraction(0)]
        assert embedded.extension == extension

    def test_generator_element(self) -> None:
        """生成元の取得テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        generator = extension.generator()
        assert generator.coefficients == [Fraction(0), Fraction(1)]  # 0 + 1*sqrt2
        assert generator.extension == extension


class TestExtensionElementArithmetic:
    """拡大体要素の算術演算のテスト"""

    def test_extension_element_addition(self) -> None:
        """拡大体要素の加法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # (3 + 2√2) + (1 + 4√2) = 4 + 6√2
        elem1 = ExtensionElement([3, 2], extension)
        elem2 = ExtensionElement([1, 4], extension)
        result = elem1 + elem2
        
        assert result.coefficients == [Fraction(4), Fraction(6)]

    def test_extension_element_subtraction(self) -> None:
        """拡大体要素の減法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # (5 + 3√2) - (2 + √2) = 3 + 2√2
        elem1 = ExtensionElement([5, 3], extension)
        elem2 = ExtensionElement([2, 1], extension)
        result = elem1 - elem2
        
        assert result.coefficients == [Fraction(3), Fraction(2)]

    def test_extension_element_multiplication(self) -> None:
        """拡大体要素の乗法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # (1 + √2) * (3 + 2√2)
        # = 3 + 2√2 + 3√2 + 2*2
        # = 3 + 5√2 + 4
        # = 7 + 5√2
        elem1 = ExtensionElement([1, 1], extension)  # 1 + √2
        elem2 = ExtensionElement([3, 2], extension)  # 3 + 2√2
        result = elem1 * elem2
        
        assert result.coefficients == [Fraction(7), Fraction(5)]

    def test_extension_element_multiplication_by_base(self) -> None:
        """基底体要素との乗法テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # 3 * (2 + √2) = 6 + 3√2
        element = ExtensionElement([2, 1], extension)
        scalar = base_field.element(3)
        result = element.multiply_by_base(scalar)
        
        assert result.coefficients == [Fraction(6), Fraction(3)]

    def test_extension_element_power(self) -> None:
        """拡大体要素の冪乗テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # (1 + √2)^2 = 1 + 2√2 + 2 = 3 + 2√2
        element = ExtensionElement([1, 1], extension)  # 1 + √2
        result = element ** 2
        
        assert result.coefficients == [Fraction(3), Fraction(2)]

    def test_extension_element_inverse(self) -> None:
        """拡大体要素の逆元テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # (1 + √2)^(-1)
        # (1 + √2)(1 - √2) = 1 - 2 = -1
        # なので (1 + √2)^(-1) = (1 - √2)/(-1) = -1 + √2
        element = ExtensionElement([1, 1], extension)  # 1 + √2
        inverse = element.inverse()
        
        assert inverse.coefficients == [Fraction(-1), Fraction(1)]
        
        # 逆元との積が1になることを確認
        product = element * inverse
        assert product.coefficients == [Fraction(1), Fraction(0)]


class TestMinimalPolynomial:
    """最小多項式の計算テスト"""

    def test_minimal_polynomial_of_rational(self) -> None:
        """有理数の最小多項式テスト"""
        base_field = RationalField()
        
        # 有理数 3/2 の最小多項式は x - 3/2
        rational_element = AlgebraicElement(Fraction(3, 2), base_field)
        minimal = MinimalPolynomial.compute(rational_element, base_field)
        
        # x - 3/2 = -3/2 + x
        expected_coeffs = [Fraction(-3, 2), Fraction(1)]
        assert minimal.coefficients == expected_coeffs

    def test_minimal_polynomial_of_sqrt2(self) -> None:
        """√2の最小多項式テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2)を構築
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # √2 ∈ Q(√2)
        sqrt2 = extension.generator()
        
        # √2の最小多項式を計算（Q上で）
        minimal = MinimalPolynomial.compute_in_extension(sqrt2, base_field)
        
        # x^2 - 2
        expected_coeffs = [Fraction(-2), Fraction(0), Fraction(1)]
        assert minimal.coefficients == expected_coeffs

    def test_minimal_polynomial_degree_bound(self) -> None:
        """最小多項式の次数の上界テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(∛2): [Q(∛2):Q] = 3
        minimal_poly = poly_ring.from_coefficients([-2, 0, 0, 1])  # x^3 - 2
        extension = SimpleExtension(base_field, minimal_poly, "cbrt2")
        
        # ∛2の最小多項式の次数は3以下
        cbrt2 = extension.generator()
        minimal = MinimalPolynomial.compute_in_extension(cbrt2, base_field)
        
        assert minimal.degree() <= 3
        assert minimal.degree() == 3  # 実際には3


class TestTowerOfExtensions:
    """拡大の塔のテスト"""

    def test_tower_degree_multiplication(self) -> None:
        """拡大次数の乗法性テスト: [K:F] = [K:L][L:F]"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q ⊆ Q(√2) ⊆ Q(√2, √3)
        
        # Q(√2): [Q(√2):Q] = 2
        sqrt2_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        q_sqrt2 = SimpleExtension(base_field, sqrt2_poly, "sqrt2")
        
        # Q(√2)(√3): [Q(√2, √3):Q(√2)] = 2 (√3は Q(√2)上で最小多項式 x^2 - 3)
        poly_ring_ext = PolynomialRing(q_sqrt2, "y")
        sqrt3_poly = poly_ring_ext.from_coefficients([-3, 0, 1])  # y^2 - 3
        q_sqrt2_sqrt3 = SimpleExtension(q_sqrt2, sqrt3_poly, "sqrt3")
        
        # [Q(√2, √3):Q] = [Q(√2, √3):Q(√2)] * [Q(√2):Q] = 2 * 2 = 4
        assert q_sqrt2.degree() == 2
        assert q_sqrt2_sqrt3.degree() == 2
        assert q_sqrt2_sqrt3.absolute_degree() == 4

    def test_primitive_element_theorem(self) -> None:
        """原始元定理のテスト: Q(√2, √3) = Q(√2 + √3)"""
        # この複雑なテストは実装後に詳細化
        pass


class TestSplittingField:
    """分解体のテスト"""

    def test_splitting_field_quadratic(self) -> None:
        """2次多項式の分解体テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 - 2 の分解体は Q(√2)
        polynomial = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        splitting_field = SplittingField.construct(polynomial.polynomial, base_field)
        
        assert splitting_field.degree() == 2
        
        # 分解体内で多項式が完全に分解することを確認
        roots = splitting_field.find_roots(polynomial.polynomial)
        assert len(roots) == 2

    def test_splitting_field_cubic(self) -> None:
        """3次多項式の分解体テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^3 - 2 の分解体
        polynomial = poly_ring.from_coefficients([-2, 0, 0, 1])  # x^3 - 2
        splitting_field = SplittingField.construct(polynomial.polynomial, base_field)
        
        # [Q(∛2, ω):Q] = 6 (ωは1の原始3乗根)
        assert splitting_field.degree() == 6
        
        # 分解体内で多項式が完全に分解
        roots = splitting_field.find_roots(polynomial.polynomial)
        assert len(roots) == 3

    def test_splitting_field_irreducible(self) -> None:
        """既約多項式の分解体テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 + 1 (有理数体上で既約)
        polynomial = poly_ring.from_coefficients([1, 0, 1])  # x^2 + 1
        splitting_field = SplittingField.construct(polynomial.polynomial, base_field)
        
        assert splitting_field.degree() == 2
        
        # 複素数 i が含まれることを確認
        roots = splitting_field.find_roots(polynomial.polynomial)
        assert len(roots) == 2


class TestFiniteFieldExtensions:
    """有限体の拡大テスト"""

    def test_finite_field_simple_extension(self) -> None:
        """有限体の単純拡大テスト"""
        # F_2 ⊆ F_4 = F_2[x]/(x^2 + x + 1)
        base_field = FiniteField(2)
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 + x + 1 (F_2上で既約)
        minimal_poly = poly_ring.from_coefficients([1, 1, 1])  # x^2 + x + 1
        f4 = SimpleExtension(base_field, minimal_poly, "alpha")
        
        assert f4.degree() == 2
        assert f4.cardinality() == 4

    def test_finite_field_extension_elements(self) -> None:
        """有限体拡大の要素テスト"""
        base_field = FiniteField(2)
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([1, 1, 1])  # x^2 + x + 1
        f4 = SimpleExtension(base_field, minimal_poly, "alpha")
        
        # F_4 = {0, 1, α, α+1}
        zero = ExtensionElement([0], f4)
        one = ExtensionElement([1], f4)
        alpha = ExtensionElement([0, 1], f4)  # α
        alpha_plus_one = ExtensionElement([1, 1], f4)  # α + 1
        
        # α^2 + α + 1 = 0 なので α^2 = α + 1
        alpha_squared = alpha ** 2
        assert alpha_squared == alpha_plus_one

    def test_finite_field_multiplicative_group(self) -> None:
        """有限体の乗法群テスト"""
        base_field = FiniteField(2)
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([1, 1, 1])
        f4 = SimpleExtension(base_field, minimal_poly, "alpha")
        
        alpha = ExtensionElement([0, 1], f4)
        
        # F_4*の位数は3、αは原始元
        assert alpha.multiplicative_order() == 3
        
        # α^3 = 1
        alpha_cubed = alpha ** 3
        one = ExtensionElement([1], f4)
        assert alpha_cubed == one


class TestAlgebraicClosure:
    """代数的閉体のテスト"""

    def test_algebraic_closure_existence(self) -> None:
        """代数的閉体の存在テスト（概念的）"""
        # 有理数体の代数的閉体が存在することの確認
        base_field = RationalField()
        
        # 代数的閉体では全ての非定数多項式が根を持つ
        # これは実装上の概念テストなので、具体的な計算は行わない
        assert hasattr(base_field, 'algebraic_closure')

    def test_algebraically_closed_field_property(self) -> None:
        """代数的閉体の性質テスト"""
        # 複素数体（概念的なテスト）
        # 実装時に詳細化
        pass


class TestFieldAutomorphisms:
    """体の自己同型写像のテスト"""

    def test_galois_group_quadratic(self) -> None:
        """2次拡大のガロア群テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2)/Q のガロア群
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # ガロア群は位数2の巡回群 {id, σ}
        # σ: √2 ↦ -√2
        galois_group = extension.galois_group(base_field)
        assert galois_group.order() == 2
        assert galois_group.is_cyclic()

    def test_automorphism_fixed_field(self) -> None:
        """自己同型写像の固定体テスト"""
        # Gal(Q(√2)/Q) = {id, σ} where σ(√2) = -√2
        # Fixed field of σ should be Q
        pass

    def test_fundamental_theorem_galois_theory(self) -> None:
        """ガロア理論の基本定理テスト"""
        # 拡大と群の間の対応
        # 実装後に詳細化
        pass


class TestFieldExtensionExceptions:
    """体拡大の例外処理テスト"""

    def test_non_irreducible_polynomial_error(self) -> None:
        """既約でない多項式による拡大のエラーテスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 - 1 = (x-1)(x+1) (可約)
        reducible_poly = poly_ring.from_coefficients([-1, 0, 1])
        
        with pytest.raises(FieldExtensionException, match="多項式は既約である必要があります"):
            SimpleExtension(base_field, reducible_poly, "alpha")

    def test_incompatible_field_operations_error(self) -> None:
        """異なる体の要素同士の演算エラーテスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2) と Q(√3)
        sqrt2_poly = poly_ring.from_coefficients([-2, 0, 1])
        sqrt3_poly = poly_ring.from_coefficients([-3, 0, 1])
        
        ext_sqrt2 = SimpleExtension(base_field, sqrt2_poly, "sqrt2")
        ext_sqrt3 = SimpleExtension(base_field, sqrt3_poly, "sqrt3")
        
        elem_sqrt2 = ExtensionElement([1, 1], ext_sqrt2)  # 1 + √2
        elem_sqrt3 = ExtensionElement([1, 1], ext_sqrt3)  # 1 + √3
        
        with pytest.raises(ValueError, match="異なる体拡大の要素同士の演算はできません"):
            elem_sqrt2 + elem_sqrt3

    def test_degree_zero_extension_error(self) -> None:
        """次数0の拡大エラーテスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 定数多項式
        constant_poly = poly_ring.from_coefficients([5])
        
        with pytest.raises(FieldExtensionException, match="多項式は既約である必要があります"):
            SimpleExtension(base_field, constant_poly, "alpha")


class TestComputationalExamples:
    """計算例のテスト"""

    def test_quadratic_formula_in_extension(self) -> None:
        """拡大体での2次方程式の解テスト"""
        # x^2 + x + 1 = 0 in Q(√(-3))
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√(-3)) = Q[x]/(x^2 + 3)
        minimal_poly = poly_ring.from_coefficients([3, 0, 1])  # x^2 + 3
        extension = SimpleExtension(base_field, minimal_poly, "sqrt_neg3")
        
        # x^2 + x + 1 の解を拡大体で計算
        equation = poly_ring.from_coefficients([1, 1, 1])  # x^2 + x + 1
        roots = extension.solve_polynomial(equation.polynomial)
        
        assert len(roots) == 2

    def test_cyclotomic_field(self) -> None:
        """円分体のテスト"""
        # Q(ζ_3) where ζ_3 is a primitive 3rd root of unity
        # ζ_3 satisfies x^2 + x + 1 = 0
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x^2 + x + 1 (3次円分多項式)
        cyclotomic_poly = poly_ring.from_coefficients([1, 1, 1])
        cyclotomic_field = SimpleExtension(base_field, cyclotomic_poly, "zeta3")
        
        assert cyclotomic_field.degree() == 2
        
        # ζ_3 の位数は3
        zeta3 = cyclotomic_field.generator()
        assert zeta3.multiplicative_order() == 3

    def test_field_norm_and_trace(self) -> None:
        """体のノルムとトレースのテスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x^2 - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        # 1 + √2 のノルムとトレース
        element = ExtensionElement([1, 1], extension)  # 1 + √2
        
        # ノルム: N(1 + √2) = (1 + √2)(1 - √2) = 1 - 2 = -1
        norm = element.norm()
        assert norm == Fraction(-1)
        
        # トレース: Tr(1 + √2) = (1 + √2) + (1 - √2) = 2
        trace = element.trace()
        assert trace == Fraction(2) 