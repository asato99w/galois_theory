"""
体（Field）のテストケース

このモジュールは、体の実装をテストします。
体は環の特殊な場合で、零以外の全ての元が乗法逆元を持つ代数構造です。
"""

import pytest
from fractions import Fraction
from galois_theory.field import Field, FieldElement, RationalField


class TestFieldElement:
    """FieldElementクラスのテスト"""

    def test_field_element_creation(self):
        """体の要素の作成をテスト"""
        Q = RationalField()
        element = Q.element(Fraction(3, 4))
        
        assert element.value == Fraction(3, 4)
        assert element.field == Q

    def test_field_element_equality(self):
        """体の要素の等価性をテスト"""
        Q = RationalField()
        element1 = Q.element(Fraction(3, 4))
        element2 = Q.element(Fraction(3, 4))
        element3 = Q.element(Fraction(1, 2))
        
        assert element1 == element2
        assert element1 != element3

    def test_field_element_addition(self):
        """体の要素の加法をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(1, 3))
        b = Q.element(Fraction(1, 6))
        
        result = a + b
        expected = Q.element(Fraction(1, 2))
        
        assert result == expected

    def test_field_element_multiplication(self):
        """体の要素の乗法をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(2, 3))
        b = Q.element(Fraction(3, 4))
        
        result = a * b
        expected = Q.element(Fraction(1, 2))
        
        assert result == expected

    def test_field_element_division(self):
        """体の要素の除法をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(1, 2))
        b = Q.element(Fraction(1, 3))
        
        result = a / b
        expected = Q.element(Fraction(3, 2))
        
        assert result == expected

    def test_field_element_division_by_zero_raises_error(self):
        """零による除法がエラーを発生させることをテスト"""
        Q = RationalField()
        a = Q.element(Fraction(1, 2))
        zero = Q.zero()
        
        with pytest.raises(ValueError, match="零要素による除法はできません"):
            a / zero

    def test_field_element_additive_inverse(self):
        """加法逆元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(3, 4))
        neg_a = -a
        
        assert a + neg_a == Q.zero()

    def test_field_element_multiplicative_inverse(self):
        """乗法逆元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(3, 4))
        inv_a = a.inverse()
        
        assert a * inv_a == Q.one()

    def test_zero_has_no_multiplicative_inverse(self):
        """零要素に乗法逆元が存在しないことをテスト"""
        Q = RationalField()
        zero = Q.zero()
        
        with pytest.raises(ValueError, match="零要素の乗法逆元は存在しません"):
            zero.inverse()


class TestRationalField:
    """有理数体のテスト"""

    def test_rational_field_creation(self):
        """有理数体の作成をテスト"""
        Q = RationalField()
        assert Q.name == "有理数体 Q"

    def test_rational_field_contains(self):
        """有理数体の要素判定をテスト"""
        Q = RationalField()
        
        assert Q.contains(Fraction(3, 4))
        assert Q.contains(5)  # 整数も含む
        assert Q.contains(0)
        assert not Q.contains("string")
        assert not Q.contains(3.14)  # 浮動小数点数は含まない

    def test_rational_field_zero_and_one(self):
        """有理数体の零元と単位元をテスト"""
        Q = RationalField()
        
        zero = Q.zero()
        one = Q.one()
        
        assert zero.value == Fraction(0)
        assert one.value == Fraction(1)

    def test_rational_field_element_creation_with_integers(self):
        """整数からの有理数体要素作成をテスト"""
        Q = RationalField()
        element = Q.element(5)
        
        assert element.value == Fraction(5)
        assert element.field == Q

    def test_field_axioms_associativity_addition(self):
        """加法の結合律をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(1, 2))
        b = Q.element(Fraction(1, 3))
        c = Q.element(Fraction(1, 6))
        
        # (a + b) + c = a + (b + c)
        left = (a + b) + c
        right = a + (b + c)
        
        assert left == right

    def test_field_axioms_commutativity_addition(self):
        """加法の交換律をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(2, 3))
        b = Q.element(Fraction(1, 4))
        
        # a + b = b + a
        assert a + b == b + a

    def test_field_axioms_associativity_multiplication(self):
        """乗法の結合律をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(2, 3))
        b = Q.element(Fraction(3, 4))
        c = Q.element(Fraction(1, 2))
        
        # (a * b) * c = a * (b * c)
        left = (a * b) * c
        right = a * (b * c)
        
        assert left == right

    def test_field_axioms_commutativity_multiplication(self):
        """乗法の交換律をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(3, 4))
        b = Q.element(Fraction(2, 5))
        
        # a * b = b * a
        assert a * b == b * a

    def test_field_axioms_distributivity(self):
        """分配律をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(2, 3))
        b = Q.element(Fraction(1, 4))
        c = Q.element(Fraction(1, 6))
        
        # a * (b + c) = a * b + a * c
        left = a * (b + c)
        right = a * b + a * c
        
        assert left == right

    def test_field_axioms_additive_identity(self):
        """加法単位元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(3, 7))
        zero = Q.zero()
        
        # a + 0 = a
        assert a + zero == a
        assert zero + a == a

    def test_field_axioms_multiplicative_identity(self):
        """乗法単位元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(5, 8))
        one = Q.one()
        
        # a * 1 = a
        assert a * one == a
        assert one * a == a

    def test_field_axioms_additive_inverse(self):
        """加法逆元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(4, 9))
        neg_a = -a
        zero = Q.zero()
        
        # a + (-a) = 0
        assert a + neg_a == zero

    def test_field_axioms_multiplicative_inverse(self):
        """乗法逆元をテスト"""
        Q = RationalField()
        a = Q.element(Fraction(7, 11))
        inv_a = a.inverse()
        one = Q.one()
        
        # a * a^(-1) = 1 (a ≠ 0)
        assert a * inv_a == one

    def test_field_contains_proper_subset_operations(self):
        """適切な部分集合での演算をテスト"""
        Q = RationalField()
        
        # 正の有理数での演算
        a = Q.element(Fraction(3, 4))
        b = Q.element(Fraction(2, 5))
        
        product = a * b
        assert product.value == Fraction(6, 20)  # = Fraction(3, 10)

    def test_complex_rational_arithmetic(self):
        """複雑な有理数演算をテスト"""
        Q = RationalField()
        
        # (2/3 + 1/4) * (5/6 - 1/3) / (1/2)
        a = Q.element(Fraction(2, 3))
        b = Q.element(Fraction(1, 4))
        c = Q.element(Fraction(5, 6))
        d = Q.element(Fraction(1, 3))
        e = Q.element(Fraction(1, 2))
        
        result = ((a + b) * (c - d)) / e
        
        # 手計算: (8/12 + 3/12) * (5/6 - 2/6) / (1/2)
        #        = (11/12) * (3/6) / (1/2)
        #        = (11/12) * (1/2) / (1/2)
        #        = (11/12) * 1
        #        = 11/12
        expected = Q.element(Fraction(11, 12))
        
        assert result == expected


class TestFiniteField:
    """有限体のテスト"""

    def test_finite_field_mod_5_creation(self):
        """有限体 F_5 の作成をテスト"""
        from galois_theory.field import FiniteField
        
        F5 = FiniteField(5)
        assert F5.name == "有限体 F_5"
        assert F5.characteristic == 5

    def test_finite_field_mod_5_elements(self):
        """有限体 F_5 の要素をテスト"""
        from galois_theory.field import FiniteField
        
        F5 = FiniteField(5)
        
        # F_5 = {0, 1, 2, 3, 4}
        for i in range(5):
            assert F5.contains(i)
        
        assert not F5.contains(5)
        assert not F5.contains(-1)

    def test_finite_field_mod_5_addition(self):
        """有限体 F_5 での加法をテスト"""
        from galois_theory.field import FiniteField
        
        F5 = FiniteField(5)
        
        # 3 + 4 = 2 (mod 5)
        a = F5.element(3)
        b = F5.element(4)
        result = a + b
        expected = F5.element(2)
        
        assert result == expected

    def test_finite_field_mod_5_multiplication(self):
        """有限体 F_5 での乗法をテスト"""
        from galois_theory.field import FiniteField
        
        F5 = FiniteField(5)
        
        # 3 * 4 = 2 (mod 5)
        a = F5.element(3)
        b = F5.element(4)
        result = a * b
        expected = F5.element(2)
        
        assert result == expected

    def test_finite_field_mod_5_multiplicative_inverse(self):
        """有限体 F_5 での乗法逆元をテスト"""
        from galois_theory.field import FiniteField
        
        F5 = FiniteField(5)
        
        # 2の逆元は3 (2 * 3 = 6 ≡ 1 (mod 5))
        a = F5.element(2)
        inv_a = a.inverse()
        expected = F5.element(3)
        
        assert inv_a == expected
        assert a * inv_a == F5.one()

    def test_finite_field_is_prime_order_required(self):
        """有限体の位数は素数である必要があることをテスト"""
        from galois_theory.field import FiniteField
        
        # 素数でない場合はエラー
        with pytest.raises(ValueError, match="有限体の位数は素数である必要があります"):
            FiniteField(4)  # 4 = 2^2 は素数ではない
        
        with pytest.raises(ValueError, match="有限体の位数は素数である必要があります"):
            FiniteField(6)  # 6 = 2*3 は素数ではない


class TestFieldOperationsIntegration:
    """体の演算の統合テスト"""

    def test_mixed_arithmetic_operations(self):
        """混合演算のテスト"""
        Q = RationalField()
        
        # (3/4 + 1/2) * (2/3)^(-1) - 1/6
        a = Q.element(Fraction(3, 4))
        b = Q.element(Fraction(1, 2))
        c = Q.element(Fraction(2, 3))
        d = Q.element(Fraction(1, 6))
        
        result = (a + b) * c.inverse() - d
        
        # 手計算: (3/4 + 2/4) * (3/2) - 1/6
        #        = (5/4) * (3/2) - 1/6
        #        = 15/8 - 1/6
        #        = 45/24 - 4/24
        #        = 41/24
        expected = Q.element(Fraction(41, 24))
        
        assert result == expected

    def test_repeated_operations(self):
        """反復演算のテスト"""
        Q = RationalField()
        
        # ((2/3)^2)^2 = (2/3)^4
        a = Q.element(Fraction(2, 3))
        
        # a^2
        a_squared = a * a
        expected_squared = Q.element(Fraction(4, 9))
        assert a_squared == expected_squared
        
        # (a^2)^2 = a^4
        a_fourth = a_squared * a_squared
        expected_fourth = Q.element(Fraction(16, 81))
        assert a_fourth == expected_fourth

    def test_field_closure_under_operations(self):
        """演算に対する閉性をテスト"""
        Q = RationalField()
        
        # 様々な有理数での演算が体内に閉じていることを確認
        test_values = [
            Fraction(1, 2), Fraction(3, 4), Fraction(5, 7),
            Fraction(-2, 3), Fraction(7, 11), Fraction(0)
        ]
        
        for a_val in test_values:
            for b_val in test_values:
                a = Q.element(a_val)
                b = Q.element(b_val)
                
                # 加法で閉じている
                sum_result = a + b
                assert Q.contains(sum_result.value)
                
                # 乗法で閉じている
                mul_result = a * b
                assert Q.contains(mul_result.value)
                
                # 除法で閉じている（ゼロ除算以外）
                if b_val != 0:
                    div_result = a / b
                    assert Q.contains(div_result.value) 