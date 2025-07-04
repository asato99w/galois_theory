"""
テスト環境の動作確認用テスト

このファイルは、テスト環境が正しく設定されているかを確認するための
基本的なテストを含んでいます。
"""

import unittest
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import sympy as sp
from hypothesis import given, strategies as st


class TestEnvironment(unittest.TestCase):
    """テスト環境の基本動作確認"""

    def test_basic_assertion(self) -> None:
        """基本的なアサーションのテスト"""
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)

    def test_fractions_import(self) -> None:
        """標準ライブラリのfractionsが正しく動作するかテスト"""
        frac = Fraction(1, 2)
        self.assertEqual(frac + Fraction(1, 3), Fraction(5, 6))

    def test_numpy_import(self) -> None:
        """NumPyが正しく動作するかテスト"""
        arr = np.array([1, 2, 3])
        self.assertEqual(arr.sum(), 6)

    def test_sympy_import(self) -> None:
        """SymPyが正しく動作するかテスト"""
        x = sp.Symbol("x")
        expr = x**2 + 2 * x + 1
        factored = sp.factor(expr)
        self.assertEqual(factored, (x + 1) ** 2)


class TestPytestFeatures:
    """pytest固有の機能のテスト"""

    def test_pytest_basic(self) -> None:
        """pytestの基本機能テスト"""
        assert True
        assert 2 + 2 == 4

    @pytest.mark.unit
    def test_unit_marker(self) -> None:
        """単体テストマーカーのテスト"""
        assert "unit" in "unittest"

    @pytest.mark.integration
    def test_integration_marker(self) -> None:
        """統合テストマーカーのテスト"""
        assert "integration" in "integration_test"

    @pytest.mark.slow
    def test_slow_marker(self) -> None:
        """重いテストマーカーのテスト"""
        # 実際には重い処理はしない
        assert True

    def test_parametrize(self) -> None:
        """パラメータ化テストの例"""
        test_cases = [
            (1, 1, 2),
            (2, 3, 5),
            (0, 0, 0),
            (-1, 1, 0),
        ]

        for a, b, expected in test_cases:
            assert a + b == expected


class TestHypothesis:
    """Hypothesisプロパティベーステストの確認"""

    @given(st.integers())
    def test_integer_property(self, n: int) -> None:
        """整数の性質をテスト"""
        assert n + 0 == n
        assert n * 1 == n

    @given(st.integers(min_value=0, max_value=100))
    def test_positive_integer_property(self, n: int) -> None:
        """正の整数の性質をテスト"""
        assert n >= 0
        assert n <= 100

    @given(st.lists(st.integers(), min_size=1))
    def test_list_property(self, lst: list[int]) -> None:
        """リストの性質をテスト"""
        assert len(lst) >= 1
        assert max(lst) in lst
        assert min(lst) in lst


if __name__ == "__main__":
    unittest.main()
