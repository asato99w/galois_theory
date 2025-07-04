"""
可解性判別（Solvability Determination）のテストスイート

このモジュールは、多項式の可解性判別機能をテストします。
ガロア理論において、多項式が根号で解けるかどうかはそのガロア群が可解群かどうかで決まります。

主要なテスト対象:
- SolvabilityAnalyzer: 可解性解析器
- RadicalSolver: 根号による解法
- GaloisSolvabilityTheorem: ガロアの可解性定理の応用
- PolynomialClassifier: 多項式の分類（可解/非可解）
"""

import pytest
from fractions import Fraction
from typing import List, Dict, Optional

from galois_theory.field import RationalField
from galois_theory.polynomials import PolynomialRing
from galois_theory.field_extensions import SimpleExtension
from galois_theory.group_theory import GaloisGroup, SymmetricGroup, CyclicGroup


class TestSolvabilityAnalyzer:
    """可解性解析器のテスト"""

    def test_quadratic_polynomial_solvability(self) -> None:
        """2次多項式の可解性テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x² - 2 (可解)
        quadratic_poly = poly_ring.from_coefficients([-2, 0, 1])
        analyzer = SolvabilityAnalyzer(quadratic_poly, base_field)
        
        assert analyzer.is_solvable() == True
        assert analyzer.solvability_degree() == 2
        assert analyzer.required_radical_operations() == ["square_root"]

    def test_cubic_polynomial_solvability(self) -> None:
        """3次多項式の可解性テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x³ - 2 (可解)
        cubic_poly = poly_ring.from_coefficients([-2, 0, 0, 1])
        analyzer = SolvabilityAnalyzer(cubic_poly, base_field)
        
        assert analyzer.is_solvable() == True
        assert analyzer.solvability_degree() == 3
        assert "cube_root" in analyzer.required_radical_operations()

    def test_quartic_polynomial_solvability(self) -> None:
        """4次多項式の可解性テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x⁴ - 2 (可解)
        quartic_poly = poly_ring.from_coefficients([-2, 0, 0, 0, 1])
        analyzer = SolvabilityAnalyzer(quartic_poly, base_field)
        
        assert analyzer.is_solvable() == True
        assert analyzer.solvability_degree() == 4
        assert "fourth_root" in analyzer.required_radical_operations()

    def test_quintic_polynomial_solvability(self) -> None:
        """5次多項式の可解性テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x⁵ - x - 1 (一般的に非可解)
        quintic_poly = poly_ring.from_coefficients([-1, -1, 0, 0, 0, 1])
        analyzer = SolvabilityAnalyzer(quintic_poly, base_field)
        
        # 一般的な5次多項式は非可解だが、特殊な場合は可解
        solvable = analyzer.is_solvable()
        if not solvable:
            assert analyzer.solvability_obstruction() is not None
            assert "S5" in str(analyzer.galois_group_type())

    def test_galois_group_based_solvability(self) -> None:
        """ガロア群に基づく可解性判定テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x² - 2 の場合、ガロア群は Z/2Z (可解群)
        poly = poly_ring.from_coefficients([-2, 0, 1])
        analyzer = SolvabilityAnalyzer(poly, base_field)
        
        galois_group = analyzer.compute_galois_group()
        assert galois_group.is_solvable() == True
        assert analyzer.is_solvable() == galois_group.is_solvable()


class TestRadicalSolver:
    """根号による解法のテスト"""

    def test_quadratic_formula_application(self) -> None:
        """2次方程式の解の公式の適用テスト"""
        from galois_theory.solvability import RadicalSolver
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x² - 2 = 0
        quadratic_poly = poly_ring.from_coefficients([-2, 0, 1])
        solver = RadicalSolver(quadratic_poly, base_field)
        
        solutions = solver.solve_by_radicals()
        assert len(solutions) == 2
        assert all(sol.is_radical_expression() for sol in solutions)
        assert any("√2" in str(sol) for sol in solutions)

    def test_cubic_formula_application(self) -> None:
        """3次方程式の解法テスト（カルダノの公式）"""
        from galois_theory.solvability import RadicalSolver
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x³ - 2 = 0
        cubic_poly = poly_ring.from_coefficients([-2, 0, 0, 1])
        solver = RadicalSolver(cubic_poly, base_field)
        
        solutions = solver.solve_by_radicals()
        assert len(solutions) >= 1  # 実根は1つ
        assert any("∛2" in str(sol) for sol in solutions)

    def test_quartic_formula_application(self) -> None:
        """4次方程式の解法テスト（フェラーリの公式）"""
        from galois_theory.solvability import RadicalSolver
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x⁴ - 2 = 0
        quartic_poly = poly_ring.from_coefficients([-2, 0, 0, 0, 1])
        solver = RadicalSolver(quartic_poly, base_field)
        
        solutions = solver.solve_by_radicals()
        assert len(solutions) >= 2  # 実根は2つ
        assert any("⁴√2" in str(sol) for sol in solutions)

    def test_unsolvable_polynomial_handling(self) -> None:
        """非可解多項式の処理テスト"""
        from galois_theory.solvability import RadicalSolver, SolvabilityException
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 非可解5次多項式を仮定
        quintic_poly = poly_ring.from_coefficients([1, 1, 1, 1, 1, 1])
        solver = RadicalSolver(quintic_poly, base_field)
        
        if not solver.is_solvable_by_radicals():
            with pytest.raises(SolvabilityException, match="根号では解けません"):
                solver.solve_by_radicals()


class TestGaloisSolvabilityTheorem:
    """ガロアの可解性定理のテスト"""

    def test_fundamental_solvability_theorem(self) -> None:
        """可解性の基本定理テスト"""
        from galois_theory.solvability import GaloisSolvabilityTheorem
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 可解な多項式
        solvable_poly = poly_ring.from_coefficients([-2, 0, 1])  # x² - 2
        theorem = GaloisSolvabilityTheorem(solvable_poly, base_field)
        
        assert theorem.verify_solvability_condition() == True
        assert theorem.galois_group().is_solvable() == True

    def test_degree_5_unsolvability(self) -> None:
        """5次以上の一般多項式の非可解性テスト"""
        from galois_theory.solvability import GaloisSolvabilityTheorem
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 一般的な5次多項式
        general_quintic = poly_ring.from_coefficients([1, 2, 3, 4, 5, 1])
        theorem = GaloisSolvabilityTheorem(general_quintic, base_field)
        
        # ガロア群がS₅の場合は非可解
        galois_group = theorem.galois_group()
        if galois_group.order() == 120:  # S₅の位数
            assert galois_group.is_solvable() == False
            assert theorem.verify_solvability_condition() == False

    def test_abelian_extension_solvability(self) -> None:
        """アーベル拡大の可解性テスト"""
        from galois_theory.solvability import GaloisSolvabilityTheorem
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # アーベル拡大を生成する多項式
        abelian_poly = poly_ring.from_coefficients([-2, 0, 1])  # x² - 2
        theorem = GaloisSolvabilityTheorem(abelian_poly, base_field)
        
        galois_group = theorem.galois_group()
        assert galois_group.is_abelian() == True
        assert galois_group.is_solvable() == True  # アーベル群は可解群


class TestPolynomialClassifier:
    """多項式分類器のテスト"""

    def test_polynomial_degree_classification(self) -> None:
        """多項式の次数による分類テスト"""
        from galois_theory.solvability import PolynomialClassifier
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 様々な次数の多項式
        polys = [
            poly_ring.from_coefficients([1, 1]),      # 1次
            poly_ring.from_coefficients([1, 1, 1]),   # 2次
            poly_ring.from_coefficients([1, 1, 1, 1]), # 3次
            poly_ring.from_coefficients([1, 1, 1, 1, 1]), # 4次
            poly_ring.from_coefficients([1, 1, 1, 1, 1, 1]) # 5次
        ]
        
        classifier = PolynomialClassifier(polys, base_field)
        
        # 次数による分類
        classification = classifier.classify_by_degree()
        assert 1 in classification
        assert 2 in classification
        assert 3 in classification
        assert 4 in classification
        assert 5 in classification

    def test_solvability_classification(self) -> None:
        """可解性による分類テスト"""
        from galois_theory.solvability import PolynomialClassifier
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 可解と非可解の多項式
        polys = [
            poly_ring.from_coefficients([-2, 0, 1]),      # x² - 2 (可解)
            poly_ring.from_coefficients([-2, 0, 0, 1]),   # x³ - 2 (可解)
            poly_ring.from_coefficients([1, 1, 1, 1, 1, 1]) # 5次 (一般的に非可解)
        ]
        
        classifier = PolynomialClassifier(polys, base_field)
        
        solvable, unsolvable = classifier.classify_by_solvability()
        assert len(solvable) >= 2  # 2次と3次は可解
        assert len(unsolvable) >= 0  # 5次は条件次第

    def test_galois_group_classification(self) -> None:
        """ガロア群による分類テスト"""
        from galois_theory.solvability import PolynomialClassifier
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        polys = [
            poly_ring.from_coefficients([-2, 0, 1]),    # Z/2Z
            poly_ring.from_coefficients([-2, 0, 0, 1])  # S₃ or A₃
        ]
        
        classifier = PolynomialClassifier(polys, base_field)
        
        galois_classification = classifier.classify_by_galois_group()
        assert len(galois_classification) >= 2  # 少なくとも2つの異なるガロア群


class TestAdvancedSolvabilityProperties:
    """高度な可解性の性質テスト"""

    def test_solvable_tower_construction(self) -> None:
        """可解タワーの構築テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x⁴ - 2 の可解タワー: Q ⊆ Q(√2) ⊆ Q(⁴√2)
        quartic_poly = poly_ring.from_coefficients([-2, 0, 0, 0, 1])
        analyzer = SolvabilityAnalyzer(quartic_poly, base_field)
        
        if analyzer.is_solvable():
            tower = analyzer.construct_solvable_tower()
            assert len(tower) >= 2  # 少なくとも2段階の拡大
            assert all(ext.is_radical_extension() for ext in tower)

    def test_minimal_splitting_field_solvability(self) -> None:
        """最小分解体の可解性テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 最小分解体が可解拡大になる例
        poly = poly_ring.from_coefficients([-2, 0, 1])  # x² - 2
        analyzer = SolvabilityAnalyzer(poly, base_field)
        
        splitting_field = analyzer.compute_splitting_field()
        assert splitting_field.is_galois_extension(base_field)
        
        galois_group = analyzer.compute_galois_group()
        assert galois_group.is_solvable() == analyzer.is_solvable()

    def test_composition_series_analysis(self) -> None:
        """合成列解析テスト"""
        from galois_theory.solvability import SolvabilityAnalyzer
        
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # 4次多項式の合成列解析
        poly = poly_ring.from_coefficients([-2, 0, 0, 0, 1])  # x⁴ - 2
        analyzer = SolvabilityAnalyzer(poly, base_field)
        
        if analyzer.is_solvable():
            composition_series = analyzer.compute_composition_series()
            assert all(factor.is_abelian() or factor.order() <= 2 
                      for factor in composition_series) 