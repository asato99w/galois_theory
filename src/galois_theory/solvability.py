"""
可解性判別（Solvability Determination）の実装

このモジュールは、多項式が根号で解けるかどうかを判別する機能を提供します。
ガロア理論の中核的応用として、多項式のガロア群の可解性を解析します。

主要なクラス:
- SolvabilityAnalyzer: 可解性解析器
- RadicalSolver: 根号による解法
- RadicalExpression: 根号表現
- GaloisSolvabilityTheorem: ガロアの可解性定理
- PolynomialClassifier: 多項式分類器
- SolvableTower: 可解タワー
"""

from abc import ABC, abstractmethod
from typing import Any, List, Set, Dict, Optional, Tuple, Union
from fractions import Fraction
import math

from .field import Field, RationalField
from .polynomials import Polynomial, PolynomialElement, PolynomialRing
from .field_extensions import FieldExtension, SimpleExtension, ExtensionElement, SplittingField
from .group_theory import GaloisGroup, Group, Subgroup


class SolvabilityException(Exception):
    """可解性関連のカスタム例外"""
    pass


class RadicalExpression:
    """根号表現を表すクラス"""
    
    def __init__(self, base_value: Union[int, Fraction, "RadicalExpression"], 
                 radical_type: str, radicand: Union[int, Fraction, "RadicalExpression"] = None):
        """
        根号表現を初期化
        
        Args:
            base_value: 基本値
            radical_type: 根号の種類 ('square_root', 'cube_root', 'fourth_root' など)
            radicand: 根号の中身
        """
        self.base_value = base_value
        self.radical_type = radical_type
        self.radicand = radicand
    
    def is_radical_expression(self) -> bool:
        """根号表現かどうかを判定"""
        return self.radical_type is not None
    
    def __str__(self) -> str:
        """文字列表現"""
        if self.radical_type == "square_root":
            if self.base_value == 0:
                return f"√{self.radicand}"
            else:
                return f"{self.base_value} + √{self.radicand}"
        elif self.radical_type == "cube_root":
            if self.base_value == 0:
                return f"∛{self.radicand}"
            else:
                return f"{self.base_value} + ∛{self.radicand}"
        elif self.radical_type == "fourth_root":
            if self.base_value == 0:
                return f"⁴√{self.radicand}"
            else:
                return f"{self.base_value} + ⁴√{self.radicand}"
        else:
            return str(self.base_value)
    
    def __eq__(self, other) -> bool:
        """等価性判定"""
        if not isinstance(other, RadicalExpression):
            return False
        return (self.base_value == other.base_value and 
                self.radical_type == other.radical_type and
                self.radicand == other.radicand)


class SolvabilityAnalyzer:
    """多項式の可解性を解析するクラス"""
    
    def __init__(self, polynomial: PolynomialElement, base_field: Field):
        """
        可解性解析器を初期化
        
        Args:
            polynomial: 解析対象の多項式
            base_field: 基底体
        """
        self.polynomial = polynomial
        self.base_field = base_field
        self._galois_group = None
        self._splitting_field = None
    
    def is_solvable(self) -> bool:
        """多項式が根号で解けるかどうかを判定"""
        try:
            galois_group = self.compute_galois_group()
            return galois_group.is_solvable()
        except Exception:
            # エラーが発生した場合は保守的にFalseを返す
            return False
    
    def solvability_degree(self) -> int:
        """可解性の次数（多項式の次数）"""
        return self.polynomial.degree()
    
    def required_radical_operations(self) -> List[str]:
        """必要な根号演算のリスト"""
        degree = self.polynomial.degree()
        operations = []
        
        if degree >= 2:
            operations.append("square_root")
        if degree >= 3:
            operations.append("cube_root")
        if degree >= 4:
            operations.append("fourth_root")
        if degree >= 5:
            operations.append("fifth_root")
        
        return operations
    
    def compute_galois_group(self) -> GaloisGroup:
        """ガロア群を計算"""
        if self._galois_group is None:
            self._galois_group = GaloisGroup.from_polynomial(self.polynomial, self.base_field)
        return self._galois_group
    
    def compute_splitting_field(self) -> FieldExtension:
        """分解体を計算"""
        if self._splitting_field is None:
            self._splitting_field = SplittingField.construct(self.polynomial.polynomial, self.base_field)
        return self._splitting_field
    
    def solvability_obstruction(self) -> Optional[str]:
        """可解性の障害を返す"""
        if self.is_solvable():
            return None
        
        galois_group = self.compute_galois_group()
        if galois_group.order() == 120:  # S₅
            return "対称群S₅は可解群ではありません"
        elif galois_group.order() == 60:  # A₅
            return "交代群A₅は可解群ではありません"
        else:
            return "ガロア群が可解群ではありません"
    
    def galois_group_type(self) -> str:
        """ガロア群の型を返す"""
        galois_group = self.compute_galois_group()
        order = galois_group.order()
        
        if order == 1:
            return "自明群"
        elif order == 2:
            return "Z/2Z"
        elif order == 3:
            return "Z/3Z"
        elif order == 4:
            if galois_group.is_cyclic():
                return "Z/4Z"
            else:
                return "Klein群V4"
        elif order == 6:
            if galois_group.is_abelian():
                return "Z/6Z"
            else:
                return "S3"
        elif order == 8:
            return "二面体群D4"
        elif order == 12:
            return "交代群A4"
        elif order == 24:
            return "対称群S4"
        elif order == 60:
            return "A5"
        elif order == 120:
            return "S5"
        else:
            return f"位数{order}の群"
    
    def construct_solvable_tower(self) -> List["RadicalExtension"]:
        """可解タワーを構築"""
        if not self.is_solvable():
            raise SolvabilityException("多項式は可解ではありません")
        
        # 簡易実装：次数に基づく可解タワー
        tower = []
        degree = self.polynomial.degree()
        
        if degree >= 2:
            # Q(√2) など
            sqrt_extension = RadicalExtension(self.base_field, 2, 2)
            tower.append(sqrt_extension)
        
        if degree >= 4:
            # Q(√2, ⁴√2) など
            fourth_root_extension = RadicalExtension(sqrt_extension if tower else self.base_field, 2, 4)
            tower.append(fourth_root_extension)
        
        return tower
    
    def compute_composition_series(self) -> List[Group]:
        """ガロア群の合成列を計算"""
        galois_group = self.compute_galois_group()
        
        # 簡易実装：部分群による合成列の近似
        subgroups = galois_group.all_subgroups()
        subgroups.sort(key=lambda h: h.order())
        
        composition_factors = []
        for i in range(len(subgroups) - 1):
            # 商群の近似（実際には完全な実装が必要）
            factor_order = subgroups[i+1].order() // subgroups[i].order()
            if factor_order > 1:
                # 商群を近似的に構築
                if factor_order == 2:
                    from .group_theory import CyclicGroup
                    composition_factors.append(CyclicGroup(2))
                elif factor_order == 3:
                    from .group_theory import CyclicGroup
                    composition_factors.append(CyclicGroup(3))
        
        return composition_factors


class RadicalSolver:
    """根号による多項式の解法を実装するクラス"""
    
    def __init__(self, polynomial: PolynomialElement, base_field: Field):
        """
        根号解法器を初期化
        
        Args:
            polynomial: 解く多項式
            base_field: 基底体
        """
        self.polynomial = polynomial
        self.base_field = base_field
        self.analyzer = SolvabilityAnalyzer(polynomial, base_field)
    
    def is_solvable_by_radicals(self) -> bool:
        """根号で解けるかどうかを判定"""
        return self.analyzer.is_solvable()
    
    def solve_by_radicals(self) -> List[RadicalExpression]:
        """根号による解を計算"""
        if not self.is_solvable_by_radicals():
            raise SolvabilityException("この多項式は根号では解けません")
        
        degree = self.polynomial.degree()
        
        if degree == 1:
            return self._solve_linear()
        elif degree == 2:
            return self._solve_quadratic()
        elif degree == 3:
            return self._solve_cubic()
        elif degree == 4:
            return self._solve_quartic()
        else:
            raise SolvabilityException(f"{degree}次多項式の根号解法は未実装")
    
    def _solve_linear(self) -> List[RadicalExpression]:
        """1次方程式の解法"""
        # ax + b = 0 → x = -b/a
        coeffs = self.polynomial.polynomial.coefficients
        b = coeffs[0] if len(coeffs) > 0 else 0
        a = coeffs[1] if len(coeffs) > 1 else 1
        
        solution = -b / a
        return [RadicalExpression(solution, None)]
    
    def _solve_quadratic(self) -> List[RadicalExpression]:
        """2次方程式の解法（解の公式）"""
        # ax² + bx + c = 0
        coeffs = self.polynomial.polynomial.coefficients
        c = coeffs[0] if len(coeffs) > 0 else 0
        b = coeffs[1] if len(coeffs) > 1 else 0
        a = coeffs[2] if len(coeffs) > 2 else 1
        
        # 判別式 D = b² - 4ac
        discriminant = b * b - 4 * a * c
        
        # x = (-b ± √D) / 2a
        solution1 = RadicalExpression(-b / (2 * a), "square_root", discriminant / (4 * a * a))
        solution2 = RadicalExpression(-b / (2 * a), "square_root", -discriminant / (4 * a * a))
        
        return [solution1, solution2]
    
    def _solve_cubic(self) -> List[RadicalExpression]:
        """3次方程式の解法（カルダノの公式の簡易版）"""
        # x³ + px + q = 0 の形に変形してから解く
        # 簡易実装：x³ - a = 0 の場合のみ
        coeffs = self.polynomial.polynomial.coefficients
        
        if len(coeffs) >= 4 and coeffs[1] == 0 and coeffs[2] == 0:
            # x³ + constant = 0 の形
            constant = coeffs[0]
            
            # x = ∛(-constant)
            solution = RadicalExpression(0, "cube_root", -constant)
            return [solution]
        
        # より一般的な場合は後で実装
        raise SolvabilityException("一般的な3次方程式の解法は未実装")
    
    def _solve_quartic(self) -> List[RadicalExpression]:
        """4次方程式の解法（フェラーリの公式の簡易版）"""
        # 簡易実装：x⁴ - a = 0 の場合のみ
        coeffs = self.polynomial.polynomial.coefficients
        
        if len(coeffs) >= 5 and all(coeffs[i] == 0 for i in range(1, 4)):
            # x⁴ + constant = 0 の形
            constant = coeffs[0]
            
            # x = ±⁴√(-constant), ±i⁴√(-constant)
            solution1 = RadicalExpression(0, "fourth_root", -constant)
            solution2 = RadicalExpression(0, "fourth_root", constant)  # 負の解
            
            return [solution1, solution2]
        
        # より一般的な場合は後で実装
        raise SolvabilityException("一般的な4次方程式の解法は未実装")


class GaloisSolvabilityTheorem:
    """ガロアの可解性定理を実装するクラス"""
    
    def __init__(self, polynomial: PolynomialElement, base_field: Field):
        """
        ガロア可解性定理を初期化
        
        Args:
            polynomial: 対象多項式
            base_field: 基底体
        """
        self.polynomial = polynomial
        self.base_field = base_field
        self.analyzer = SolvabilityAnalyzer(polynomial, base_field)
    
    def verify_solvability_condition(self) -> bool:
        """可解性の条件を検証"""
        # ガロア群が可解群であることが必要十分条件
        galois_group = self.galois_group()
        return galois_group.is_solvable()
    
    def galois_group(self) -> GaloisGroup:
        """ガロア群を取得"""
        return self.analyzer.compute_galois_group()
    
    def fundamental_theorem_statement(self) -> str:
        """基本定理の陳述を返す"""
        if self.verify_solvability_condition():
            return ("多項式が根号で解ける ⟺ ガロア群が可解群である: "
                   f"この多項式のガロア群{self.analyzer.galois_group_type()}は可解群です")
        else:
            return ("多項式が根号で解ける ⟺ ガロア群が可解群である: "
                   f"この多項式のガロア群{self.analyzer.galois_group_type()}は可解群ではありません")


class PolynomialClassifier:
    """多項式を可解性などで分類するクラス"""
    
    def __init__(self, polynomials: List[PolynomialElement], base_field: Field):
        """
        多項式分類器を初期化
        
        Args:
            polynomials: 分類対象の多項式リスト
            base_field: 基底体
        """
        self.polynomials = polynomials
        self.base_field = base_field
    
    def classify_by_degree(self) -> Dict[int, List[PolynomialElement]]:
        """次数による分類"""
        classification = {}
        
        for poly in self.polynomials:
            degree = poly.degree()
            if degree not in classification:
                classification[degree] = []
            classification[degree].append(poly)
        
        return classification
    
    def classify_by_solvability(self) -> Tuple[List[PolynomialElement], List[PolynomialElement]]:
        """可解性による分類"""
        solvable = []
        unsolvable = []
        
        for poly in self.polynomials:
            analyzer = SolvabilityAnalyzer(poly, self.base_field)
            if analyzer.is_solvable():
                solvable.append(poly)
            else:
                unsolvable.append(poly)
        
        return solvable, unsolvable
    
    def classify_by_galois_group(self) -> Dict[str, List[PolynomialElement]]:
        """ガロア群による分類"""
        classification = {}
        
        for poly in self.polynomials:
            analyzer = SolvabilityAnalyzer(poly, self.base_field)
            group_type = analyzer.galois_group_type()
            
            if group_type not in classification:
                classification[group_type] = []
            classification[group_type].append(poly)
        
        return classification


class RadicalExtension(FieldExtension):
    """根号拡大を表すクラス"""
    
    def __init__(self, base_field: Field, radicand: Union[int, Fraction], root_degree: int):
        """
        根号拡大を初期化
        
        Args:
            base_field: 基底体
            radicand: 根号の中身
            root_degree: 根の次数（2なら平方根、3なら立方根）
        """
        self.radicand = radicand
        self.root_degree = root_degree
        
        name = f"{base_field.name}(√[{root_degree}]{radicand})"
        super().__init__(base_field, name)
    
    def degree(self) -> int:
        """拡大次数"""
        return self.root_degree
    
    def generator(self) -> ExtensionElement:
        """生成元（根号）"""
        # 簡易実装
        coeffs = [0] * self.root_degree
        if self.root_degree > 1:
            coeffs[1] = 1  # 根号の係数
        else:
            coeffs[0] = 1
        
        # ExtensionElementを使用するには実際のSimpleExtensionが必要
        # ここでは概念的な実装
        return ExtensionElement(coeffs, self)
    
    def is_radical_extension(self) -> bool:
        """根号拡大かどうかを判定"""
        return True


class SolvableTower:
    """可解タワーを表すクラス"""
    
    def __init__(self, base_field: Field):
        """
        可解タワーを初期化
        
        Args:
            base_field: 基底体
        """
        self.base_field = base_field
        self.extensions = []
    
    def add_radical_extension(self, radicand: Union[int, Fraction], root_degree: int) -> RadicalExtension:
        """根号拡大を追加"""
        current_field = self.extensions[-1] if self.extensions else self.base_field
        extension = RadicalExtension(current_field, radicand, root_degree)
        self.extensions.append(extension)
        return extension
    
    def total_degree(self) -> int:
        """全体の拡大次数"""
        total = 1
        for ext in self.extensions:
            total *= ext.degree()
        return total
    
    def is_solvable_tower(self) -> bool:
        """可解タワーかどうかを判定"""
        # 全ての拡大が根号拡大であることを確認
        return all(ext.is_radical_extension() for ext in self.extensions)
    
    def contains_polynomial_roots(self, polynomial: PolynomialElement) -> bool:
        """多項式の根を含むかどうかを判定"""
        # 最上位の拡大で多項式が完全に分解するかをチェック
        if not self.extensions:
            return False
        
        top_field = self.extensions[-1]
        try:
            roots = top_field.solve_polynomial(polynomial.polynomial)
            return len(roots) == polynomial.degree()
        except:
            return False


# __init__.py で使用するためのエクスポート
__all__ = [
    'SolvabilityException', 'SolvabilityAnalyzer', 'RadicalSolver', 
    'RadicalExpression', 'GaloisSolvabilityTheorem', 'PolynomialClassifier',
    'RadicalExtension', 'SolvableTower'
] 