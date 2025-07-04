"""
多項式環（Polynomial Ring）の実装

このモジュールは、ガロア理論で使用される多項式環の実装を提供します。
多項式環 R[x] は、環 R 上の多項式からなる環です。

主要なクラス:
- Polynomial: 多項式を表すクラス
- PolynomialRing: 多項式環を表すクラス
- PolynomialElement: 多項式環の要素を表すクラス
- PolynomialException: 多項式操作のカスタム例外
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, List, Union, Tuple, Optional
import copy

from .ring import Ring, RingElement
from .field import Field, FieldElement


class PolynomialException(Exception):
    """多項式操作のカスタム例外"""

    pass


class Polynomial:
    """
    多項式を表すクラス

    係数は定数項から高次の項へと順番に格納されます。
    例: f(x) = 3x^2 + 2x + 1 の場合、coefficients = [1, 2, 3]
    """

    def __init__(
        self,
        coefficients: List[Union[int, Fraction, Any]],
        base_ring: Union[Ring, Field],
    ):
        """
        多項式を初期化

        Args:
            coefficients: 係数のリスト（定数項から高次へ）
            base_ring: 係数が属する環または体
        """
        self.base_ring = base_ring

        if not coefficients:
            # 空のリストの場合は零多項式
            self.coefficients = [self._convert_to_base_element(0)]
        else:
            # 係数を基底環の要素に変換
            converted_coeffs = [
                self._convert_to_base_element(coeff) for coeff in coefficients
            ]

            # 先頭（高次）の零係数を除去
            while len(converted_coeffs) > 1 and self._is_zero_element(
                converted_coeffs[-1]
            ):
                converted_coeffs.pop()

            self.coefficients = converted_coeffs

    def _convert_to_base_element(self, value: Any) -> Any:
        """値を基底環の要素の値に変換"""
        if isinstance(self.base_ring, Field):
            if isinstance(value, FieldElement):
                return value.value
            elif isinstance(value, (int, Fraction)):
                return Fraction(value)
            else:
                return Fraction(value) if value is not None else Fraction(0)
        else:  # Ring
            if isinstance(value, RingElement):
                return value.value
            else:
                return value if value is not None else 0

    def _is_zero_element(self, value: Any) -> bool:
        """値が零要素かどうかを判定"""
        if isinstance(self.base_ring, Field):
            return bool(value == Fraction(0))
        else:
            return bool(value == 0)

    def degree(self) -> int:
        """多項式の次数を取得"""
        if self.is_zero():
            return 0  # 零多項式の次数は0とする
        return len(self.coefficients) - 1

    def is_zero(self) -> bool:
        """零多項式かどうかを判定"""
        return len(self.coefficients) == 1 and self._is_zero_element(
            self.coefficients[0]
        )

    def is_monic(self) -> bool:
        """モニック多項式（最高次係数が1）かどうかを判定"""
        if self.is_zero():
            return False
        leading_coeff = self.coefficients[-1]
        if isinstance(self.base_ring, Field):
            return bool(leading_coeff == Fraction(1))
        else:
            return bool(leading_coeff == 1)

    def leading_coefficient(self) -> Any:
        """最高次係数を取得"""
        if self.is_zero():
            return self._convert_to_base_element(0)
        return self.coefficients[-1]

    def __eq__(self, other: object) -> bool:
        """多項式の等価性判定"""
        if not isinstance(other, Polynomial):
            return False
        return (
            self.coefficients == other.coefficients
            and self.base_ring == other.base_ring
        )

    def __str__(self) -> str:
        """多項式の文字列表現"""
        if self.is_zero():
            return "0"

        terms = []
        for i, coeff in enumerate(self.coefficients):
            if self._is_zero_element(coeff):
                continue

            # 係数の表示
            if i == 0:  # 定数項
                terms.append(str(coeff))
            elif i == 1:  # 1次項
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff}*x")
            else:  # 高次項
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff}*x^{i}")

        if not terms:
            return "0"

        # 高次項から低次項の順で結合
        terms.reverse()
        result = terms[0]

        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result

    def __repr__(self) -> str:
        """多項式のデバッグ用文字列表現"""
        return f"Polynomial({self.coefficients}, {self.base_ring})"

    def evaluate(
        self, x: Union[FieldElement, RingElement]
    ) -> Union[FieldElement, RingElement]:
        """多項式の値を計算（ホーナー法）"""
        if self.is_zero():
            if isinstance(x, FieldElement):
                return x.field.zero()
            else:
                return x.ring.zero()

        # ホーナー法で計算 - 型安全な実装
        if isinstance(x, FieldElement):
            # 体要素の場合
            result: Optional[FieldElement] = None
            for i in range(len(self.coefficients) - 1, -1, -1):
                coeff_value = self.coefficients[i]
                
                # 有限体の場合、Fractionを整数に変換
                if hasattr(x.field, 'characteristic'):
                    if hasattr(coeff_value, 'numerator'):
                        # Fractionの場合
                        coeff_int = int(coeff_value.numerator) % x.field.characteristic
                    else:
                        coeff_int = int(coeff_value) % x.field.characteristic
                    coeff_elem = x.field.element(coeff_int)
                else:
                    coeff_elem = x.field.element(coeff_value)

                if result is None:
                    result = coeff_elem
                else:
                    result = result * x + coeff_elem

            return result if result is not None else x.field.zero()
        else:
            # 環要素の場合 - RingElementとして処理
            result_ring: Optional[RingElement] = None
            for i in range(len(self.coefficients) - 1, -1, -1):
                coeff_value = self.coefficients[i]
                coeff_elem = x.ring.element(coeff_value)

                if result_ring is None:
                    result_ring = coeff_elem
                else:
                    result_ring = result_ring * x + coeff_elem

            return result_ring if result_ring is not None else x.ring.zero()

    def __add__(self, other: "Polynomial") -> "Polynomial":
        """多項式の加法"""
        if self.base_ring != other.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        # より長い方の長さに合わせる
        max_len = max(len(self.coefficients), len(other.coefficients))

        result_coeffs = []
        for i in range(max_len):
            a = (
                self.coefficients[i]
                if i < len(self.coefficients)
                else self._convert_to_base_element(0)
            )
            b = (
                other.coefficients[i]
                if i < len(other.coefficients)
                else self._convert_to_base_element(0)
            )
            result_coeffs.append(a + b)

        return Polynomial(result_coeffs, self.base_ring)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        """多項式の減法"""
        if self.base_ring != other.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        max_len = max(len(self.coefficients), len(other.coefficients))

        result_coeffs = []
        for i in range(max_len):
            a = (
                self.coefficients[i]
                if i < len(self.coefficients)
                else self._convert_to_base_element(0)
            )
            b = (
                other.coefficients[i]
                if i < len(other.coefficients)
                else self._convert_to_base_element(0)
            )
            result_coeffs.append(a - b)

        return Polynomial(result_coeffs, self.base_ring)

    def __neg__(self) -> "Polynomial":
        """多項式の加法逆元"""
        result_coeffs = [-coeff for coeff in self.coefficients]
        return Polynomial(result_coeffs, self.base_ring)

    def scalar_multiply(self, scalar: Union[FieldElement, RingElement]) -> "Polynomial":
        """多項式のスカラー倍"""
        scalar_value = scalar.value
        result_coeffs = [coeff * scalar_value for coeff in self.coefficients]
        return Polynomial(result_coeffs, self.base_ring)

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        """多項式の乗法"""
        if self.base_ring != other.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        if self.is_zero() or other.is_zero():
            return Polynomial([0], self.base_ring)

        # 結果の係数配列のサイズ
        result_degree = self.degree() + other.degree()
        result_coeffs = [self._convert_to_base_element(0)] * (result_degree + 1)

        # 畳み込み
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result_coeffs[i + j] += a * b

        return Polynomial(result_coeffs, self.base_ring)

    def __pow__(self, exponent: int) -> "Polynomial":
        """多項式の冪乗"""
        if exponent < 0:
            raise ValueError("負の指数での冪乗はサポートされていません")

        if exponent == 0:
            return Polynomial([1], self.base_ring)

        if exponent == 1:
            return copy.deepcopy(self)

        # 繰り返し二乗法
        result = Polynomial([1], self.base_ring)
        base = copy.deepcopy(self)

        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base
            base = base * base
            exponent //= 2

        return result

    def divide(self, divisor: "Polynomial") -> Tuple["Polynomial", "Polynomial"]:
        """多項式の除法（商と余りを返す）"""
        if divisor.is_zero():
            raise PolynomialException("零多項式による除法はできません")

        if self.base_ring != divisor.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        if self.degree() < divisor.degree():
            return Polynomial([0], self.base_ring), copy.deepcopy(self)

        # 多項式の長除法
        dividend = copy.deepcopy(self)
        quotient_coeffs = []

        while dividend.degree() >= divisor.degree() and not dividend.is_zero():
            # 最高次項の比
            lead_coeff_dividend = dividend.leading_coefficient()
            lead_coeff_divisor = divisor.leading_coefficient()

            # 体の場合は除法を使用
            if isinstance(self.base_ring, Field):
                coeff_ratio = lead_coeff_dividend / lead_coeff_divisor
            else:
                # 環の場合は整除が必要（実装を簡略化）
                if lead_coeff_dividend % lead_coeff_divisor != 0:
                    break
                coeff_ratio = lead_coeff_dividend // lead_coeff_divisor

            degree_diff = dividend.degree() - divisor.degree()

            # 商の項を作成
            term_coeffs = [self._convert_to_base_element(0)] * (degree_diff + 1)
            term_coeffs[degree_diff] = coeff_ratio
            term = Polynomial(term_coeffs, self.base_ring)

            quotient_coeffs.append((degree_diff, coeff_ratio))

            # 被除数から除数×項を引く
            dividend = dividend - (divisor * term)

        # 商を構築
        if not quotient_coeffs:
            quotient = Polynomial([0], self.base_ring)
        else:
            max_degree = max(deg for deg, _ in quotient_coeffs)
            q_coeffs = [self._convert_to_base_element(0)] * (max_degree + 1)
            for deg, coeff in quotient_coeffs:
                q_coeffs[deg] = coeff
            quotient = Polynomial(q_coeffs, self.base_ring)

        return quotient, dividend

    def gcd(self, other: "Polynomial") -> "Polynomial":
        """多項式の最大公約数（ユークリッドの互除法）"""
        if self.base_ring != other.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        a = copy.deepcopy(self)
        b = copy.deepcopy(other)

        while not b.is_zero():
            _, remainder = a.divide(b)
            a, b = b, remainder

        # モニック多項式に正規化
        if not a.is_zero() and not a.is_monic():
            leading_coeff = a.leading_coefficient()
            if isinstance(self.base_ring, Field):
                # 体の場合は最高次係数で割る
                inv_leading = self._convert_to_base_element(1) / leading_coeff
                normalized_coeffs = [coeff * inv_leading for coeff in a.coefficients]
                a = Polynomial(normalized_coeffs, self.base_ring)

        return a

    def derivative(self) -> "Polynomial":
        """多項式の形式的微分"""
        if self.is_zero() or self.degree() == 0:
            return Polynomial([0], self.base_ring)

        deriv_coeffs = []
        for i in range(1, len(self.coefficients)):
            deriv_coeffs.append(self.coefficients[i] * i)

        return Polynomial(deriv_coeffs, self.base_ring)

    def compose(self, other: "Polynomial") -> "Polynomial":
        """多項式の合成 self(other(x))"""
        if self.base_ring != other.base_ring:
            raise ValueError("異なる基底環の多項式同士の演算はできません")

        if self.is_zero():
            return Polynomial([0], self.base_ring)

        # ホーナー法で合成を計算
        result = Polynomial([self.coefficients[-1]], self.base_ring)

        for i in range(len(self.coefficients) - 2, -1, -1):
            result = result * other + Polynomial([self.coefficients[i]], self.base_ring)

        return result

    def content(self) -> Any:
        """多項式の内容（係数の最大公約数）を取得"""
        if self.is_zero():
            return self._convert_to_base_element(0)
        
        if isinstance(self.base_ring, Field):
            # 体の場合、内容は先頭係数の逆数を含む
            from math import gcd
            from fractions import Fraction
            
            # 分数の場合のGCD計算
            numerators = []
            denominators = []
            
            for coeff in self.coefficients:
                if not self._is_zero_element(coeff):
                    frac = Fraction(coeff)
                    numerators.append(abs(frac.numerator))
                    denominators.append(frac.denominator)
            
            if not numerators:
                return Fraction(0)
                
            # 分子のGCDと分母のLCM
            num_gcd = numerators[0]
            for num in numerators[1:]:
                num_gcd = gcd(num_gcd, num)
                
            den_lcm = denominators[0]
            for den in denominators[1:]:
                den_lcm = (den_lcm * den) // gcd(den_lcm, den)
                
            return Fraction(num_gcd, den_lcm)
        else:
            # 環の場合
            from math import gcd
            
            non_zero_coeffs = [coeff for coeff in self.coefficients 
                             if not self._is_zero_element(coeff)]
            if not non_zero_coeffs:
                return 0
                
            result = abs(non_zero_coeffs[0])
            for coeff in non_zero_coeffs[1:]:
                result = gcd(result, abs(coeff))
                if result == 1:
                    break
                    
            return result

    def primitive_part(self) -> "Polynomial":
        """多項式の原始部分（内容で割った多項式）を取得"""
        content_val = self.content()
        
        if self._is_zero_element(content_val) or content_val == 1:
            return copy.deepcopy(self)
            
        if isinstance(self.base_ring, Field):
            # 体の場合は除法
            primitive_coeffs = [coeff / content_val for coeff in self.coefficients]
        else:
            # 環の場合は整除
            primitive_coeffs = [coeff // content_val for coeff in self.coefficients]
            
        return Polynomial(primitive_coeffs, self.base_ring)

    def is_square_free(self) -> bool:
        """多項式が平方因子を持たないかを判定"""
        if self.is_zero() or self.degree() <= 1:
            return True
            
        # f と f' の最大公約数が1かを確認
        derivative = self.derivative()
        if derivative.is_zero():
            return False
            
        gcd_result = self.gcd(derivative)
        return gcd_result.degree() == 0

    def has_rational_root(self) -> bool:
        """有理根を持つかを判定（有理根定理を使用）"""
        if not isinstance(self.base_ring, Field):
            return False
            
        if self.degree() <= 1:
            return True
            
        # 有理根定理: p/q が根なら、pは定数項の約数、qは最高次係数の約数
        from fractions import Fraction
        import math
        
        constant_term = self.coefficients[0]
        leading_coeff = self.coefficients[-1]
        
        # 分数を整数に変換して約数を求める
        if isinstance(constant_term, Fraction):
            const_num = abs(constant_term.numerator)
            const_den = constant_term.denominator
        else:
            const_num = abs(int(constant_term))
            const_den = 1
            
        if isinstance(leading_coeff, Fraction):
            lead_num = abs(leading_coeff.numerator)
            lead_den = leading_coeff.denominator
        else:
            lead_num = abs(int(leading_coeff))
            lead_den = 1
        
        # 約数を求める
        def get_divisors(n):
            if n == 0:
                return [1]
            divisors = []
            for i in range(1, int(math.sqrt(abs(n))) + 1):
                if n % i == 0:
                    divisors.extend([i, n // i])
            return list(set(divisors))
        
        p_divisors = get_divisors(const_num)
        q_divisors = get_divisors(lead_num)
        
        # 可能な有理根をテスト
        for p in p_divisors:
            for q in q_divisors:
                for sign in [1, -1]:
                    candidate = Fraction(sign * p * const_den, q * lead_den)
                    if isinstance(self.base_ring, Field):
                        test_value = self.base_ring.element(candidate)
                        if self.evaluate(test_value) == self.base_ring.zero():
                            return True
        
        return False

    def _try_factor_by_roots(self) -> List["Polynomial"]:
        """有理根による因数分解を試行"""
        if not self.has_rational_root():
            return [copy.deepcopy(self)]
            
        factors = []
        remaining = copy.deepcopy(self)
        
        # 有理根定理による根の探索
        from fractions import Fraction
        import math
        
        while remaining.degree() > 1 and remaining.has_rational_root():
            # 根を見つける
            constant_term = remaining.coefficients[0]
            leading_coeff = remaining.coefficients[-1]
            
            if isinstance(constant_term, Fraction):
                const_num = abs(constant_term.numerator)
                const_den = constant_term.denominator
            else:
                const_num = abs(int(constant_term))
                const_den = 1
                
            if isinstance(leading_coeff, Fraction):
                lead_num = abs(leading_coeff.numerator)
                lead_den = leading_coeff.denominator
            else:
                lead_num = abs(int(leading_coeff))
                lead_den = 1
            
            def get_divisors(n):
                if n == 0:
                    return [1]
                divisors = []
                for i in range(1, int(math.sqrt(abs(n))) + 1):
                    if n % i == 0:
                        divisors.extend([i, n // i])
                return list(set(divisors))
            
            p_divisors = get_divisors(const_num)
            q_divisors = get_divisors(lead_num)
            
            root_found = False
            for p in p_divisors:
                for q in q_divisors:
                    for sign in [1, -1]:
                        candidate = Fraction(sign * p * const_den, q * lead_den)
                        test_value = self.base_ring.element(candidate)
                        if remaining.evaluate(test_value) == self.base_ring.zero():
                            # (x - root) で除法
                            linear_factor = Polynomial([-candidate, 1], self.base_ring)
                            factors.append(linear_factor)
                            quotient, remainder = remaining.divide(linear_factor)
                            if not remainder.is_zero():
                                # 数値誤差があれば諦める
                                break
                            remaining = quotient
                            root_found = True
                            break
                    if root_found:
                        break
                if root_found:
                    break
            
            if not root_found:
                break
        
        if remaining.degree() > 0:
            factors.append(remaining)
            
        return factors

    def is_irreducible(self) -> bool:
        """多項式が既約かを判定"""
        # 基本的なケース
        if self.is_zero():
            return False
            
        degree = self.degree()
        
        # 定数多項式は既約ではない
        if degree == 0:
            return False
            
        # 1次多項式は常に既約
        if degree == 1:
            return True
        
        # 有限体上での判定
        if hasattr(self.base_ring, 'characteristic'):
            return self._is_irreducible_finite_field()
            
        # 平方因子を持つ場合は可約
        if not self.is_square_free():
            return False
            
        # 原始部分での判定（内容が1でない場合）
        content_val = self.content()
        if not (self._is_zero_element(content_val) or content_val == 1):
            primitive = self.primitive_part()
            return primitive.is_irreducible()
        
        # 2次多項式の場合：判別式による判定
        if degree == 2:
            return self._is_irreducible_quadratic()
            
        # 3次多項式の場合：有理根を持たなければ既約
        if degree == 3:
            return not self.has_rational_root()
        
        # 高次多項式：部分的な因数分解を試行
        if degree >= 4:
            return self._is_irreducible_higher_degree()
        
        return True

    def _is_irreducible_quadratic(self) -> bool:
        """2次多項式の既約性を判定"""
        if self.degree() != 2:
            return False
            
        # ax^2 + bx + c の形
        a = self.coefficients[2]
        b = self.coefficients[1] if len(self.coefficients) > 1 else self._convert_to_base_element(0)
        c = self.coefficients[0]
        
        # 判別式 D = b^2 - 4ac
        discriminant = b * b - 4 * a * c
        
        if isinstance(self.base_ring, Field):
            from fractions import Fraction
            
            # 有理数体上では、判別式が負または完全平方数でなければ既約
            if isinstance(discriminant, Fraction):
                # 判別式が負の場合は既約
                if discriminant < 0:
                    return True
                    
                # 判別式が非負の場合、平方根が有理数かを確認
                import math
                sqrt_num = math.isqrt(abs(discriminant.numerator))
                sqrt_den = math.isqrt(discriminant.denominator)
                
                is_perfect_square = (sqrt_num * sqrt_num == abs(discriminant.numerator) and
                                   sqrt_den * sqrt_den == discriminant.denominator)
                
                return not is_perfect_square
            else:
                # 整数の場合
                if discriminant < 0:
                    return True
                    
                import math
                sqrt_val = math.isqrt(abs(int(discriminant)))
                return sqrt_val * sqrt_val != abs(int(discriminant))
        
        return True

    def _is_irreducible_higher_degree(self) -> bool:
        """高次多項式の既約性を判定"""
        # 有理根による因数分解を試行
        factors = self._try_factor_by_roots()
        
        # 因数分解できた場合は可約
        if len(factors) > 1:
            return False
            
        # 部分的なアイゼンシュタイン判定法
        if self._satisfies_eisenstein_criterion():
            return True
            
        # それ以外の場合は、簡単な因数分解チェック
        # より高度な判定法（Berlekamp-Zassenhausアルゴリズムなど）は今後実装
        return self._basic_factorization_check()

    def _satisfies_eisenstein_criterion(self) -> bool:
        """アイゼンシュタインの既約判定法をチェック"""
        if not isinstance(self.base_ring, Field):
            return False
            
        from fractions import Fraction
        
        # 小さい素数でテスト
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for p in primes:
            # 最高次係数がpで割り切れない
            leading_coeff = self.coefficients[-1]
            if isinstance(leading_coeff, Fraction):
                if leading_coeff.numerator % p == 0:
                    continue
            else:
                if int(leading_coeff) % p == 0:
                    continue
                    
            # 他の係数がpで割り切れる
            all_divisible = True
            for i in range(len(self.coefficients) - 1):
                coeff = self.coefficients[i]
                if isinstance(coeff, Fraction):
                    if coeff.numerator % p != 0:
                        all_divisible = False
                        break
                else:
                    if int(coeff) % p != 0:
                        all_divisible = False
                        break
                        
            if not all_divisible:
                continue
                
            # 定数項がp^2で割り切れない
            constant_term = self.coefficients[0]
            if isinstance(constant_term, Fraction):
                if constant_term.numerator % (p * p) == 0:
                    continue
            else:
                if int(constant_term) % (p * p) == 0:
                    continue
                    
            return True
            
        return False

    def _basic_factorization_check(self) -> bool:
        """基本的な因数分解チェック"""
        # 次数が小さい因子を持つかチェック
        degree = self.degree()
        
        # 次数1の因子をチェック（有理根テストと同じ）
        if self.has_rational_root():
            return False
            
        # 次数2の因子を持つかの簡単なチェック
        if degree >= 4:
            # より高度な判定は今後実装
            pass
            
        return True

    def _is_irreducible_finite_field(self) -> bool:
        """有限体上での既約性判定"""
        field = self.base_ring
        p = field.characteristic
        n = self.degree()
        
        # 基本的なケース
        if n <= 1:
            return n == 1
        
        # 平方因子を持つ場合は可約
        if not self.is_square_free():
            return False
        
        # 2次多項式の場合：有限体での根の存在をチェック
        if n == 2:
            return self._is_irreducible_quadratic_finite_field()
        
        # 3次以上の場合：簡単な根のチェックから始める
        if self._has_root_in_finite_field():
            return False
        
        # より高次の場合は、今後Rabinの既約性テストを実装
        # 現在は簡単な判定のみ
        return True

    def _is_irreducible_quadratic_finite_field(self) -> bool:
        """有限体上での2次多項式の既約性判定"""
        field = self.base_ring
        p = field.characteristic
        
        # ax^2 + bx + c の形
        a = self.coefficients[2].value if hasattr(self.coefficients[2], 'value') else int(self.coefficients[2])
        b = (self.coefficients[1].value if hasattr(self.coefficients[1], 'value') else int(self.coefficients[1])) if len(self.coefficients) > 2 else 0
        c = self.coefficients[0].value if hasattr(self.coefficients[0], 'value') else int(self.coefficients[0])
        
        # 正規化（mod p）
        a = a % p
        b = b % p
        c = c % p
        
        # 有限体の全ての要素で評価して根があるかチェック
        for x in range(p):
            value = (a * x * x + b * x + c) % p
            if value == 0:
                return False  # 根があるので可約
        
        return True  # 根がないので既約

    def _has_root_in_finite_field(self) -> bool:
        """有限体で根を持つかをチェック"""
        field = self.base_ring
        p = field.characteristic
        
        # 有限体の全ての要素で評価
        for x in range(p):
            try:
                field_element = field.element(x)
                value = self.evaluate(field_element)
                if value == field.zero():
                    return True
            except:
                # 評価でエラーが発生した場合はスキップ
                continue
        
        return False


class PolynomialElement:
    """多項式環の要素を表すクラス"""

    def __init__(self, polynomial: Polynomial, ring: "PolynomialRing"):
        """
        多項式環の要素を初期化

        Args:
            polynomial: 多項式
            ring: 所属する多項式環
        """
        self.polynomial = polynomial
        self.ring = ring

    def __eq__(self, other: object) -> bool:
        """等価性の判定"""
        if not isinstance(other, PolynomialElement):
            return False
        return self.polynomial == other.polynomial and self.ring == other.ring

    def __add__(self, other: "PolynomialElement") -> "PolynomialElement":
        """加法演算"""
        if self.ring != other.ring:
            raise ValueError("異なる多項式環の要素同士の演算はできません")

        result_poly = self.polynomial + other.polynomial
        return PolynomialElement(result_poly, self.ring)

    def __sub__(self, other: "PolynomialElement") -> "PolynomialElement":
        """減法演算"""
        if self.ring != other.ring:
            raise ValueError("異なる多項式環の要素同士の演算はできません")

        result_poly = self.polynomial - other.polynomial
        return PolynomialElement(result_poly, self.ring)

    def __mul__(self, other: "PolynomialElement") -> "PolynomialElement":
        """乗法演算"""
        if self.ring != other.ring:
            raise ValueError("異なる多項式環の要素同士の演算はできません")

        result_poly = self.polynomial * other.polynomial
        return PolynomialElement(result_poly, self.ring)

    def __neg__(self) -> "PolynomialElement":
        """加法逆元"""
        result_poly = -self.polynomial
        return PolynomialElement(result_poly, self.ring)

    def __pow__(self, exponent: int) -> "PolynomialElement":
        """冪乗演算"""
        result_poly = self.polynomial**exponent
        return PolynomialElement(result_poly, self.ring)

    def __str__(self) -> str:
        """文字列表現"""
        return str(self.polynomial)

    def __repr__(self) -> str:
        """デバッグ用文字列表現"""
        return f"PolynomialElement({self.polynomial!r}, {self.ring.name})"


class PolynomialRing:
    """多項式環を表すクラス"""

    def __init__(self, base_ring: Union[Ring, Field], variable: str = "x"):
        """
        多項式環を初期化

        Args:
            base_ring: 基底環または体
            variable: 変数名
        """
        self.base_ring = base_ring
        self.variable = variable
        self.name = f"{base_ring.name}[{variable}]"

    def __eq__(self, other: object) -> bool:
        """多項式環の等価性判定"""
        if not isinstance(other, PolynomialRing):
            return False
        return self.base_ring == other.base_ring and self.variable == other.variable

    def __repr__(self) -> str:
        """文字列表現"""
        return f"PolynomialRing({self.base_ring!r}, '{self.variable}')"

    def zero(self) -> PolynomialElement:
        """零元を取得"""
        zero_poly = Polynomial([0], self.base_ring)
        return PolynomialElement(zero_poly, self)

    def one(self) -> PolynomialElement:
        """単位元を取得"""
        one_poly = Polynomial([1], self.base_ring)
        return PolynomialElement(one_poly, self)

    def x(self) -> PolynomialElement:
        """変数 x を取得"""
        x_poly = Polynomial([0, 1], self.base_ring)
        return PolynomialElement(x_poly, self)

    def constant(self, value: Union[int, Fraction, Any]) -> PolynomialElement:
        """定数多項式を作成"""
        const_poly = Polynomial([value], self.base_ring)
        return PolynomialElement(const_poly, self)

    def from_coefficients(
        self, coefficients: List[Union[int, Fraction, Any]]
    ) -> PolynomialElement:
        """係数リストから多項式環の要素を作成"""
        poly = Polynomial(coefficients, self.base_ring)
        return PolynomialElement(poly, self)

    def element(self, polynomial: Polynomial) -> PolynomialElement:
        """多項式から多項式環の要素を作成"""
        if polynomial.base_ring != self.base_ring:
            raise ValueError("多項式の基底環が多項式環の基底環と一致しません")

        return PolynomialElement(polynomial, self)
