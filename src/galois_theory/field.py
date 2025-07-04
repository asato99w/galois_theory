"""
体（Field）の実装

このモジュールは、ガロア理論で使用される体の実装を提供します。
体は環の特殊な場合で、零以外の全ての元が乗法逆元を持つ代数構造です。
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Union


class FieldElement:
    """
    体の要素を表すクラス
    
    体の要素は、その値と所属する体への参照を持ちます。
    """
    
    def __init__(self, value: Any, field: 'Field'):
        """
        体の要素を初期化
        
        Args:
            value: 要素の値
            field: この要素が所属する体
        """
        self.value = value
        self.field = field
    
    def __eq__(self, other: 'FieldElement') -> bool:
        """等価性の判定"""
        if not isinstance(other, FieldElement):
            return False
        return self.value == other.value and self.field == other.field
    
    def __hash__(self) -> int:
        """ハッシュ値の計算"""
        return hash((self.value, id(self.field)))
    
    def __repr__(self) -> str:
        """文字列表現"""
        return f"FieldElement({self.value}, {self.field.name})"
    
    def __str__(self) -> str:
        """ユーザー向け文字列表現"""
        return str(self.value)
    
    def __add__(self, other: 'FieldElement') -> 'FieldElement':
        """加法演算"""
        if self.field != other.field:
            raise ValueError("異なる体の要素同士では演算できません")
        return self.field.add(self, other)
    
    def __mul__(self, other: 'FieldElement') -> 'FieldElement':
        """乗法演算"""
        if self.field != other.field:
            raise ValueError("異なる体の要素同士では演算できません")
        return self.field.multiply(self, other)
    
    def __sub__(self, other: 'FieldElement') -> 'FieldElement':
        """減法演算"""
        if self.field != other.field:
            raise ValueError("異なる体の要素同士では演算できません")
        return self.field.subtract(self, other)
    
    def __truediv__(self, other: 'FieldElement') -> 'FieldElement':
        """除法演算"""
        if self.field != other.field:
            raise ValueError("異なる体の要素同士では演算できません")
        return self.field.divide(self, other)
    
    def __neg__(self) -> 'FieldElement':
        """加法逆元"""
        return self.field.additive_inverse(self)
    
    def inverse(self) -> 'FieldElement':
        """乗法逆元を取得"""
        return self.field.multiplicative_inverse(self)


class Field(ABC):
    """
    体の抽象基底クラス
    
    体は以下の公理を満たす代数構造です：
    1. 加法について可換群を形成
    2. 乗法について（零元を除いて）可換群を形成
    3. 分配律が成り立つ
    """
    
    def __init__(self, name: str = "Field"):
        """
        体を初期化
        
        Args:
            name: 体の名前
        """
        self.name = name
        # 代数的閉体の概念的プロパティ
        self.algebraic_closure = None
    
    def __eq__(self, other: 'Field') -> bool:
        """体の等価性判定"""
        return isinstance(other, type(self)) and self.name == other.name
    
    def __hash__(self) -> int:
        """ハッシュ値の計算"""
        return hash((type(self).__name__, self.name))
    
    def __repr__(self) -> str:
        """文字列表現"""
        return f"{type(self).__name__}('{self.name}')"
    
    @abstractmethod
    def contains(self, value: Any) -> bool:
        """値がこの体に含まれるかを判定"""
        pass
    
    @abstractmethod
    def element(self, value: Any) -> FieldElement:
        """値から体の要素を作成"""
        pass
    
    @abstractmethod
    def zero(self) -> FieldElement:
        """加法単位元（零元）を取得"""
        pass
    
    @abstractmethod
    def one(self) -> FieldElement:
        """乗法単位元（単位元）を取得"""
        pass
    
    @abstractmethod
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """加法演算"""
        pass
    
    @abstractmethod
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """乗法演算"""
        pass
    
    def subtract(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """減法演算（加法逆元を使用）"""
        return self.add(a, self.additive_inverse(b))
    
    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """除法演算（乗法逆元を使用）"""
        if b.value == self.zero().value:
            raise ValueError("零要素による除法はできません")
        return self.multiply(a, self.multiplicative_inverse(b))
    
    @abstractmethod
    def additive_inverse(self, a: FieldElement) -> FieldElement:
        """加法逆元を取得"""
        pass
    
    @abstractmethod
    def multiplicative_inverse(self, a: FieldElement) -> FieldElement:
        """乗法逆元を取得"""
        pass


class RationalField(Field):
    """
    有理数体 Q の実装
    
    有理数体は、分数で表現できる全ての数からなる体です。
    Pythonのfractions.Fractionを使用して実装されています。
    """
    
    def __init__(self):
        """有理数体を初期化"""
        super().__init__("有理数体 Q")
    
    def contains(self, value: Any) -> bool:
        """値が有理数体に含まれるかを判定"""
        if isinstance(value, (int, Fraction)):
            return True
        return False
    
    def element(self, value: Union[int, Fraction]) -> FieldElement:
        """値から有理数体の要素を作成"""
        if isinstance(value, int):
            fraction_value = Fraction(value)
        elif isinstance(value, Fraction):
            fraction_value = value
        else:
            raise ValueError(f"有理数体に変換できない値です: {value}")
        
        return FieldElement(fraction_value, self)
    
    def zero(self) -> FieldElement:
        """加法単位元（零元）を取得"""
        return FieldElement(Fraction(0), self)
    
    def one(self) -> FieldElement:
        """乗法単位元（単位元）を取得"""
        return FieldElement(Fraction(1), self)
    
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """加法演算"""
        result_value = a.value + b.value
        return FieldElement(result_value, self)
    
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """乗法演算"""
        result_value = a.value * b.value
        return FieldElement(result_value, self)
    
    def additive_inverse(self, a: FieldElement) -> FieldElement:
        """加法逆元を取得"""
        result_value = -a.value
        return FieldElement(result_value, self)
    
    def multiplicative_inverse(self, a: FieldElement) -> FieldElement:
        """乗法逆元を取得"""
        if a.value == 0:
            raise ValueError("零要素の乗法逆元は存在しません")
        
        # Fractionの場合、1/aで逆元を計算
        result_value = Fraction(1) / a.value
        return FieldElement(result_value, self)


def _is_prime(n: int) -> bool:
    """
    数が素数かどうかを判定する補助関数
    
    Args:
        n: 判定する整数
        
    Returns:
        素数の場合True、そうでなければFalse
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    拡張ユークリッドの互除法
    
    ax + by = gcd(a, b) となる x, y を求める
    
    Args:
        a, b: 整数
        
    Returns:
        (gcd, x, y) のタプル
    """
    if a == 0:
        return b, 0, 1
    
    gcd, x1, y1 = _extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y


def _mod_inverse(a: int, m: int) -> int:
    """
    モジュラ逆元を計算
    
    Args:
        a: 逆元を求める数
        m: 法
        
    Returns:
        a の mod m での逆元
        
    Raises:
        ValueError: 逆元が存在しない場合
    """
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"{a} の mod {m} での逆元は存在しません")
    return (x % m + m) % m


class FiniteField(Field):
    """
    有限体 F_p の実装（p は素数）
    
    有限体は有限個の要素からなる体です。
    この実装では、素数 p に対する F_p = Z/pZ を扱います。
    """
    
    def __init__(self, p: int):
        """
        有限体 F_p を初期化
        
        Args:
            p: 素数（体の位数）
            
        Raises:
            ValueError: p が素数でない場合
        """
        if not _is_prime(p):
            raise ValueError("有限体の位数は素数である必要があります")
        
        self.characteristic = p
        super().__init__(f"有限体 F_{p}")
    
    def contains(self, value: Any) -> bool:
        """値が有限体に含まれるかを判定"""
        if isinstance(value, int):
            return 0 <= value < self.characteristic
        return False
    
    def element(self, value: int) -> FieldElement:
        """値から有限体の要素を作成"""
        if not isinstance(value, int):
            raise ValueError(f"有限体の要素は整数である必要があります: {value}")
        
        # mod p で正規化
        normalized_value = value % self.characteristic
        return FieldElement(normalized_value, self)
    
    def zero(self) -> FieldElement:
        """加法単位元（零元）を取得"""
        return FieldElement(0, self)
    
    def one(self) -> FieldElement:
        """乗法単位元（単位元）を取得"""
        return FieldElement(1, self)
    
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """加法演算"""
        result_value = (a.value + b.value) % self.characteristic
        return FieldElement(result_value, self)
    
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """乗法演算"""
        result_value = (a.value * b.value) % self.characteristic
        return FieldElement(result_value, self)
    
    def additive_inverse(self, a: FieldElement) -> FieldElement:
        """加法逆元を取得"""
        if a.value == 0:
            result_value = 0
        else:
            result_value = self.characteristic - a.value
        return FieldElement(result_value, self)
    
    def multiplicative_inverse(self, a: FieldElement) -> FieldElement:
        """乗法逆元を取得"""
        if a.value == 0:
            raise ValueError("零要素の乗法逆元は存在しません")
        
        inverse_value = _mod_inverse(a.value, self.characteristic)
        return FieldElement(inverse_value, self)


# よく使用される体のインスタンス
Q = RationalField()  # 有理数体
F2 = FiniteField(2)  # 2元体
F3 = FiniteField(3)  # 3元体
F5 = FiniteField(5)  # 5元体
F7 = FiniteField(7)  # 7元体 