"""
環（Ring）の実装

このモジュールは、ガロア理論における環の概念を実装します。
環は加法と乗法を持つ代数構造で、以下の公理を満たします：

1. 加法について：
   - 結合律: (a + b) + c = a + (b + c)
   - 交換律: a + b = b + a
   - 単位元の存在: a + 0 = a
   - 逆元の存在: a + (-a) = 0

2. 乗法について：
   - 結合律: (a * b) * c = a * (b * c)
   - 単位元の存在: a * 1 = a

3. 分配律: a * (b + c) = a * b + a * c
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Union


class RingElement:
    """環の要素を表すクラス"""

    def __init__(self, value: Any, ring: "Ring") -> None:
        """
        環の要素を初期化

        Args:
            value: 要素の値
            ring: 所属する環
        """
        self.value = value
        self.ring = ring

    def __eq__(self, other: object) -> bool:
        """等価性の判定"""
        if not isinstance(other, RingElement):
            return False
        return self.value == other.value and self.ring == other.ring

    def __add__(self, other: "RingElement") -> "RingElement":
        """加法演算"""
        if self.ring != other.ring:
            raise ValueError("異なる環の要素同士の演算はできません")
        
        result_value = self.ring._add(self.value, other.value)
        return RingElement(result_value, self.ring)

    def __mul__(self, other: "RingElement") -> "RingElement":
        """乗法演算"""
        if self.ring != other.ring:
            raise ValueError("異なる環の要素同士の演算はできません")
        
        result_value = self.ring._multiply(self.value, other.value)
        return RingElement(result_value, self.ring)

    def __neg__(self) -> "RingElement":
        """加法逆元"""
        inverse_value = self.ring._additive_inverse(self.value)
        return RingElement(inverse_value, self.ring)

    def __sub__(self, other: "RingElement") -> "RingElement":
        """減法演算"""
        return self + (-other)

    def __repr__(self) -> str:
        """文字列表現"""
        return f"RingElement({self.value}, {self.ring.name})"


class Ring(ABC):
    """環の抽象基底クラス"""

    def __init__(self, name: str) -> None:
        """
        環を初期化

        Args:
            name: 環の名前
        """
        self.name = name

    @abstractmethod
    def contains(self, value: Any) -> bool:
        """
        値が環に含まれるかを判定

        Args:
            value: 判定する値

        Returns:
            環に含まれる場合True
        """
        pass

    @abstractmethod
    def _add(self, a: Any, b: Any) -> Any:
        """
        環の加法演算（内部実装）

        Args:
            a: 第一要素
            b: 第二要素

        Returns:
            a + b の結果
        """
        pass

    @abstractmethod
    def _multiply(self, a: Any, b: Any) -> Any:
        """
        環の乗法演算（内部実装）

        Args:
            a: 第一要素
            b: 第二要素

        Returns:
            a * b の結果
        """
        pass

    @abstractmethod
    def _additive_inverse(self, a: Any) -> Any:
        """
        加法逆元の計算（内部実装）

        Args:
            a: 要素

        Returns:
            -a の結果
        """
        pass

    @abstractmethod
    def _zero_value(self) -> Any:
        """
        零元の値を取得

        Returns:
            零元の値
        """
        pass

    @abstractmethod
    def _one_value(self) -> Any:
        """
        単位元の値を取得

        Returns:
            単位元の値
        """
        pass

    def zero(self) -> RingElement:
        """
        零元を取得

        Returns:
            零元のRingElement
        """
        return RingElement(self._zero_value(), self)

    def one(self) -> RingElement:
        """
        単位元を取得

        Returns:
            単位元のRingElement
        """
        return RingElement(self._one_value(), self)

    def element(self, value: Any) -> RingElement:
        """
        環の要素を作成

        Args:
            value: 要素の値

        Returns:
            RingElement

        Raises:
            ValueError: 値が環に含まれない場合
        """
        if not self.contains(value):
            raise ValueError(f"値 {value} は環 {self.name} に含まれません")
        
        return RingElement(value, self)

    def __eq__(self, other: object) -> bool:
        """環の等価性判定"""
        if not isinstance(other, Ring):
            return False
        return self.name == other.name and type(self) == type(other)

    def __repr__(self) -> str:
        """文字列表現"""
        return f"{self.__class__.__name__}({self.name})"


class IntegerRing(Ring):
    """整数環 Z の実装"""

    def __init__(self) -> None:
        """整数環を初期化"""
        super().__init__("整数環 Z")

    def contains(self, value: Any) -> bool:
        """整数かどうかを判定"""
        return isinstance(value, int)

    def _add(self, a: int, b: int) -> int:
        """整数の加法"""
        return a + b

    def _multiply(self, a: int, b: int) -> int:
        """整数の乗法"""
        return a * b

    def _additive_inverse(self, a: int) -> int:
        """整数の加法逆元"""
        return -a

    def _zero_value(self) -> int:
        """整数環の零元"""
        return 0

    def _one_value(self) -> int:
        """整数環の単位元"""
        return 1


class RationalRing(Ring):
    """有理数環 Q の実装"""

    def __init__(self) -> None:
        """有理数環を初期化"""
        super().__init__("有理数環 Q")

    def contains(self, value: Any) -> bool:
        """有理数（またはFraction、整数）かどうかを判定"""
        return isinstance(value, (Fraction, int))

    def _add(self, a: Union[Fraction, int], b: Union[Fraction, int]) -> Fraction:
        """有理数の加法"""
        return Fraction(a) + Fraction(b)

    def _multiply(self, a: Union[Fraction, int], b: Union[Fraction, int]) -> Fraction:
        """有理数の乗法"""
        return Fraction(a) * Fraction(b)

    def _additive_inverse(self, a: Union[Fraction, int]) -> Fraction:
        """有理数の加法逆元"""
        return -Fraction(a)

    def _zero_value(self) -> Fraction:
        """有理数環の零元"""
        return Fraction(0)

    def _one_value(self) -> Fraction:
        """有理数環の単位元"""
        return Fraction(1)

    def element(self, value: Any) -> RingElement:
        """
        有理数環の要素を作成（整数の自動変換付き）

        Args:
            value: 要素の値

        Returns:
            RingElement

        Raises:
            ValueError: 値が環に含まれない場合
        """
        if not self.contains(value):
            raise ValueError(f"値 {value} は環 {self.name} に含まれません")
        
        # 整数をFractionに変換
        if isinstance(value, int):
            value = Fraction(value)
        
        return RingElement(value, self) 