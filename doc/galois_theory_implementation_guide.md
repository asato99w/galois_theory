# ガロア理論による5次方程式可解性判定 - 段階的実装ガイド

## 概要
このガイドでは、ガロア理論を用いて5次方程式の可解性を判定するPythonプログラムを、数学的基礎から段階的に実装する方法を説明します。

## 実装スケジュール（3-6ヶ月）

### Phase 1: 抽象代数の基礎実装 (2-3週間)
#### 目標
- 環(Ring)と体(Field)の基本構造を実装
- 代数的演算の基盤を構築

#### 実装内容
```python
class Ring:
    """環の抽象基底クラス"""
    def __init__(self, elements, add_op, mul_op):
        self.elements = set(elements)
        self.add = add_op
        self.mul = mul_op
        self.zero = self._find_zero()
        self.one = self._find_one()
    
    def _find_zero(self):
        """加法単位元を見つける"""
        for e in self.elements:
            if all(self.add(e, x) == x for x in self.elements):
                return e
        raise ValueError("加法単位元が存在しません")
    
    def _find_one(self):
        """乗法単位元を見つける"""
        for e in self.elements:
            if all(self.mul(e, x) == x for x in self.elements):
                return e
        raise ValueError("乗法単位元が存在しません")
    
    def is_commutative(self):
        """可換性の判定"""
        return all(self.mul(a, b) == self.mul(b, a) 
                  for a in self.elements for b in self.elements)
    
    def additive_inverse(self, element):
        """加法逆元"""
        for x in self.elements:
            if self.add(element, x) == self.zero:
                return x
        raise ValueError(f"{element}の加法逆元が存在しません")

class Field(Ring):
    """体の実装"""
    def __init__(self, elements, add_op, mul_op):
        super().__init__(elements, add_op, mul_op)
        if not self._is_field():
            raise ValueError("体の条件を満たしていません")
    
    def _is_field(self):
        """体の公理を満たすかチェック"""
        # 1. 可換環であること
        if not self.is_commutative():
            return False
        
        # 2. 零でない全ての元が乗法逆元を持つこと
        for element in self.elements:
            if element != self.zero:
                try:
                    self.multiplicative_inverse(element)
                except ValueError:
                    return False
        return True
    
    def multiplicative_inverse(self, element):
        """乗法逆元"""
        if element == self.zero:
            raise ValueError("零元の乗法逆元は存在しません")
        
        for x in self.elements:
            if self.mul(element, x) == self.one:
                return x
        raise ValueError(f"{element}の乗法逆元が存在しません")
    
    def divide(self, a, b):
        """除法演算"""
        if b == self.zero:
            raise ValueError("零で除算することはできません")
        return self.mul(a, self.multiplicative_inverse(b))
```

#### 週次タスク
- **Week 1**: Ring クラスの実装と基本テスト
- **Week 2**: Field クラスの実装と有限体の例
- **Week 3**: 有理数体 Q の実装とテスト強化

#### 成果物
- `abstract_algebra.py`: 環と体の基本実装
- `test_algebra.py`: 単体テスト
- `examples_finite_fields.py`: 有限体の具体例

---

### Phase 2: 多項式環の実装 (2-3週間)
#### 目標
- 多項式の基本演算を実装
- 既約性判定アルゴリズムを構築

#### 実装内容
```python
class Polynomial:
    """多項式クラス"""
    def __init__(self, coefficients, field):
        self.coeffs = list(coefficients)
        self.field = field
        self._normalize()
    
    def _normalize(self):
        """最高次の係数が0でないように正規化"""
        while len(self.coeffs) > 1 and self.coeffs[-1] == self.field.zero:
            self.coeffs.pop()
    
    def degree(self):
        """次数を返す"""
        return len(self.coeffs) - 1
    
    def __add__(self, other):
        """多項式の加法"""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else self.field.zero
            b = other.coeffs[i] if i < len(other.coeffs) else self.field.zero
            result.append(self.field.add(a, b))
        
        return Polynomial(result, self.field)
    
    def __mul__(self, other):
        """多項式の乗法"""
        if isinstance(other, Polynomial):
            result = [self.field.zero] * (len(self.coeffs) + len(other.coeffs) - 1)
            
            for i, a in enumerate(self.coeffs):
                for j, b in enumerate(other.coeffs):
                    result[i + j] = self.field.add(result[i + j], 
                                                  self.field.mul(a, b))
            
            return Polynomial(result, self.field)
        else:
            # スカラー倍
            return Polynomial([self.field.mul(c, other) for c in self.coeffs], 
                            self.field)
    
    def evaluate(self, value):
        """多項式の値を計算（ホーナー法）"""
        if not self.coeffs:
            return self.field.zero
        
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = self.field.add(self.coeffs[i], 
                                  self.field.mul(result, value))
        return result
    
    def derivative(self):
        """導関数を計算"""
        if self.degree() == 0:
            return Polynomial([self.field.zero], self.field)
        
        result = []
        for i in range(1, len(self.coeffs)):
            # i * coeffs[i] を計算
            coeff = self.coeffs[i]
            for _ in range(i):
                coeff = self.field.add(coeff, self.coeffs[i])
            result.append(coeff)
        
        return Polynomial(result, self.field)
    
    def gcd(self, other):
        """最大公約式（ユークリッドの互除法）"""
        a, b = self, other
        while not b.is_zero():
            a, b = b, a.mod(b)
        return a.monic()
    
    def is_irreducible(self):
        """既約性の判定"""
        # アイゼンシュタインの既約判定法
        if self._eisenstein_criterion():
            return True
        
        # 有限体上での既約性テスト
        return self._finite_field_irreducibility_test()
    
    def _eisenstein_criterion(self):
        """アイゼンシュタインの既約判定法"""
        # 実装は体の性質に依存
        pass
    
    def _finite_field_irreducibility_test(self):
        """有限体上での既約性テスト"""
        # Rabin's irreducibility test
        pass
```

#### 週次タスク
- **Week 1**: 基本的な多項式演算の実装
- **Week 2**: 既約性判定アルゴリズムの実装
- **Week 3**: 多項式の因数分解アルゴリズム

#### 成果物
- `polynomial.py`: 多項式クラスの実装
- `irreducibility.py`: 既約性判定アルゴリズム
- `test_polynomials.py`: 多項式のテストスイート

---

### Phase 3: 体の拡大理論 (3-4週間)
#### 目標
- 体の拡大の実装
- 最小多項式の計算
- 分解体の構成

#### 実装内容
```python
class FieldExtension:
    """体の拡大 K(α)/K"""
    def __init__(self, base_field, minimal_polynomial, generator_name='α'):
        self.base_field = base_field
        self.minimal_poly = minimal_polynomial
        self.generator = generator_name
        self.degree_value = minimal_polynomial.degree()
        
        # 拡大体の元を表現するための基底
        self.basis = self._construct_basis()
    
    def _construct_basis(self):
        """拡大体の基底 {1, α, α^2, ..., α^(n-1)} を構築"""
        basis = []
        for i in range(self.degree_value):
            # α^i を表現
            coeffs = [self.base_field.zero] * self.degree_value
            if i < len(coeffs):
                coeffs[i] = self.base_field.one
            basis.append(ExtensionElement(coeffs, self))
        return basis
    
    def degree(self):
        """拡大次数 [K(α):K]"""
        return self.degree_value
    
    def contains_element(self, element):
        """元が拡大体に含まれるかチェック"""
        return isinstance(element, ExtensionElement) and element.extension == self

class ExtensionElement:
    """拡大体の元 a₀ + a₁α + a₂α² + ... + aₙ₋₁α^(n-1)"""
    def __init__(self, coefficients, extension):
        self.coeffs = list(coefficients)
        self.extension = extension
        self._normalize()
    
    def _normalize(self):
        """最小多項式による簡約"""
        while len(self.coeffs) >= self.extension.degree_value:
            # α^n を α^(n-1), α^(n-2), ..., 1 の線形結合で表現
            self._reduce_by_minimal_polynomial()
    
    def _reduce_by_minimal_polynomial(self):
        """最小多項式 f(α) = 0 を使って簡約"""
        if len(self.coeffs) < self.extension.degree_value:
            return
        
        # 最高次の係数を取得
        leading_coeff = self.coeffs[-1]
        
        # f(x) = x^n + a_(n-1)x^(n-1) + ... + a_1x + a_0 = 0
        # なので x^n = -(a_(n-1)x^(n-1) + ... + a_1x + a_0)
        minimal_coeffs = self.extension.minimal_poly.coeffs
        
        # 簡約を実行
        for i in range(len(minimal_coeffs) - 1):
            reduction = self.extension.base_field.mul(
                leading_coeff, 
                self.extension.base_field.additive_inverse(minimal_coeffs[i])
            )
            if i < len(self.coeffs) - 1:
                self.coeffs[i] = self.extension.base_field.add(
                    self.coeffs[i], reduction
                )
        
        # 最高次の項を削除
        self.coeffs.pop()
    
    def __add__(self, other):
        """拡大体での加法"""
        if not isinstance(other, ExtensionElement):
            raise TypeError("ExtensionElement同士でのみ加法が可能です")
        
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else self.extension.base_field.zero
            b = other.coeffs[i] if i < len(other.coeffs) else self.extension.base_field.zero
            result.append(self.extension.base_field.add(a, b))
        
        return ExtensionElement(result, self.extension)
    
    def __mul__(self, other):
        """拡大体での乗法"""
        if not isinstance(other, ExtensionElement):
            raise TypeError("ExtensionElement同士でのみ乗法が可能です")
        
        # 係数の畳み込み
        result_coeffs = [self.extension.base_field.zero] * (
            len(self.coeffs) + len(other.coeffs) - 1
        )
        
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                product = self.extension.base_field.mul(a, b)
                result_coeffs[i + j] = self.extension.base_field.add(
                    result_coeffs[i + j], product
                )
        
        return ExtensionElement(result_coeffs, self.extension)
    
    def minimal_polynomial(self):
        """この元の最小多項式を計算"""
        # 線形従属性を調べて最小多項式を求める
        powers = [self.extension.basis[0]]  # 1
        current = self
        
        for i in range(1, self.extension.degree_value + 1):
            powers.append(current)
            
            # 線形従属性をチェック
            if self._is_linearly_dependent(powers):
                return self._find_minimal_polynomial(powers)
            
            current = current * self
        
        # degree_value + 1 個の元は必ず線形従属
        return self._find_minimal_polynomial(powers)
    
    def _is_linearly_dependent(self, elements):
        """元のリストが線形従属かチェック"""
        # 連立方程式を解いて非自明解が存在するかチェック
        pass
    
    def _find_minimal_polynomial(self, powers):
        """線形従属関係から最小多項式を求める"""
        # ガウスの消去法で係数を求める
        pass

class SplittingField:
    """分解体の構成"""
    def __init__(self, polynomial, base_field):
        self.polynomial = polynomial
        self.base_field = base_field
        self.extensions = []
        self.splitting_field = self._construct_splitting_field()
    
    def _construct_splitting_field(self):
        """分解体を段階的に構成"""
        current_field = self.base_field
        current_poly = self.polynomial
        
        while not current_poly.splits_completely_over(current_field):
            # 既約因子を見つける
            irreducible_factor = current_poly.find_irreducible_factor(current_field)
            
            # 体を拡大
            extension = FieldExtension(current_field, irreducible_factor)
            self.extensions.append(extension)
            
            current_field = extension
            current_poly = current_poly.factor_over(current_field)
        
        return current_field
    
    def degree(self):
        """分解体の次数 [分解体:基底体]"""
        degree = 1
        for ext in self.extensions:
            degree *= ext.degree()
        return degree
    
    def all_roots(self):
        """多項式の全ての根を返す"""
        return self.polynomial.roots_in(self.splitting_field)
```

#### 週次タスク
- **Week 1**: 基本的な体の拡大の実装
- **Week 2**: 最小多項式の計算アルゴリズム
- **Week 3**: 分解体の構成アルゴリズム
- **Week 4**: 拡大体での演算の最適化

#### 成果物
- `field_extension.py`: 体の拡大の実装
- `splitting_field.py`: 分解体の構成
- `minimal_polynomial.py`: 最小多項式の計算
- `test_extensions.py`: 拡大体のテストスイート

---

### Phase 4: 群論の実装 (3-4週間)
#### 目標
- 群の基本構造を実装
- 可解群の判定アルゴリズム
- 対称群と交代群の実装

#### 実装内容
```python
class Group:
    """群の抽象基底クラス"""
    def __init__(self, elements, operation, identity=None):
        self.elements = set(elements)
        self.op = operation
        self.identity = identity or self._find_identity()
        self._verify_group_axioms()
    
    def _find_identity(self):
        """単位元を見つける"""
        for e in self.elements:
            if all(self.op(e, x) == x and self.op(x, e) == x 
                   for x in self.elements):
                return e
        raise ValueError("単位元が見つかりません")
    
    def _verify_group_axioms(self):
        """群の公理を確認"""
        # 1. 結合律
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.op(self.op(a, b), c) != self.op(a, self.op(b, c)):
                        raise ValueError("結合律が成り立ちません")
        
        # 2. 単位元の存在（既に確認済み）
        
        # 3. 逆元の存在
        for x in self.elements:
            inverse_found = False
            for y in self.elements:
                if self.op(x, y) == self.identity and self.op(y, x) == self.identity:
                    inverse_found = True
                    break
            if not inverse_found:
                raise ValueError(f"{x}の逆元が存在しません")
    
    def order(self):
        """群の位数"""
        return len(self.elements)
    
    def inverse(self, element):
        """元の逆元を求める"""
        for x in self.elements:
            if self.op(element, x) == self.identity:
                return x
        raise ValueError(f"{element}の逆元が見つかりません")
    
    def is_abelian(self):
        """可換群かどうかを判定"""
        return all(self.op(a, b) == self.op(b, a) 
                  for a in self.elements for b in self.elements)
    
    def subgroups(self):
        """全ての部分群を求める"""
        subgroups = []
        
        # 全ての部分集合について部分群かどうかをチェック
        from itertools import combinations
        
        for r in range(1, len(self.elements) + 1):
            for subset in combinations(self.elements, r):
                if self._is_subgroup(set(subset)):
                    subgroups.append(Group(subset, self.op, self.identity))
        
        return subgroups
    
    def _is_subgroup(self, subset):
        """部分集合が部分群かどうかを判定"""
        # 1. 単位元を含む
        if self.identity not in subset:
            return False
        
        # 2. 演算について閉じている
        for a in subset:
            for b in subset:
                if self.op(a, b) not in subset:
                    return False
        
        # 3. 逆元を含む
        for a in subset:
            if self.inverse(a) not in subset:
                return False
        
        return True
    
    def normal_subgroups(self):
        """正規部分群を求める"""
        normal_subgroups = []
        
        for subgroup in self.subgroups():
            if self._is_normal_subgroup(subgroup):
                normal_subgroups.append(subgroup)
        
        return normal_subgroups
    
    def _is_normal_subgroup(self, subgroup):
        """正規部分群かどうかを判定"""
        H = subgroup.elements
        
        for g in self.elements:
            for h in H:
                # gHg^(-1) ⊆ H をチェック
                conjugate = self.op(self.op(g, h), self.inverse(g))
                if conjugate not in H:
                    return False
        
        return True
    
    def composition_series(self):
        """合成列を求める"""
        series = [self]
        current = self
        
        while current.order() > 1:
            # 最大正規部分群を見つける
            max_normal = None
            max_order = 0
            
            for normal in current.normal_subgroups():
                if normal.order() > max_order and normal.order() < current.order():
                    max_normal = normal
                    max_order = normal.order()
            
            if max_normal is None:
                break
            
            series.append(max_normal)
            current = max_normal
        
        return series
    
    def is_solvable(self):
        """可解群かどうかを判定"""
        series = self.composition_series()
        
        # 合成列の各商群がアーベル群かどうかをチェック
        for i in range(len(series) - 1):
            quotient = self._quotient_group(series[i], series[i + 1])
            if not quotient.is_abelian():
                return False
        
        return True
    
    def _quotient_group(self, group, normal_subgroup):
        """商群を構成"""
        # 左剰余類を求める
        cosets = []
        H = normal_subgroup.elements
        
        for g in group.elements:
            coset = {group.op(g, h) for h in H}
            if coset not in cosets:
                cosets.append(coset)
        
        # 商群の演算を定義
        def quotient_op(coset1, coset2):
            # 代表元を取って演算
            g1 = next(iter(coset1))
            g2 = next(iter(coset2))
            result_rep = group.op(g1, g2)
            
            # result_repを含む剰余類を見つける
            for coset in cosets:
                if result_rep in coset:
                    return coset
            
            raise ValueError("商群の演算でエラーが発生しました")
        
        return Group(cosets, quotient_op)

class SymmetricGroup(Group):
    """対称群 S_n"""
    def __init__(self, n):
        self.n = n
        self.permutations = self._generate_permutations()
        super().__init__(self.permutations, self._compose_permutations)
    
    def _generate_permutations(self):
        """全ての置換を生成"""
        from itertools import permutations
        return list(permutations(range(self.n)))
    
    def _compose_permutations(self, perm1, perm2):
        """置換の合成"""
        return tuple(perm1[perm2[i]] for i in range(self.n))
    
    def alternating_subgroup(self):
        """交代群 A_n を返す"""
        even_permutations = []
        
        for perm in self.permutations:
            if self._is_even_permutation(perm):
                even_permutations.append(perm)
        
        return Group(even_permutations, self._compose_permutations)
    
    def _is_even_permutation(self, perm):
        """偶置換かどうかを判定"""
        inversions = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if perm[i] > perm[j]:
                    inversions += 1
        return inversions % 2 == 0
    
    def cycle_decomposition(self, perm):
        """置換の巡回分解"""
        visited = [False] * self.n
        cycles = []
        
        for i in range(self.n):
            if not visited[i]:
                cycle = []
                current = i
                
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    current = perm[current]
                
                if len(cycle) > 1:
                    cycles.append(tuple(cycle))
        
        return cycles
```

#### 週次タスク
- **Week 1**: 基本的な群の実装
- **Week 2**: 部分群と正規部分群の実装
- **Week 3**: 対称群と交代群の実装
- **Week 4**: 可解群の判定アルゴリズム

#### 成果物
- `group_theory.py`: 群の基本実装
- `symmetric_group.py`: 対称群の実装
- `solvable_groups.py`: 可解群の判定
- `test_groups.py`: 群論のテストスイート

---

### Phase 5: ガロア群の計算 (4-5週間)
#### 目標
- ガロア群の実装
- 自己同型写像の生成
- ガロアの基本定理の実装

#### 実装内容
```python
class GaloisGroup(Group):
    """ガロア群 Gal(L/K)"""
    def __init__(self, polynomial, base_field):
        self.polynomial = polynomial
        self.base_field = base_field
        self.splitting_field = SplittingField(polynomial, base_field)
        self.roots = self.splitting_field.all_roots()
        
        # 自己同型写像を生成
        self.automorphisms = self._generate_automorphisms()
        
        super().__init__(self.automorphisms, self._compose_automorphisms)
    
    def _generate_automorphisms(self):
        """K-自己同型写像を生成"""
        automorphisms = []
        
        # 根の置換から自己同型写像を構成
        from itertools import permutations
        
        for perm in permutations(self.roots):
            # この置換が自己同型写像に拡張できるかチェック
            if self._can_extend_to_automorphism(perm):
                automorphism = self._construct_automorphism(perm)
                automorphisms.append(automorphism)
        
        return automorphisms
    
    def _can_extend_to_automorphism(self, root_permutation):
        """根の置換が自己同型写像に拡張できるかチェック"""
        # 多項式の関係式を保つかどうかをチェック
        for i, root in enumerate(self.roots):
            mapped_root = root_permutation[i]
            
            # f(root) = 0 ⟹ f(mapped_root) = 0
            if self.polynomial.evaluate(mapped_root) != self.base_field.zero:
                return False
        
        return True
    
    def _construct_automorphism(self, root_permutation):
        """根の置換から自己同型写像を構成"""
        def automorphism(element):
            # 分解体の元を根の多項式として表現し、
            # 根を置換して新しい元を得る
            return self._apply_root_permutation(element, root_permutation)
        
        return automorphism
    
    def _apply_root_permutation(self, element, root_permutation):
        """根の置換を元に適用"""
        # 実装は分解体の構造に依存
        pass
    
    def _compose_automorphisms(self, auto1, auto2):
        """自己同型写像の合成"""
        def composed(element):
            return auto1(auto2(element))
        return composed
    
    def fixed_field(self, subgroup):
        """部分群の固定体を求める"""
        fixed_elements = []
        
        for element in self.splitting_field.elements:
            is_fixed = True
            for automorphism in subgroup.elements:
                if automorphism(element) != element:
                    is_fixed = False
                    break
            
            if is_fixed:
                fixed_elements.append(element)
        
        return Field(fixed_elements, self.base_field.add, self.base_field.mul)
    
    def fundamental_theorem(self):
        """ガロアの基本定理による対応"""
        correspondence = {}
        
        # 全ての部分群について固定体を計算
        for subgroup in self.subgroups():
            fixed = self.fixed_field(subgroup)
            correspondence[subgroup] = fixed
        
        return correspondence
    
    def is_galois_extension(self):
        """ガロア拡大かどうかを判定"""
        # 1. 正規拡大かどうか
        if not self._is_normal_extension():
            return False
        
        # 2. 分離可能拡大かどうか
        if not self._is_separable_extension():
            return False
        
        return True
    
    def _is_normal_extension(self):
        """正規拡大かどうかを判定"""
        # 最小多項式が分解体で完全に分解するかチェック
        return self.polynomial.splits_completely_over(self.splitting_field)
    
    def _is_separable_extension(self):
        """分離可能拡大かどうかを判定"""
        # 最小多項式が重根を持たないかチェック
        derivative = self.polynomial.derivative()
        gcd = self.polynomial.gcd(derivative)
        return gcd.degree() == 0
    
    def resolvent_polynomial(self):
        """レゾルベント多項式を計算"""
        # 5次方程式の場合の特殊な実装
        if self.polynomial.degree() == 5:
            return self._quintic_resolvent()
        else:
            raise NotImplementedError("5次以外のレゾルベントは未実装")
    
    def _quintic_resolvent(self):
        """5次方程式のレゾルベント多項式"""
        # 6次レゾルベント多項式を計算
        roots = self.roots
        
        # Tschirnhaus変換を使用
        # 実装は複雑なので概要のみ
        pass
    
    def galois_group_classification(self):
        """ガロア群の分類"""
        order = self.order()
        
        if self.polynomial.degree() == 5:
            return self._classify_quintic_galois_group()
        else:
            # 一般的な分類
            return self._classify_general_galois_group()
    
    def _classify_quintic_galois_group(self):
        """5次方程式のガロア群を分類"""
        order = self.order()
        
        if order == 5:
            return "C5"  # 巡回群
        elif order == 10:
            return "D5"  # 二面体群
        elif order == 20:
            return "F20"  # Frobenius群
        elif order == 60:
            return "A5"  # 交代群
        elif order == 120:
            return "S5"  # 対称群
        else:
            return f"Unknown order {order}"
    
    def _classify_general_galois_group(self):
        """一般的なガロア群の分類"""
        # 群の不変量による分類
        if self.is_abelian():
            return f"Abelian group of order {self.order()}"
        elif self.is_solvable():
            return f"Solvable group of order {self.order()}"
        else:
            return f"Non-solvable group of order {self.order()}"
```

#### 週次タスク
- **Week 1**: 基本的なガロア群の実装
- **Week 2**: 自己同型写像の生成アルゴリズム
- **Week 3**: ガロアの基本定理の実装
- **Week 4**: レゾルベント多項式の計算
- **Week 5**: ガロア群の分類アルゴリズム

#### 成果物
- `galois_group.py`: ガロア群の実装
- `automorphisms.py`: 自己同型写像の生成
- `fundamental_theorem.py`: ガロアの基本定理
- `resolvent.py`: レゾルベント多項式
- `test_galois.py`: ガロア群のテストスイート

---

### Phase 6: 可解性判定の実装 (2-3週間)
#### 目標
- 5次方程式の可解性判定の完成
- 統合テストとデバッグ
- 性能最適化

#### 実装内容
```python
class QuinticSolvabilityTester:
    """5次方程式の可解性判定器"""
    
    def __init__(self, base_field=None):
        self.base_field = base_field or RationalField()
    
    def is_solvable_by_radicals(self, coefficients):
        """根号による可解性を判定"""
        polynomial = Polynomial(coefficients, self.base_field)
        
        # Step 1: 前処理とチェック
        if not self._preliminary_checks(polynomial):
            return False, "前処理でエラーが発生しました"
        
        # Step 2: 特殊ケースの判定
        special_result = self._check_special_cases(polynomial)
        if special_result is not None:
            return special_result
        
        # Step 3: ガロア群の計算
        try:
            galois_group = GaloisGroup(polynomial, self.base_field)
            group_type = galois_group.galois_group_classification()
            
            # Step 4: 可解性の判定
            is_solvable = galois_group.is_solvable()
            
            return is_solvable, f"ガロア群: {group_type}"
            
        except Exception as e:
            return False, f"ガロア群の計算でエラー: {str(e)}"
    
    def _preliminary_checks(self, polynomial):
        """前処理とチェック"""
        # 1. 5次方程式かどうか
        if polynomial.degree() != 5:
            raise ValueError("5次方程式ではありません")
        
        # 2. 既約性の確認
        if not polynomial.is_irreducible():
            return False  # 既約でない場合は別途処理が必要
        
        # 3. 係数の妥当性チェック
        if not self._validate_coefficients(polynomial):
            return False
        
        return True
    
    def _validate_coefficients(self, polynomial):
        """係数の妥当性をチェック"""
        # 零多項式でないことを確認
        if all(coeff == self.base_field.zero for coeff in polynomial.coeffs):
            return False
        
        # 最高次の係数が零でないことを確認
        if polynomial.coeffs[-1] == self.base_field.zero:
            return False
        
        return True
    
    def _check_special_cases(self, polynomial):
        """特殊ケースの判定"""
        # 1. 二項式 x^5 + a = 0
        if self._is_binomial(polynomial):
            return True, "二項式形式のため可解"
        
        # 2. 既知の可解形式
        if self._is_known_solvable_form(polynomial):
            return True, "既知の可解形式"
        
        # 3. 対称式
        if self._is_symmetric(polynomial):
            return True, "対称式のため可解"
        
        return None
    
    def _is_binomial(self, polynomial):
        """二項式かどうかを判定"""
        non_zero_coeffs = [i for i, coeff in enumerate(polynomial.coeffs) 
                          if coeff != self.base_field.zero]
        return len(non_zero_coeffs) == 2 and 0 in non_zero_coeffs and 5 in non_zero_coeffs
    
    def _is_known_solvable_form(self, polynomial):
        """既知の可解形式かどうかを判定"""
        # Emma Lehmer's quintics など
        return False  # 実装は省略
    
    def _is_symmetric(self, polynomial):
        """対称式かどうかを判定"""
        # 係数の対称性をチェック
        coeffs = polynomial.coeffs
        return coeffs == coeffs[::-1]
    
    def detailed_analysis(self, coefficients):
        """詳細な解析結果を返す"""
        polynomial = Polynomial(coefficients, self.base_field)
        
        result = {
            'polynomial': str(polynomial),
            'degree': polynomial.degree(),
            'is_irreducible': polynomial.is_irreducible(),
            'discriminant': self._compute_discriminant(polynomial),
            'special_cases': self._analyze_special_cases(polynomial),
            'galois_analysis': None,
            'solvability': None
        }
        
        try:
            galois_group = GaloisGroup(polynomial, self.base_field)
            result['galois_analysis'] = {
                'group_order': galois_group.order(),
                'group_type': galois_group.galois_group_classification(),
                'is_abelian': galois_group.is_abelian(),
                'is_solvable': galois_group.is_solvable(),
                'normal_subgroups': len(galois_group.normal_subgroups())
            }
            
            result['solvability'] = galois_group.is_solvable()
            
        except Exception as e:
            result['galois_analysis'] = f"エラー: {str(e)}"
            result['solvability'] = False
        
        return result
    
    def _compute_discriminant(self, polynomial):
        """判別式を計算"""
        # 5次方程式の判別式は非常に複雑
        # 簡略化された実装
        return "計算省略"
    
    def _analyze_special_cases(self, polynomial):
        """特殊ケースの詳細分析"""
        cases = []
        
        if self._is_binomial(polynomial):
            cases.append("二項式")
        
        if self._is_symmetric(polynomial):
            cases.append("対称式")
        
        return cases

# 使用例とテスト
def main():
    """メイン関数"""
    tester = QuinticSolvabilityTester()
    
    # テストケース
    test_cases = [
        # 可解な例
        ([1, 0, 0, 0, 0, -2], "x^5 - 2 = 0"),
        ([1, 0, -5, 0, 12, 0], "x^5 - 5x^3 + 12x = 0"),
        
        # 不可解な例
        ([1, 0, 0, -4, 2, 0], "x^5 - 4x^2 + 2x = 0"),
        ([1, -1, 0, 0, 0, 1], "x^5 - x + 1 = 0"),
    ]
    
    for coeffs, description in test_cases:
        print(f"\n=== {description} ===")
        
        # 基本的な可解性判定
        is_solvable, reason = tester.is_solvable_by_radicals(coeffs)
        print(f"可解性: {is_solvable}")
        print(f"理由: {reason}")
        
        # 詳細分析
        analysis = tester.detailed_analysis(coeffs)
        print(f"詳細分析: {analysis}")

if __name__ == "__main__":
    main()
```

#### 週次タスク
- **Week 1**: 統合テストとデバッグ
- **Week 2**: 性能最適化とエラーハンドリング
- **Week 3**: ドキュメント作成とリファクタリング

#### 成果物
- `quintic_solver.py`: 統合された可解性判定器
- `test_integration.py`: 統合テストスイート
- `performance_analysis.py`: 性能分析ツール
- `documentation.md`: 完全なドキュメント

---

## 実装上の注意点

### 数値精度の管理
```python
class RationalField:
    """有理数体の実装（精度管理）"""
    def __init__(self):
        from fractions import Fraction
        self.Fraction = Fraction
        self.zero = Fraction(0)
        self.one = Fraction(1)
```

### メモリ効率の考慮
```python
class LazyGaloisGroup:
    """遅延評価によるガロア群の実装"""
    def __init__(self, polynomial, base_field):
        self.polynomial = polynomial
        self.base_field = base_field
        self._automorphisms = None  # 遅延評価
    
    @property
    def automorphisms(self):
        if self._automorphisms is None:
            self._automorphisms = self._generate_automorphisms()
        return self._automorphisms
```

### エラーハンドリング
```python
class GaloisTheoryError(Exception):
    """ガロア理論計算のエラー"""
    pass

class IrreducibilityError(GaloisTheoryError):
    """既約性判定のエラー"""
    pass

class SolvabilityError(GaloisTheoryError):
    """可解性判定のエラー"""
    pass
```

## 推奨開発環境

### 必要なライブラリ
```python
# requirements.txt
sympy>=1.12
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.0.0
```

### 開発ツール
- **IDE**: PyCharm, VSCode
- **テスト**: pytest
- **ドキュメント**: Sphinx
- **バージョン管理**: Git

## 学習リソース

### 推奨書籍
1. "Galois Theory" by David A. Cox
2. "Abstract Algebra" by Dummit and Foote
3. "A First Course in Abstract Algebra" by Fraleigh

### オンラインリソース
- SymPy Documentation
- SageMath Tutorials
- Wolfram MathWorld

## 最終目標

この実装により、以下が達成されます：

1. **理論的理解**: ガロア理論の完全な理解
2. **実用的ツール**: 5次方程式の可解性判定器
3. **拡張可能性**: 他の次数への応用可能性
4. **教育価値**: 数学教育への活用

実装完了後は、具体的な5次方程式の例を用いて検証を行い、理論と実装の整合性を確認します。 