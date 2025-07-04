# Phase 3: 体の拡大理論

## 概要と目的

体の拡大理論は、ガロア理論の心臓部です。このフェーズでは、基本体から始めて、代数的元を添加することで新しい体を構築する方法を実装します。5次方程式の可解性判定において、体の拡大は多項式の根を含む最小の体（分解体）を構築するために不可欠です。

体の拡大を理解することで、なぜ5次方程式の一般解が根号で表現できないのか、そしてどのような特殊な場合に解けるのかを具体的に把握できるようになります。

## 理論的背景

### 体の拡大とは何か

体の拡大とは、体Kを含むより大きな体Lを構築することです。記号的には K ⊆ L と表現されます。例えば：

- **有理数体から実数体への拡大**: Q ⊆ R
- **実数体から複素数体への拡大**: R ⊆ C
- **有理数体から√2を含む体への拡大**: Q ⊆ Q(√2)

### 代数的拡大と超越拡大

体の拡大は、その性質によって分類されます：

#### 代数的拡大
拡大体Lの全ての元が、基本体K上の多項式の根となる場合を代数的拡大と呼びます。

**例**: Q(√2) は Q の代数的拡大です。√2 は多項式 x² - 2 = 0 の根だからです。

#### 超越拡大
代数的でない元（超越元）を含む拡大を超越拡大と呼びます。

**例**: Q(π) は Q の超越拡大です。π は Q 上のどの多項式の根でもありません。

### 最小多項式の重要性

代数的元 α に対して、α を根とする K 上の最小次数のモニック多項式を最小多項式と呼びます。最小多項式は以下の性質を持ちます：

1. **既約性**: 最小多項式は常に既約です
2. **一意性**: 各代数的元に対して一意に決まります
3. **分割性**: α を根とする全ての多項式は、最小多項式で割り切れます

### 拡大の次数

体の拡大 L/K について、L を K 上のベクトル空間と見たときの次元を拡大の次数と呼び、[L:K] で表します。

**重要な性質**:
- [L:K] = deg(最小多項式) （単純拡大の場合）
- [M:K] = [M:L] × [L:K] （拡大の乗法性）

## 実装の戦略

### なぜ段階的に拡大を構築するのか

一度に複雑な体を構築するのではなく、段階的に拡大を構築する理由：

1. **理論的明確性**: 各段階での構造が明確になります
2. **計算効率**: 必要最小限の拡大を構築できます
3. **デバッグ容易性**: 問題が発生した場合の特定が容易です
4. **一般化可能性**: 任意の多項式に対応できます

### 実装上の課題と解決策

#### 課題1: 代数的元の表現
代数的元をコンピュータ上でどう表現するかという根本的な問題があります。

**解決策**: 最小多項式を使用した剰余表現を採用します。α の最小多項式を m(x) とすると、K(α) の元は m(x) で割った余りとして表現できます。

#### 課題2: 演算の効率性
体の拡大における演算は、基本体の演算よりも複雑になります。

**解決策**: 
- 最小多項式による剰余演算を効率化
- 頻繁に使用される演算結果をキャッシュ
- 特殊ケース（次数が小さい場合）の最適化

#### 課題3: 最小多項式の計算
与えられた代数的元の最小多項式を効率的に計算する必要があります。

**解決策**: 
- 線形代数的手法を使用
- 元の冪乗の線形従属性を利用
- 既約性判定アルゴリズムとの連携

## 具体的な実装手順

### Week 1: 基本的な拡大体の実装

#### 1.1 FieldExtensionクラスの設計
体の拡大を表現するクラスを設計します：

```python
class FieldExtension:
    def __init__(self, base_field, minimal_polynomial):
        self.base_field = base_field
        self.minimal_polynomial = minimal_polynomial
        self.degree = minimal_polynomial.degree()
        self.generator = self._create_generator()
```

#### 1.2 ExtensionElementクラスの実装
拡大体の元を表現するクラス：

```python
class ExtensionElement:
    def __init__(self, coefficients, field_extension):
        self.coefficients = coefficients  # 最小多項式による表現
        self.field_extension = field_extension
```

#### 1.3 基本演算の実装
拡大体における加法と乗法：

**加法**: 係数ごとの加法（基本体と同じ）
```
(a₀ + a₁α + a₂α²) + (b₀ + b₁α + b₂α²) = (a₀+b₀) + (a₁+b₁)α + (a₂+b₂)α²
```

**乗法**: 多項式の乗法後、最小多項式による剰余
```
(a₀ + a₁α) × (b₀ + b₁α) = a₀b₀ + (a₀b₁ + a₁b₀)α + a₁b₁α²
```
α² を最小多項式で置換

#### 1.4 逆元の計算
拡大ユークリッドの互除法を使用して逆元を計算：
```
gcd(f(α), m(α)) = 1 = u(α)f(α) + v(α)m(α)
```
ここで u(α) が f(α) の逆元

### Week 2: 最小多項式の計算アルゴリズム

#### 2.1 線形代数的手法
代数的元 α に対して、1, α, α², α³, ... の線形従属関係を見つけます：

```python
def find_minimal_polynomial(element, base_field, max_degree):
    # 元の冪乗を計算
    powers = [element**i for i in range(max_degree + 1)]
    
    # 線形従属関係を見つける
    matrix = create_matrix_from_powers(powers)
    null_space = find_null_space(matrix)
    
    # 最小次数の関係から最小多項式を構築
    return construct_minimal_polynomial(null_space[0])
```

#### 2.2 既約性の検証
計算した多項式が実際に既約であることを確認：

```python
def verify_minimal_polynomial(polynomial, element):
    # 多項式がelementを根とすることを確認
    if not polynomial.evaluate(element).is_zero():
        return False
    
    # 既約性を確認
    return is_irreducible(polynomial)
```

#### 2.3 特殊ケースの最適化
よく使用される代数的元の最小多項式を事前計算：

- √n の場合: x² - n
- ∛n の場合: x³ - n
- 1の原始n乗根の場合: 円分多項式

### Week 3: 複合拡大の実装

#### 3.1 塔状拡大の構築
K ⊆ L ⊆ M の形の拡大を効率的に管理：

```python
class TowerExtension:
    def __init__(self, base_field):
        self.base_field = base_field
        self.extensions = []  # 拡大の履歴
        self.current_field = base_field
    
    def add_extension(self, minimal_polynomial):
        new_extension = FieldExtension(self.current_field, minimal_polynomial)
        self.extensions.append(new_extension)
        self.current_field = new_extension
        return new_extension
```

#### 3.2 原始元定理の活用
複合拡大 K(α, β) を単純拡大 K(γ) として表現：

```python
def find_primitive_element(alpha, beta):
    # α + cβ の形で原始元を探索
    for c in range(1, 100):  # 適当な範囲で探索
        candidate = alpha + c * beta
        if is_primitive_element(candidate, alpha, beta):
            return candidate
    raise ValueError("原始元が見つかりませんでした")
```

#### 3.3 分解体の構築
多項式の全ての根を含む最小の体を構築：

```python
def splitting_field(polynomial, base_field):
    current_field = base_field
    remaining_polynomial = polynomial
    
    while remaining_polynomial.degree() > 1:
        # 既約因子を見つける
        irreducible_factor = find_irreducible_factor(remaining_polynomial)
        
        # 拡大を追加
        current_field = current_field.extend(irreducible_factor)
        
        # 多項式を更新
        remaining_polynomial = remaining_polynomial.factor_over(current_field)
    
    return current_field
```

### Week 4: 高度な拡大理論の実装

#### 4.1 ガロア拡大の判定
拡大が正規かつ分離可能であることを確認：

```python
def is_galois_extension(extension):
    # 正規性の確認：最小多項式が完全に分解する
    if not extension.minimal_polynomial.splits_completely():
        return False
    
    # 分離可能性の確認：重根を持たない
    if extension.minimal_polynomial.has_multiple_roots():
        return False
    
    return True
```

#### 4.2 中間体の計算
拡大の中間体を系統的に列挙：

```python
def find_intermediate_fields(extension):
    intermediate_fields = []
    degree = extension.degree
    
    # 次数の約数に対応する中間体を探索
    for d in divisors(degree):
        if d > 1 and d < degree:
            intermediate_field = find_intermediate_field_of_degree(extension, d)
            if intermediate_field:
                intermediate_fields.append(intermediate_field)
    
    return intermediate_fields
```

#### 4.3 共役元の計算
代数的元の全ての共役元を計算：

```python
def find_conjugates(element, base_field):
    minimal_poly = element.minimal_polynomial()
    roots = minimal_poly.roots_in_splitting_field()
    return [root for root in roots if root != element]
```

## 理論的な深掘り

### 体の拡大における基本定理

#### 原始元定理
有限拡大 L/K において、L = K(α) となる α が存在します。この α を原始元と呼びます。

**証明の概要**:
1. 有限体上では常に成立
2. 無限体上では、適切な線形結合が原始元となる

#### 拡大の乗法性
K ⊆ L ⊆ M について、[M:K] = [M:L] × [L:K] が成立します。

**意義**: 複雑な拡大を段階的に理解できます

### 5次方程式との関連

#### 分解体の構造
5次多項式 f(x) の分解体 K(α₁, α₂, α₃, α₄, α₅) は、以下の性質を持ちます：

1. **最大次数**: [K(α₁, α₂, α₃, α₄, α₅):K] ≤ 5! = 120
2. **ガロア群**: 分解体のガロア群は S₅ の部分群
3. **中間体**: 可解な場合のみ、適切な中間体の塔が存在

#### 具体例による理解

**例1: x⁵ - 2 = 0**
- 分解体: Q(∛2, ζ₅)（ζ₅ は1の原始5乗根）
- 拡大次数: [Q(∛2, ζ₅):Q] = 20
- ガロア群: フロベニウス群 F₂₀（可解群）

**例2: x⁵ - 4x + 2 = 0**
- 分解体: より複雑な構造
- ガロア群: S₅（非可解群）
- 根号による解の表現は不可能

### 計算例

#### √2 の拡大
Q(√2) = {a + b√2 | a, b ∈ Q}

**演算例**:
```
(1 + √2) × (2 + 3√2) = 2 + 3√2 + 2√2 + 3×2 = 8 + 5√2
```

#### ∛2 の拡大
Q(∛2) = {a + b∛2 + c∛4 | a, b, c ∈ Q}

**最小多項式**: x³ - 2
**基底**: {1, ∛2, ∛4}

## 実装上の注意点

### 数値精度の管理

代数的元の演算では、係数が複雑な分数になる可能性があります：

1. **有理数演算**: 常に正確な有理数で計算
2. **分母の管理**: 共通分母を効率的に計算
3. **約分**: 演算後の自動約分

### メモリ効率の最適化

拡大体の元は多くのメモリを消費する可能性があります：

1. **疎表現**: 零係数の項を省略
2. **共有**: 同じ拡大体の元で共通部分を共有
3. **キャッシュ**: 頻繁に使用される演算結果を保存

### エラーハンドリング

体の拡大では様々なエラーが発生する可能性があります：

1. **非代数的元**: 超越元を代数的元として扱う
2. **次数の不整合**: 期待した次数と異なる拡大
3. **最小多項式の計算失敗**: 数値的不安定性

## 次のフェーズへの準備

Phase 3で実装した体の拡大理論は、Phase 4の群論実装で活用されます：

1. **ガロア群の台集合**: 拡大体の自己同型写像
2. **固定体**: 群の作用による不変元
3. **ガロア対応**: 中間体と部分群の対応

## 学習のポイント

### 抽象代数の具体化

体の拡大理論を実装することで、抽象的な概念を具体的に理解できます：

1. **代数的元の実体**: 最小多項式による表現
2. **拡大の構造**: 段階的な構築過程
3. **演算の意味**: 剰余演算としての体の演算

### アルゴリズムの洞察

このフェーズでは、数学的アルゴリズムの本質を理解できます：

1. **線形代数の応用**: 最小多項式の計算
2. **多項式演算の活用**: 剰余演算による体の構築
3. **効率化の技法**: 特殊ケースの最適化

## 成果物の活用

Phase 3で作成される成果物：

1. **`field_extension.py`**: 体の拡大の基本クラス
2. **`extension_element.py`**: 拡大体の元のクラス
3. **`minimal_polynomial.py`**: 最小多項式の計算
4. **`splitting_field.py`**: 分解体の構築
5. **`test_extensions.py`**: 拡大体の動作テスト

このフェーズを完了することで、5次方程式の根を含む体の構造を理解し、次のフェーズでガロア群の計算に進む準備が整います。特に、分解体の構築能力は、ガロア群の定義域となる体を提供します。 