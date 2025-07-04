# Phase 5: ガロア群の計算

## 概要と目的

ガロア群の計算は、ガロア理論実装の最終段階であり、5次方程式の可解性判定の核心部分です。このフェーズでは、多項式からそのガロア群を実際に計算し、群の構造を解析することで、方程式が根号で解けるかどうかを判定します。

ガロア群の計算を通じて、「なぜ一般的な5次方程式は解けないのか」「どのような特殊な5次方程式なら解けるのか」という問いに対する具体的で計算可能な答えを得ることができます。

## 理論的背景

### ガロア群とは何か

多項式 f(x) ∈ K[x] に対して、その分解体 L における K-自己同型写像の全体がガロア群 Gal(L/K) を形成します。

**具体的な定義**:
- L: f(x) の分解体（f(x) の全ての根を含む最小の体）
- σ: L → L の体同型写像で、K の全ての元を固定する
- Gal(L/K) = {σ | σ は L の K-自己同型写像}

### ガロア群の性質

#### 1. 群構造
ガロア群は写像の合成を演算として群になります：
- 単位元: 恒等写像
- 逆元: 逆写像
- 結合法則: 写像の合成の結合法則

#### 2. 根の置換
ガロア群の元は、多項式の根を置換する作用を持ちます：
- f(x) の根を α₁, α₂, ..., αₙ とする
- σ ∈ Gal(L/K) は αᵢ を αⱼ に写す
- この作用により、ガロア群は対称群の部分群と見なせる

#### 3. 基本定理
**ガロアの基本定理**: 有限ガロア拡大 L/K について、
- 中間体と部分群の間に1対1対応が存在
- 正規部分群は正規拡大に対応
- 可解群は根号拡大に対応

### 5次方程式の特殊性

#### 一般的な5次方程式
一般的な5次多項式 x⁵ + ax⁴ + bx³ + cx² + dx + e のガロア群は S₅ です。

**重要な事実**:
- |S₅| = 120
- S₅ は非可解群
- したがって、一般的な5次方程式は根号で解けない

#### 特殊な可解例
以下のような特殊な場合は可解です：
- **x⁵ - a**: ガロア群は巡回群またはフロベニウス群
- **既約でない場合**: より低次の方程式に分解可能
- **特殊な係数**: 対称性により可解群となる場合

## 実装の戦略

### ガロア群計算の段階

ガロア群の計算は以下の段階で行います：

1. **分解体の構築**: 多項式の全ての根を含む体を構築
2. **自己同型写像の発見**: 基本体を固定する体同型写像を見つける
3. **群構造の解析**: 見つけた自己同型写像が作る群の構造を調べる
4. **可解性の判定**: 群が可解かどうかを判定

### 実装上の課題と解決策

#### 課題1: 分解体の構築
多項式の全ての根を含む体を効率的に構築する必要があります。

**解決策**:
- 段階的な体の拡大
- 既約因子による拡大の繰り返し
- 数値的近似と代数的厳密性の両立

#### 課題2: 自己同型写像の発見
無限に多くの可能性から実際の自己同型写像を見つける必要があります。

**解決策**:
- 生成元の写像による決定
- 最小多項式の根の置換
- 体の構造を保つ写像の条件

#### 課題3: 群の同型判定
計算したガロア群が既知の群のどれに同型かを判定する必要があります。

**解決策**:
- 群の不変量（位数、共役類など）による分類
- 既知の群との比較
- 生成元と関係式による表現

## 具体的な実装手順

### Week 1: 基本的なガロア群の計算

#### 1.1 GaloisGroupクラスの設計
ガロア群を表現し、計算するクラス：

```python
class GaloisGroup:
    def __init__(self, polynomial, base_field):
        self.polynomial = polynomial
        self.base_field = base_field
        self.splitting_field = self._construct_splitting_field()
        self.automorphisms = self._find_automorphisms()
        self.group = self._construct_group()
    
    def _construct_splitting_field(self):
        """分解体を構築"""
        current_field = self.base_field
        remaining_polynomial = self.polynomial
        
        while remaining_polynomial.degree() > 1:
            # 既約因子を見つける
            irreducible_factor = self._find_irreducible_factor(remaining_polynomial)
            
            # 体を拡大
            current_field = current_field.extend(irreducible_factor)
            
            # 多項式を因数分解
            remaining_polynomial = remaining_polynomial.factor_over(current_field)
        
        return current_field
    
    def _find_automorphisms(self):
        """自己同型写像を見つける"""
        automorphisms = []
        
        # 分解体の生成元を特定
        generators = self.splitting_field.generators()
        
        # 各生成元の可能な写像先を計算
        for mapping in self._enumerate_mappings(generators):
            if self._is_valid_automorphism(mapping):
                automorphisms.append(mapping)
        
        return automorphisms
    
    def _construct_group(self):
        """自己同型写像から群を構築"""
        def composition(f, g):
            return lambda x: f(g(x))
        
        return Group(self.automorphisms, composition)
```

#### 1.2 自己同型写像の実装
体の自己同型写像を効率的に表現：

```python
class FieldAutomorphism:
    def __init__(self, field_extension, root_mapping):
        self.field_extension = field_extension
        self.root_mapping = root_mapping  # 根の写像を辞書で表現
        self.inverse_mapping = self._compute_inverse()
    
    def __call__(self, element):
        """元素に自己同型写像を適用"""
        if element in self.field_extension.base_field:
            return element  # 基本体の元は固定
        
        # 拡大体の元を基底の線形結合として表現
        coefficients = element.coefficients
        basis = self.field_extension.basis
        
        result = self.field_extension.zero()
        for i, coeff in enumerate(coefficients):
            mapped_basis = self._map_basis_element(basis[i])
            result += coeff * mapped_basis
        
        return result
    
    def _map_basis_element(self, basis_element):
        """基底元素の写像"""
        # 基底元素を根の多項式として表現し、根の写像を適用
        if basis_element in self.root_mapping:
            return self.root_mapping[basis_element]
        
        # 複合的な基底元素の場合
        return self._compute_composite_mapping(basis_element)
    
    def compose(self, other):
        """自己同型写像の合成"""
        new_mapping = {}
        for root, image in self.root_mapping.items():
            new_mapping[root] = self(other(root))
        return FieldAutomorphism(self.field_extension, new_mapping)
    
    def inverse(self):
        """逆写像"""
        return FieldAutomorphism(self.field_extension, self.inverse_mapping)
```

#### 1.3 根の置換による表現
ガロア群を根の置換として表現：

```python
def galois_group_as_permutation_group(galois_group):
    """ガロア群を置換群として表現"""
    roots = galois_group.polynomial.roots_in_splitting_field()
    permutations = []
    
    for automorphism in galois_group.automorphisms:
        # 各根の写像先を計算
        root_images = {}
        for i, root in enumerate(roots):
            image = automorphism(root)
            # 画像が根のリストの何番目かを見つける
            for j, other_root in enumerate(roots):
                if image == other_root:
                    root_images[i+1] = j+1
                    break
        
        permutation = Permutation(root_images)
        permutations.append(permutation)
    
    return SymmetricGroup(len(roots)).subgroup(permutations)
```

### Week 2: 特殊な多項式のガロア群

#### 2.1 既約多項式のガロア群
既約多項式に対する効率的なガロア群計算：

```python
def irreducible_polynomial_galois_group(polynomial, base_field):
    """既約多項式のガロア群を計算"""
    degree = polynomial.degree()
    
    # 特殊ケースの処理
    if degree == 1:
        return Group([identity_map], lambda f, g: f)
    
    if degree == 2:
        return quadratic_galois_group(polynomial, base_field)
    
    if degree == 3:
        return cubic_galois_group(polynomial, base_field)
    
    if degree == 4:
        return quartic_galois_group(polynomial, base_field)
    
    if degree == 5:
        return quintic_galois_group(polynomial, base_field)
    
    # 一般的な場合
    return general_galois_group(polynomial, base_field)

def quadratic_galois_group(polynomial, base_field):
    """2次多項式のガロア群"""
    # ax² + bx + c の判別式を計算
    a, b, c = polynomial.coefficients
    discriminant = b**2 - 4*a*c
    
    if discriminant.is_square_in(base_field):
        # 判別式が平方数なら自明群
        return Group([identity_map], lambda f, g: f)
    else:
        # そうでなければ位数2の群
        return Group([identity_map, conjugation_map], composition)

def cubic_galois_group(polynomial, base_field):
    """3次多項式のガロア群"""
    discriminant = polynomial.discriminant()
    
    if discriminant.is_square_in(base_field):
        # 判別式が平方数なら A₃ (巡回群)
        return AlternatingGroup(3)
    else:
        # そうでなければ S₃
        return SymmetricGroup(3)
```

#### 2.2 5次多項式の特殊ケース
5次多項式の重要な特殊ケースを実装：

```python
def quintic_galois_group(polynomial, base_field):
    """5次多項式のガロア群を計算"""
    # まず既約性を確認
    if not polynomial.is_irreducible():
        return reducible_polynomial_galois_group(polynomial, base_field)
    
    # 特殊な形式をチェック
    if is_binomial_form(polynomial):
        return binomial_quintic_galois_group(polynomial, base_field)
    
    if is_solvable_quintic_form(polynomial):
        return solvable_quintic_galois_group(polynomial, base_field)
    
    # 一般的な場合（通常は S₅）
    return general_quintic_galois_group(polynomial, base_field)

def binomial_quintic_galois_group(polynomial, base_field):
    """x⁵ - a 形式の5次多項式のガロア群"""
    # polynomial = x⁵ - a
    a = polynomial.constant_term()
    
    # 1の5乗根が基本体に含まれるかチェック
    if primitive_5th_root_in_field(base_field):
        # 含まれる場合は巡回群 C₅
        return CyclicGroup(5)
    else:
        # 含まれない場合はフロベニウス群 F₂₀
        return FrobeniusGroup(20)

def solvable_quintic_galois_group(polynomial, base_field):
    """特殊な可解5次多項式のガロア群"""
    # 具体的な形式に応じて判定
    if has_rational_root(polynomial):
        # 有理根を持つ場合
        return factored_quintic_galois_group(polynomial, base_field)
    
    if is_dihedral_type(polynomial):
        # 二面体群型
        return DihedralGroup(5)
    
    if is_metacyclic_type(polynomial):
        # メタサイクリック群型
        return MetacyclicGroup(20)
    
    # その他の可解ケース
    return analyze_solvable_structure(polynomial, base_field)
```

#### 2.3 ガロア群の同型判定
計算したガロア群の同型類を判定：

```python
def identify_galois_group(group):
    """ガロア群の同型類を判定"""
    order = group.order
    
    # 位数による一次分類
    if order == 1:
        return "自明群"
    elif order == 2:
        return "巡回群 C₂"
    elif order == 6:
        return classify_order_6_group(group)
    elif order == 12:
        return classify_order_12_group(group)
    elif order == 20:
        return classify_order_20_group(group)
    elif order == 24:
        return classify_order_24_group(group)
    elif order == 60:
        return classify_order_60_group(group)
    elif order == 120:
        return classify_order_120_group(group)
    else:
        return f"位数{order}の群"

def classify_order_20_group(group):
    """位数20の群を分類"""
    # 可能な群: C₂₀, D₁₀, F₂₀
    
    if group.is_cyclic():
        return "巡回群 C₂₀"
    
    if group.is_dihedral():
        return "二面体群 D₁₀"
    
    if group.is_frobenius():
        return "フロベニウス群 F₂₀"
    
    return "位数20の未知の群"

def classify_order_120_group(group):
    """位数120の群を分類"""
    # 主に S₅ が対象
    
    if group.is_symmetric_group(5):
        return "対称群 S₅"
    
    # その他の位数120の群は稀
    return "位数120の群（S₅以外）"
```

### Week 3: ガロア対応の実装

#### 3.1 中間体と部分群の対応
ガロアの基本定理を実装：

```python
class GaloisCorrespondence:
    def __init__(self, galois_group, splitting_field, base_field):
        self.galois_group = galois_group
        self.splitting_field = splitting_field
        self.base_field = base_field
        self.correspondence = self._compute_correspondence()
    
    def _compute_correspondence(self):
        """中間体と部分群の対応を計算"""
        correspondence = {}
        
        # 全ての部分群を列挙
        subgroups = self._find_all_subgroups()
        
        for subgroup in subgroups:
            # 部分群に対応する固定体を計算
            fixed_field = self._compute_fixed_field(subgroup)
            correspondence[subgroup] = fixed_field
        
        return correspondence
    
    def _compute_fixed_field(self, subgroup):
        """部分群の固定体を計算"""
        # 部分群の全ての元で固定される元を見つける
        fixed_elements = []
        
        for element in self.splitting_field.elements():
            is_fixed = True
            for automorphism in subgroup.elements:
                if automorphism(element) != element:
                    is_fixed = False
                    break
            
            if is_fixed:
                fixed_elements.append(element)
        
        return Field(fixed_elements)
    
    def subgroup_to_field(self, subgroup):
        """部分群から対応する中間体を取得"""
        return self.correspondence[subgroup]
    
    def field_to_subgroup(self, intermediate_field):
        """中間体から対応する部分群を取得"""
        for subgroup, field in self.correspondence.items():
            if field == intermediate_field:
                return subgroup
        return None
    
    def is_normal_extension(self, subgroup):
        """部分群に対応する拡大が正規拡大かを判定"""
        return subgroup.is_normal_subgroup()
    
    def is_galois_extension(self, subgroup):
        """部分群に対応する拡大がガロア拡大かを判定"""
        return (self.is_normal_extension(subgroup) and 
                self.is_separable_extension(subgroup))
```

#### 3.2 可解性の判定
ガロア群の可解性から多項式の可解性を判定：

```python
def is_solvable_by_radicals(galois_group):
    """ガロア群の可解性から根号可解性を判定"""
    return galois_group.is_solvable()

def radical_tower_construction(polynomial, base_field):
    """可解多項式の根号塔を構築"""
    galois_group = compute_galois_group(polynomial, base_field)
    
    if not galois_group.is_solvable():
        raise ValueError("多項式は根号で解けません")
    
    # 可解列を構築
    solvable_series = galois_group.solvable_series()
    
    # 対応する体の塔を構築
    field_tower = []
    current_field = base_field
    
    for i in range(len(solvable_series) - 1):
        subgroup = solvable_series[i+1]
        quotient = solvable_series[i].quotient_group(subgroup)
        
        # 商群がアーベル群なので、根号拡大が可能
        if quotient.is_cyclic():
            # 巡回拡大の場合
            extension_field = construct_cyclic_extension(current_field, quotient.order)
        else:
            # 一般のアーベル拡大の場合
            extension_field = construct_abelian_extension(current_field, quotient)
        
        field_tower.append(extension_field)
        current_field = extension_field
    
    return field_tower

def construct_cyclic_extension(base_field, degree):
    """巡回拡大を構築"""
    # n次単位根を添加
    if degree == 2:
        # 平方根拡大
        return add_square_root(base_field)
    elif degree == 3:
        # 立方根拡大
        return add_cube_root(base_field)
    elif degree == 5:
        # 5乗根拡大
        return add_fifth_root(base_field)
    else:
        # 一般的なn乗根拡大
        return add_nth_root(base_field, degree)
```

### Week 4: 実用的な可解性判定

#### 4.1 QuinticSolverクラスの実装
5次方程式の可解性を判定する統合クラス：

```python
class QuinticSolver:
    def __init__(self, polynomial, base_field=RationalField()):
        self.polynomial = polynomial
        self.base_field = base_field
        self.galois_group = None
        self.solvable = None
        self.solution_type = None
    
    def analyze_solvability(self):
        """可解性の完全な解析"""
        # Step 1: 基本的な前処理
        self._preprocess_polynomial()
        
        # Step 2: 簡単なケースをチェック
        if self._check_trivial_cases():
            return self.solution_type
        
        # Step 3: ガロア群を計算
        self.galois_group = self._compute_galois_group()
        
        # Step 4: 可解性を判定
        self.solvable = self.galois_group.is_solvable()
        
        # Step 5: 解の構造を分析
        self._analyze_solution_structure()
        
        return self.solution_type
    
    def _preprocess_polynomial(self):
        """多項式の前処理"""
        # 最高次係数を1に正規化
        self.polynomial = self.polynomial.monic()
        
        # 有理根の存在チェック
        self.rational_roots = self.polynomial.rational_roots()
        
        # 既約性の確認
        self.is_irreducible = self.polynomial.is_irreducible()
    
    def _check_trivial_cases(self):
        """簡単なケースのチェック"""
        # 有理根を持つ場合
        if self.rational_roots:
            self.solution_type = "有理根あり（因数分解可能）"
            self.solvable = True
            return True
        
        # 既約でない場合
        if not self.is_irreducible:
            self.solution_type = "既約でない（低次方程式に分解）"
            self.solvable = True
            return True
        
        # 特殊な形式
        if self._is_binomial():
            self.solution_type = "二項式（x⁵ - a 型）"
            self.solvable = True
            return True
        
        return False
    
    def _compute_galois_group(self):
        """ガロア群の計算"""
        return GaloisGroup(self.polynomial, self.base_field)
    
    def _analyze_solution_structure(self):
        """解の構造分析"""
        if not self.solvable:
            self.solution_type = "非可解（根号では解けない）"
            return
        
        # 可解な場合の詳細分析
        group_type = identify_galois_group(self.galois_group.group)
        
        if "巡回群" in group_type:
            self.solution_type = f"巡回型可解（{group_type}）"
        elif "二面体群" in group_type:
            self.solution_type = f"二面体型可解（{group_type}）"
        elif "フロベニウス群" in group_type:
            self.solution_type = f"フロベニウス型可解（{group_type}）"
        else:
            self.solution_type = f"一般可解型（{group_type}）"
    
    def get_solution_method(self):
        """解法の提案"""
        if not self.solvable:
            return "数値解法または特殊関数による解法が必要"
        
        if "有理根" in self.solution_type:
            return "有理根定理による因数分解"
        
        if "二項式" in self.solution_type:
            return "n乗根による直接解法"
        
        if "巡回型" in self.solution_type:
            return "巡回拡大による根号解法"
        
        if "二面体型" in self.solution_type:
            return "二面体群の構造を利用した解法"
        
        if "フロベニウス型" in self.solution_type:
            return "フロベニウス群の理論による解法"
        
        return "一般的な可解群の理論による解法"
```

#### 4.2 具体例による検証
実際の5次多項式での検証：

```python
def test_quintic_examples():
    """5次多項式の具体例でテスト"""
    examples = [
        # 可解な例
        ("x^5 - 2", "フロベニウス群F₂₀（可解）"),
        ("x^5 - 5*x + 12", "特殊な可解例"),
        ("x^5 - 1", "巡回群C₅（可解）"),
        ("x^5 + x^4 - 4*x^3 - 3*x^2 + 3*x + 1", "二面体群D₅（可解）"),
        
        # 非可解な例
        ("x^5 - 4*x + 2", "対称群S₅（非可解）"),
        ("x^5 + 20*x + 16", "対称群S₅（非可解）"),
        ("x^5 - x - 1", "対称群S₅（非可解）"),
    ]
    
    for poly_str, expected in examples:
        polynomial = parse_polynomial(poly_str)
        solver = QuinticSolver(polynomial)
        result = solver.analyze_solvability()
        
        print(f"多項式: {poly_str}")
        print(f"ガロア群: {solver.galois_group.group}")
        print(f"可解性: {'可解' if solver.solvable else '非可解'}")
        print(f"解法: {solver.get_solution_method()}")
        print(f"期待値: {expected}")
        print("-" * 50)

def demonstrate_galois_correspondence():
    """ガロア対応の具体例"""
    # x⁵ - 2 の場合
    polynomial = Polynomial([0, 0, 0, 0, 0, -2, 1])  # x⁵ - 2
    galois_group = GaloisGroup(polynomial, RationalField())
    
    correspondence = GaloisCorrespondence(
        galois_group, 
        galois_group.splitting_field, 
        RationalField()
    )
    
    print("ガロア対応の例: x⁵ - 2")
    print(f"ガロア群: {galois_group.group}")
    print(f"分解体: Q(∛2, ζ₅)")
    print()
    
    # 部分群と中間体の対応を表示
    for subgroup, field in correspondence.correspondence.items():
        print(f"部分群 {subgroup} ←→ 中間体 {field}")
        print(f"  指数: {subgroup.index()}")
        print(f"  拡大次数: {field.degree_over_base()}")
        print()
```

## 理論的な深掘り

### ガロア理論の威力

#### 古典的問題への応用
ガロア理論は以下の古典的問題を解決しました：

1. **5次方程式の非可解性**: 一般的な5次方程式は根号で解けない
2. **作図問題**: 角の3等分、立方体の倍積、円の正方形化は不可能
3. **正多角形の作図**: 作図可能な正多角形の完全な分類

#### 現代数学への影響
ガロア理論は現代数学の多くの分野に影響を与えています：

1. **代数的数論**: 類体論の基礎
2. **代数幾何**: エタール・コホモロジー
3. **表現論**: 群の表現と保型形式
4. **暗号理論**: 楕円曲線暗号の理論的基盤

### 計算複雑性の考察

#### ガロア群計算の複雑性
ガロア群の計算は一般に困難な問題です：

1. **指数時間**: 最悪の場合、指数時間が必要
2. **特殊ケース**: 多くの実用的なケースでは多項式時間
3. **近似アルゴリズム**: 完全な計算が困難な場合の近似手法

#### 実用的な最適化
実装では以下の最適化が重要です：

1. **前処理**: 簡単なケースの事前判定
2. **特殊化**: 特定の多項式クラスに対する専用アルゴリズム
3. **並列化**: 独立な計算の並列実行
4. **キャッシュ**: 中間結果の再利用

## 実装上の注意点

### 数値安定性

ガロア群の計算では数値的な問題が発生する可能性があります：

1. **代数的数の表現**: 正確な代数的数の算術
2. **近似誤差**: 数値計算による誤差の蓄積
3. **判定の閾値**: 等値判定における許容誤差

### 計算効率

大きな次数の多項式では計算効率が重要です：

1. **メモリ使用量**: 大きな体や群の効率的な表現
2. **計算時間**: アルゴリズムの最適化
3. **早期終了**: 不必要な計算の回避

### エラーハンドリング

複雑な計算では様々なエラーが発生する可能性があります：

1. **計算失敗**: 数値的不安定性による失敗
2. **メモリ不足**: 大きな問題でのメモリ不足
3. **タイムアウト**: 長時間計算の中断

## 成果物の活用

Phase 5で作成される成果物：

1. **`galois_group.py`**: ガロア群の基本クラス
2. **`automorphism.py`**: 体の自己同型写像
3. **`galois_correspondence.py`**: ガロア対応の実装
4. **`quintic_solver.py`**: 5次方程式の可解性判定
5. **`examples.py`**: 具体例と検証
6. **`test_galois.py`**: ガロア理論の動作テスト

## 学習のポイント

### 理論と実装の統合

このフェーズでは、抽象的なガロア理論と具体的な計算を統合します：

1. **概念の具体化**: 抽象的な概念の計算可能な実装
2. **理論の検証**: 実際の計算による理論の確認
3. **直感の獲得**: 計算を通じた理論的直感の獲得

### 問題解決能力の向上

ガロア群の計算を通じて、複雑な数学的問題を解決する能力を向上させます：

1. **分析力**: 問題の構造を分析する能力
2. **実装力**: 理論を実装に落とし込む能力
3. **検証力**: 結果の正しさを検証する能力

このフェーズを完了することで、5次方程式の可解性判定という具体的な問題を通じて、ガロア理論の深い理解を得ることができます。理論と実装の両面から学ぶことで、現代数学の美しさと実用性を実感できるでしょう。 