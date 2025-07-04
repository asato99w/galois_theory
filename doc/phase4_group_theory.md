# Phase 4: 群論の実装

## 概要と目的

群論は、ガロア理論の数学的基盤を提供する重要な分野です。このフェーズでは、群の基本構造から始めて、対称群、部分群、正規部分群、そして可解群の概念を実装します。5次方程式の可解性判定において、群論は「なぜ一般的な5次方程式が根号で解けないのか」という根本的な問いに答える鍵となります。

群論の実装を通じて、代数的構造の美しさと、それが具体的な数学的問題（5次方程式の可解性）にどのように応用されるかを理解できるようになります。

## 理論的背景

### 群とは何か

群(Group)は、以下の4つの条件を満たす集合Gと二項演算*の組(G, *)です：

1. **結合法則**: 任意の a, b, c ∈ G に対して (a * b) * c = a * (b * c)
2. **単位元の存在**: e ∈ G が存在し、任意の a ∈ G に対して e * a = a * e = a
3. **逆元の存在**: 任意の a ∈ G に対して a⁻¹ ∈ G が存在し、a * a⁻¹ = a⁻¹ * a = e
4. **演算の閉性**: 任意の a, b ∈ G に対して a * b ∈ G

### 群の具体例

#### 対称群 Sₙ
n個の要素の置換全体が作る群です。5次方程式の場合、S₅が重要な役割を果たします。

**例**: S₃ = {e, (1 2), (1 3), (2 3), (1 2 3), (1 3 2)}
- 位数: |S₃| = 3! = 6
- 演算: 置換の合成

#### 巡回群 Cₙ
一つの元によって生成される群です。

**例**: C₄ = {e, a, a², a³} (a⁴ = e)
- 位数: |C₄| = 4
- 演算: 元の累乗

#### 二面体群 Dₙ
正n角形の対称性を表す群です。

**例**: D₃ = {e, r, r², s, sr, sr²} (r³ = s² = e, srs = r²)
- 位数: |D₃| = 6
- 演算: 回転と反射の合成

### 部分群と正規部分群

#### 部分群
群Gの部分集合Hが、Gの演算に関して群になるとき、HをGの部分群と呼びます。

**ラグランジュの定理**: 有限群Gの部分群Hについて、|H| は |G| を割り切ります。

#### 正規部分群
部分群Nが、任意の g ∈ G に対して gNg⁻¹ = N を満たすとき、Nを正規部分群と呼びます。

**重要性**: 正規部分群は剰余群 G/N を定義するために必要です。

### 可解群の概念

群Gが可解であるとは、以下の条件を満たすことです：

G = G₀ ⊃ G₁ ⊃ G₂ ⊃ ... ⊃ Gₖ = {e}

各 Gᵢ₊₁ は Gᵢ の正規部分群であり、剰余群 Gᵢ/Gᵢ₊₁ がアーベル群となる。

**ガロア理論との関連**: 多項式が根号で解けることと、そのガロア群が可解であることは同値です。

## 実装の戦略

### なぜ一般的な群論から実装するのか

5次方程式に特化した実装も可能ですが、一般的な群論から始める理由：

1. **理論的基盤**: ガロア理論の完全な理解には一般的な群論が必要
2. **拡張性**: 他の次数の方程式や、より複雑な代数的構造への応用が可能
3. **アルゴリズムの理解**: 群論アルゴリズムの本質を理解できる
4. **デバッグ容易性**: 小さな群で動作を確認してから複雑な群に適用

### 実装上の課題と解決策

#### 課題1: 群元の表現
群の元をコンピュータ上でどう表現するかという問題があります。

**解決策**: 
- 置換群: 置換の配列表現
- 行列群: 行列による表現
- 抽象群: 乗積表による表現

#### 課題2: 演算の効率性
群の演算、特に置換の合成は計算量が多くなる可能性があります。

**解決策**:
- 効率的な置換アルゴリズム
- 演算結果のキャッシュ
- 特殊ケースの最適化

#### 課題3: 部分群の列挙
大きな群の全ての部分群を列挙することは計算量的に困難です。

**解決策**:
- 生成元による部分群の表現
- ラグランジュの定理による剪定
- 既知の部分群構造の活用

## 具体的な実装手順

### Week 1: 基本的な群構造の実装

#### 1.1 Groupクラスの設計
群の基本的な構造を表現するクラス：

```python
class Group:
    def __init__(self, elements, operation):
        self.elements = elements
        self.operation = operation
        self.identity = self._find_identity()
        self.order = len(elements)
        self._verify_group_axioms()
    
    def _find_identity(self):
        # 単位元を探索
        for e in self.elements:
            if all(self.operation(e, a) == a for a in self.elements):
                return e
        raise ValueError("単位元が見つかりません")
```

#### 1.2 GroupElementクラスの実装
群の元を表現するクラス：

```python
class GroupElement:
    def __init__(self, value, group):
        self.value = value
        self.group = group
    
    def __mul__(self, other):
        if self.group != other.group:
            raise ValueError("異なる群の元同士は演算できません")
        result = self.group.operation(self.value, other.value)
        return GroupElement(result, self.group)
    
    def inverse(self):
        for g in self.group.elements:
            if self.group.operation(self.value, g) == self.group.identity:
                return GroupElement(g, self.group)
        raise ValueError("逆元が見つかりません")
```

#### 1.3 群の公理の検証
実装した群が実際に群の公理を満たすかを確認：

```python
def _verify_group_axioms(self):
    # 結合法則の確認
    for a in self.elements:
        for b in self.elements:
            for c in self.elements:
                left = self.operation(self.operation(a, b), c)
                right = self.operation(a, self.operation(b, c))
                if left != right:
                    raise ValueError(f"結合法則が成立しません: {a}, {b}, {c}")
    
    # 単位元の確認
    for a in self.elements:
        if (self.operation(self.identity, a) != a or 
            self.operation(a, self.identity) != a):
            raise ValueError(f"単位元が正しくありません: {a}")
    
    # 逆元の確認
    for a in self.elements:
        inverse_found = False
        for b in self.elements:
            if (self.operation(a, b) == self.identity and 
                self.operation(b, a) == self.identity):
                inverse_found = True
                break
        if not inverse_found:
            raise ValueError(f"逆元が見つかりません: {a}")
```

#### 1.4 基本的な群の構築
よく知られた群の実装：

```python
def create_cyclic_group(n):
    """位数nの巡回群を作成"""
    elements = list(range(n))
    def operation(a, b):
        return (a + b) % n
    return Group(elements, operation)

def create_klein_four_group():
    """クライン4群を作成"""
    elements = ['e', 'a', 'b', 'c']
    multiplication_table = {
        ('e', 'e'): 'e', ('e', 'a'): 'a', ('e', 'b'): 'b', ('e', 'c'): 'c',
        ('a', 'e'): 'a', ('a', 'a'): 'e', ('a', 'b'): 'c', ('a', 'c'): 'b',
        ('b', 'e'): 'b', ('b', 'a'): 'c', ('b', 'b'): 'e', ('b', 'c'): 'a',
        ('c', 'e'): 'c', ('c', 'a'): 'b', ('c', 'b'): 'a', ('c', 'c'): 'e'
    }
    def operation(a, b):
        return multiplication_table[(a, b)]
    return Group(elements, operation)
```

### Week 2: 対称群の実装

#### 2.1 Permutationクラスの実装
置換を効率的に表現するクラス：

```python
class Permutation:
    def __init__(self, mapping):
        """
        mapping: 辞書形式の置換 {1: 2, 2: 3, 3: 1} または
                リスト形式 [2, 3, 1] (1-indexed)
        """
        if isinstance(mapping, dict):
            self.mapping = mapping
            self.size = max(mapping.keys())
        else:
            self.mapping = {i+1: mapping[i] for i in range(len(mapping))}
            self.size = len(mapping)
    
    def __call__(self, x):
        """置換を関数として適用"""
        return self.mapping.get(x, x)
    
    def __mul__(self, other):
        """置換の合成"""
        result = {}
        for i in range(1, max(self.size, other.size) + 1):
            result[i] = self(other(i))
        return Permutation(result)
    
    def inverse(self):
        """逆置換"""
        result = {}
        for key, value in self.mapping.items():
            result[value] = key
        return Permutation(result)
    
    def cycle_decomposition(self):
        """サイクル分解"""
        visited = set()
        cycles = []
        
        for start in range(1, self.size + 1):
            if start in visited:
                continue
            
            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = self(current)
            
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
        
        return cycles
```

#### 2.2 SymmetricGroupクラスの実装
対称群の完全な実装：

```python
class SymmetricGroup(Group):
    def __init__(self, n):
        self.n = n
        self.elements = self._generate_all_permutations(n)
        super().__init__(self.elements, self._permutation_multiplication)
    
    def _generate_all_permutations(self, n):
        """n次対称群の全ての置換を生成"""
        from itertools import permutations
        perms = []
        for perm in permutations(range(1, n + 1)):
            perms.append(Permutation(list(perm)))
        return perms
    
    def _permutation_multiplication(self, perm1, perm2):
        """置換の乗法"""
        return perm1 * perm2
    
    def transpositions(self):
        """全ての互換を返す"""
        transpositions = []
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                mapping = {k: k for k in range(1, self.n + 1)}
                mapping[i] = j
                mapping[j] = i
                transpositions.append(Permutation(mapping))
        return transpositions
    
    def alternating_group(self):
        """交代群を返す"""
        even_perms = []
        for perm in self.elements:
            if self._is_even_permutation(perm):
                even_perms.append(perm)
        return Group(even_perms, self._permutation_multiplication)
    
    def _is_even_permutation(self, perm):
        """置換が偶置換かどうかを判定"""
        cycles = perm.cycle_decomposition()
        inversions = sum(len(cycle) - 1 for cycle in cycles)
        return inversions % 2 == 0
```

#### 2.3 置換の性質の計算
置換の重要な性質を計算する関数：

```python
def permutation_order(perm):
    """置換の位数を計算"""
    cycles = perm.cycle_decomposition()
    if not cycles:
        return 1
    
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    
    return functools.reduce(lcm, [len(cycle) for cycle in cycles])

def permutation_sign(perm):
    """置換の符号を計算"""
    cycles = perm.cycle_decomposition()
    inversions = sum(len(cycle) - 1 for cycle in cycles)
    return 1 if inversions % 2 == 0 else -1

def conjugacy_class(perm, group):
    """置換の共役類を計算"""
    conjugates = set()
    for g in group.elements:
        conjugate = g * perm * g.inverse()
        conjugates.add(conjugate)
    return list(conjugates)
```

### Week 3: 部分群の実装

#### 3.1 Subgroupクラスの実装
部分群を効率的に管理するクラス：

```python
class Subgroup:
    def __init__(self, parent_group, generators):
        self.parent_group = parent_group
        self.generators = generators
        self.elements = self._generate_subgroup()
        self.order = len(self.elements)
    
    def _generate_subgroup(self):
        """生成元から部分群を生成"""
        subgroup_elements = {self.parent_group.identity}
        queue = list(self.generators)
        
        while queue:
            element = queue.pop(0)
            if element in subgroup_elements:
                continue
            
            subgroup_elements.add(element)
            
            # 既存の元との積を計算
            new_elements = []
            for existing in subgroup_elements:
                new_elements.append(self.parent_group.operation(element, existing))
                new_elements.append(self.parent_group.operation(existing, element))
            
            for new_elem in new_elements:
                if new_elem not in subgroup_elements:
                    queue.append(new_elem)
        
        return list(subgroup_elements)
    
    def contains(self, element):
        """元が部分群に含まれるかを判定"""
        return element in self.elements
    
    def index(self):
        """親群における指数を計算"""
        return self.parent_group.order // self.order
    
    def left_cosets(self):
        """左剰余類を計算"""
        cosets = []
        remaining = set(self.parent_group.elements)
        
        while remaining:
            representative = next(iter(remaining))
            coset = []
            for h in self.elements:
                coset_element = self.parent_group.operation(representative, h)
                coset.append(coset_element)
                remaining.discard(coset_element)
            cosets.append(coset)
        
        return cosets
```

#### 3.2 正規部分群の実装
正規部分群の特別な性質を実装：

```python
class NormalSubgroup(Subgroup):
    def __init__(self, parent_group, generators):
        super().__init__(parent_group, generators)
        if not self._is_normal():
            raise ValueError("指定された部分群は正規部分群ではありません")
    
    def _is_normal(self):
        """正規部分群かどうかを判定"""
        for g in self.parent_group.elements:
            for h in self.elements:
                # ghg^(-1) が部分群に含まれるかを確認
                conjugate = self.parent_group.operation(
                    self.parent_group.operation(g, h),
                    self._find_inverse(g)
                )
                if conjugate not in self.elements:
                    return False
        return True
    
    def quotient_group(self):
        """剰余群を構築"""
        cosets = self.left_cosets()
        
        def coset_multiplication(coset1, coset2):
            # 代表元を取って積を計算
            rep1 = coset1[0]
            rep2 = coset2[0]
            product = self.parent_group.operation(rep1, rep2)
            
            # 積が含まれる剰余類を見つける
            for coset in cosets:
                if product in coset:
                    return coset
            
            raise ValueError("剰余類の積が見つかりません")
        
        return Group(cosets, coset_multiplication)
    
    def _find_inverse(self, element):
        """元の逆元を見つける"""
        for g in self.parent_group.elements:
            if (self.parent_group.operation(element, g) == self.parent_group.identity and
                self.parent_group.operation(g, element) == self.parent_group.identity):
                return g
        raise ValueError(f"逆元が見つかりません: {element}")
```

#### 3.3 部分群の格子構造
部分群の包含関係を管理：

```python
class SubgroupLattice:
    def __init__(self, group):
        self.group = group
        self.subgroups = self._find_all_subgroups()
        self.lattice = self._build_lattice()
    
    def _find_all_subgroups(self):
        """全ての部分群を見つける"""
        subgroups = []
        
        # 自明な部分群
        trivial = Subgroup(self.group, [])
        subgroups.append(trivial)
        
        # 単一元で生成される部分群
        for element in self.group.elements:
            if element != self.group.identity:
                cyclic_subgroup = Subgroup(self.group, [element])
                if cyclic_subgroup not in subgroups:
                    subgroups.append(cyclic_subgroup)
        
        # 複数元で生成される部分群（小さな群の場合のみ）
        if self.group.order <= 12:
            for r in range(2, min(4, self.group.order)):
                for generators in itertools.combinations(self.group.elements, r):
                    if self.group.identity not in generators:
                        subgroup = Subgroup(self.group, list(generators))
                        if subgroup not in subgroups:
                            subgroups.append(subgroup)
        
        # 全体群
        full_group = Subgroup(self.group, self.group.elements)
        subgroups.append(full_group)
        
        return subgroups
    
    def _build_lattice(self):
        """部分群の包含関係を構築"""
        lattice = {}
        for subgroup in self.subgroups:
            lattice[subgroup] = {
                'contains': [],
                'contained_in': []
            }
        
        for sub1 in self.subgroups:
            for sub2 in self.subgroups:
                if sub1 != sub2 and self._is_subgroup_of(sub1, sub2):
                    lattice[sub2]['contains'].append(sub1)
                    lattice[sub1]['contained_in'].append(sub2)
        
        return lattice
    
    def _is_subgroup_of(self, sub1, sub2):
        """sub1がsub2の部分群かどうかを判定"""
        return all(element in sub2.elements for element in sub1.elements)
```

### Week 4: 可解群の実装

#### 4.1 可解群の判定
群が可解かどうかを判定するアルゴリズム：

```python
def is_solvable(group):
    """群が可解かどうかを判定"""
    return _is_solvable_recursive(group, [])

def _is_solvable_recursive(group, chain):
    """再帰的に可解性を判定"""
    # 自明群は可解
    if group.order == 1:
        return True
    
    # アーベル群は可解
    if is_abelian(group):
        return True
    
    # 正規部分群を探索
    for subgroup in find_normal_subgroups(group):
        if subgroup.order > 1 and subgroup.order < group.order:
            quotient = subgroup.quotient_group()
            
            # 剰余群がアーベルかどうかを確認
            if is_abelian(quotient):
                # 部分群が可解かどうかを再帰的に確認
                if _is_solvable_recursive(Group(subgroup.elements, group.operation), 
                                         chain + [subgroup]):
                    return True
    
    return False

def is_abelian(group):
    """群がアーベル群かどうかを判定"""
    for a in group.elements:
        for b in group.elements:
            if group.operation(a, b) != group.operation(b, a):
                return False
    return True

def find_normal_subgroups(group):
    """全ての正規部分群を見つける"""
    normal_subgroups = []
    lattice = SubgroupLattice(group)
    
    for subgroup in lattice.subgroups:
        try:
            normal_subgroup = NormalSubgroup(group, subgroup.generators)
            normal_subgroups.append(normal_subgroup)
        except ValueError:
            # 正規部分群でない場合はスキップ
            continue
    
    return normal_subgroups
```

#### 4.2 可解列の構築
可解群の可解列を明示的に構築：

```python
def solvable_series(group):
    """可解群の可解列を構築"""
    if not is_solvable(group):
        raise ValueError("群が可解ではありません")
    
    series = [group]
    current_group = group
    
    while current_group.order > 1:
        # 最大の真の正規部分群を見つける
        max_normal_subgroup = None
        max_order = 0
        
        for subgroup in find_normal_subgroups(current_group):
            if (subgroup.order > max_order and 
                subgroup.order < current_group.order):
                quotient = subgroup.quotient_group()
                if is_abelian(quotient):
                    max_normal_subgroup = subgroup
                    max_order = subgroup.order
        
        if max_normal_subgroup is None:
            break
        
        series.append(max_normal_subgroup)
        current_group = Group(max_normal_subgroup.elements, current_group.operation)
    
    return series

def composition_series(group):
    """組成列を構築"""
    series = [group]
    current_group = group
    
    while current_group.order > 1:
        # 最大の真の正規部分群を見つける
        max_normal_subgroup = None
        max_order = 0
        
        for subgroup in find_normal_subgroups(current_group):
            if (subgroup.order > max_order and 
                subgroup.order < current_group.order):
                max_normal_subgroup = subgroup
                max_order = subgroup.order
        
        if max_normal_subgroup is None:
            break
        
        series.append(max_normal_subgroup)
        current_group = Group(max_normal_subgroup.elements, current_group.operation)
    
    return series
```

#### 4.3 特殊な可解群の実装
重要な可解群の具体的な実装：

```python
def dihedral_group(n):
    """二面体群 D_n を構築"""
    elements = []
    
    # 回転元素 r^i (i = 0, 1, ..., n-1)
    for i in range(n):
        elements.append(('r', i))
    
    # 反射元素 sr^i (i = 0, 1, ..., n-1)
    for i in range(n):
        elements.append(('s', i))
    
    def operation(a, b):
        if a[0] == 'r' and b[0] == 'r':
            # r^i * r^j = r^(i+j)
            return ('r', (a[1] + b[1]) % n)
        elif a[0] == 'r' and b[0] == 's':
            # r^i * sr^j = sr^(j-i)
            return ('s', (b[1] - a[1]) % n)
        elif a[0] == 's' and b[0] == 'r':
            # sr^i * r^j = sr^(i+j)
            return ('s', (a[1] + b[1]) % n)
        else:
            # sr^i * sr^j = r^(j-i)
            return ('r', (b[1] - a[1]) % n)
    
    return Group(elements, operation)

def frobenius_group(p, q):
    """フロベニウス群を構築（p, qは適切な素数）"""
    # 実装は複雑なので、概念的な構造のみ示す
    # 実際の実装では、有限体の理論が必要
    pass
```

## 理論的な深掘り

### 群論における重要な定理

#### ラグランジュの定理
有限群Gの部分群Hについて、|H| は |G| を割り切ります。

**証明の概要**:
1. 左剰余類による分割
2. 各剰余類の大きさは |H| に等しい
3. 従って |G| = [G:H] × |H|

#### シローの定理
素数pと正整数kについて、|G| = p^k × m（pとmは互いに素）とする。

1. 位数p^kの部分群（シロー p-部分群）が存在する
2. 全てのシロー p-部分群は共役である
3. シロー p-部分群の個数は 1 + kp の形で、mを割り切る

### 対称群の構造

#### S₅の特殊性
S₅は最小の非可解群です：

1. **位数**: |S₅| = 120
2. **正規部分群**: A₅のみ（自明でない）
3. **A₅の単純性**: A₅は位数60の単純群
4. **非可解性**: A₅が非可解なため、S₅も非可解

#### 具体的な構造
```
S₅ (位数120)
├── A₅ (位数60, 正規部分群)
│   └── 単純群（非可解）
└── 他の部分群（非正規）
```

### 5次方程式との関連

#### ガロア群としてのS₅
一般的な5次多項式のガロア群はS₅です：

```python
def general_quintic_galois_group():
    """一般的な5次多項式のガロア群"""
    return SymmetricGroup(5)
```

#### 特殊な可解例
```python
def solvable_quintic_examples():
    """可解な5次多項式の例"""
    examples = [
        "x^5 - 2",      # ガロア群: フロベニウス群F_20
        "x^5 - 5x + 12", # 特殊な可解例
        "x^5 - 1"       # 円分多項式（巡回群）
    ]
    return examples
```

## 実装上の注意点

### 計算効率の最適化

群の演算は計算量が多くなる可能性があります：

1. **置換の表現**: 効率的なデータ構造の選択
2. **演算のキャッシュ**: 頻繁に使用される演算結果の保存
3. **部分群の生成**: 効率的な生成アルゴリズム

### メモリ管理

大きな群を扱う場合のメモリ効率：

1. **遅延評価**: 必要な時にのみ計算
2. **共有構造**: 同じ部分群を複数回作成しない
3. **ガベージコレクション**: 不要になったオブジェクトの適切な削除

### エラーハンドリング

群論の実装では様々なエラーが発生する可能性があります：

1. **群の公理違反**: 入力が群を形成しない場合
2. **計算の失敗**: 逆元や部分群の計算に失敗
3. **メモリ不足**: 大きな群の処理でメモリが不足

## 次のフェーズへの準備

Phase 4で実装した群論は、Phase 5のガロア群計算で直接使用されます：

1. **対称群**: 多項式の根の置換群
2. **可解性判定**: 群が可解かどうかの判定
3. **部分群構造**: ガロア対応における中間体との関係

## 学習のポイント

### 抽象代数の具体化

群論の実装により、抽象的な概念を具体的に理解できます：

1. **群の構造**: 乗積表や生成元による表現
2. **部分群の関係**: 包含関係や正規性
3. **可解性**: 具体的な可解列の構築

### アルゴリズムの理解

群論アルゴリズムの実装により、数学的構造を計算する方法を学べます：

1. **生成**: 生成元から群を構築
2. **探索**: 部分群や正規部分群の発見
3. **判定**: 可解性やアーベル性の判定

## 成果物の活用

Phase 4で作成される成果物：

1. **`group.py`**: 群の基本クラス
2. **`symmetric_group.py`**: 対称群の実装
3. **`subgroup.py`**: 部分群の管理
4. **`solvable_group.py`**: 可解群の判定
5. **`test_groups.py`**: 群論の動作テスト

このフェーズを完了することで、ガロア理論に必要な群論の基盤が整い、次のフェーズでガロア群の具体的な計算に進む準備が整います。特に、可解性の判定アルゴリズムは、5次方程式の可解性判定に直接応用されます。 