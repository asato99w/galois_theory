"""
群論（Group Theory）のテストスイート

このモジュールは、ガロア理論で使用される群論の実装をテストします。
主要なテスト対象:
- Group: 群の抽象基底クラス
- FiniteGroup: 有限群
- CyclicGroup: 巡回群
- SymmetricGroup: 対称群
- DihedralGroup: 二面体群
- GaloisGroup: ガロア群（体拡大との関連）
- GroupElement: 群要素
- GroupAction: 群の作用
- GroupHomomorphism: 群準同型写像
- Subgroup: 部分群
"""

import pytest
from fractions import Fraction
from typing import List, Set, Dict, Optional

from galois_theory.field import RationalField, FiniteField
from galois_theory.polynomials import PolynomialRing
from galois_theory.field_extensions import SimpleExtension
from galois_theory.group_theory import (
    Group, GroupElement, FiniteGroup, CyclicGroup, SymmetricGroup,
    DihedralGroup, GaloisGroup, GroupAction, GroupHomomorphism,
    Subgroup, GroupException, Permutation, GroupIsomorphism
)


class TestGroupBasics:
    """群の基本概念のテスト"""

    def test_group_creation(self) -> None:
        """群の作成テスト"""
        # 位数4の巡回群 Z/4Z
        c4 = CyclicGroup(4)
        assert c4.order() == 4
        assert c4.is_abelian()
        assert c4.is_cyclic()

    def test_group_element_operations(self) -> None:
        """群要素の演算テスト"""
        c3 = CyclicGroup(3)
        
        # 生成元
        g = c3.generator()
        e = c3.identity()
        
        # 演算の確認
        assert g * g != g
        assert g * e == g
        assert e * g == g
        assert g * g * g == e  # 位数3なので g³ = e
        
        # 逆元
        assert g.inverse() * g == e
        assert g * g.inverse() == e

    def test_group_order_and_exponent(self) -> None:
        """群の位数と指数のテスト"""
        c6 = CyclicGroup(6)
        assert c6.order() == 6
        assert c6.exponent() == 6  # 巡回群の指数は位数と同じ
        
        # 群要素の位数
        g = c6.generator()
        assert g.order() == 6
        assert (g ** 2).order() == 3
        assert (g ** 3).order() == 2

    def test_subgroup_generation(self) -> None:
        """部分群の生成テスト"""
        c12 = CyclicGroup(12)
        g = c12.generator()
        
        # <g²> は位数6の部分群
        g2 = g ** 2
        subgroup = c12.subgroup_generated_by([g2])
        assert subgroup.order() == 6
        assert subgroup.is_cyclic()

    def test_group_center(self) -> None:
        """群の中心のテスト"""
        c4 = CyclicGroup(4)
        center = c4.center()
        assert center.order() == 4  # アーベル群なので中心は群全体
        
        # 非アーベル群での中心（後で実装）
        # d3 = DihedralGroup(3)
        # center_d3 = d3.center()
        # assert center_d3.order() == 1  # 二面体群D₃の中心は自明

    def test_group_conjugacy_classes(self) -> None:
        """共役類のテスト"""
        c4 = CyclicGroup(4)
        conjugacy_classes = c4.conjugacy_classes()
        assert len(conjugacy_classes) == 4  # アーベル群なので各元が1つの共役類


class TestSymmetricGroup:
    """対称群のテスト"""

    def test_symmetric_group_creation(self) -> None:
        """対称群の作成テスト"""
        s3 = SymmetricGroup(3)
        assert s3.order() == 6  # S₃の位数は3! = 6
        assert not s3.is_abelian()  # S₃は非アーベル群
        assert not s3.is_cyclic()

    def test_permutation_operations(self) -> None:
        """置換の演算テスト"""
        s3 = SymmetricGroup(3)
        
        # 置換を作成 (1 2 3) → (2 3 1)
        cycle_123 = s3.element_from_cycle([1, 2, 3])
        assert cycle_123.order() == 3
        
        # 置換を作成 (1 2)
        transposition_12 = s3.element_from_transposition(1, 2)
        assert transposition_12.order() == 2
        
        # 合成
        composition = cycle_123 * transposition_12
        assert composition.order() == 2

    def test_permutation_cycle_decomposition(self) -> None:
        """置換の巡回分解テスト"""
        s4 = SymmetricGroup(4)
        
        # (1 2 3)(4) の巡回分解
        perm = s4.element_from_cycles([[1, 2, 3], [4]])
        cycles = perm.cycle_decomposition()
        assert len(cycles) == 2
        assert [1, 2, 3] in cycles or [2, 3, 1] in cycles or [3, 1, 2] in cycles
        
    def test_permutation_parity(self) -> None:
        """置換の偶奇性テスト"""
        s3 = SymmetricGroup(3)
        
        # 3-巡回は偶置換
        cycle_123 = s3.element_from_cycle([1, 2, 3])
        assert cycle_123.is_even()
        assert not cycle_123.is_odd()
        
        # 互換は奇置換
        transposition = s3.element_from_transposition(1, 2)
        assert transposition.is_odd()
        assert not transposition.is_even()

    def test_alternating_group(self) -> None:
        """交代群のテスト"""
        s4 = SymmetricGroup(4)
        a4 = s4.alternating_subgroup()
        
        assert a4.order() == 12  # A₄の位数は4!/2 = 12
        assert not a4.is_abelian()
        
        # すべての要素が偶置換
        for element in a4.elements():
            assert element.is_even()


class TestDihedralGroup:
    """二面体群のテスト"""

    def test_dihedral_group_creation(self) -> None:
        """二面体群の作成テスト"""
        d3 = DihedralGroup(3)  # 正三角形の対称群
        assert d3.order() == 6
        assert not d3.is_abelian()
        assert not d3.is_cyclic()

    def test_dihedral_group_generators(self) -> None:
        """二面体群の生成元テスト"""
        d4 = DihedralGroup(4)  # 正方形の対称群
        
        rotation = d4.rotation_generator()
        reflection = d4.reflection_generator()
        
        assert rotation.order() == 4
        assert reflection.order() == 2
        
        # 関係式: r⁴ = e, s² = e, srs = r⁻¹
        r, s = rotation, reflection
        e = d4.identity()
        
        assert r ** 4 == e
        assert s ** 2 == e
        assert s * r * s == r.inverse()

    def test_dihedral_subgroups(self) -> None:
        """二面体群の部分群テスト"""
        d6 = DihedralGroup(6)
        
        # 回転による部分群は巡回群
        rotation_subgroup = d6.rotation_subgroup()
        assert rotation_subgroup.order() == 6
        assert rotation_subgroup.is_cyclic()


class TestGaloisGroup:
    """ガロア群のテスト"""

    def test_galois_group_quadratic_extension(self) -> None:
        """2次拡大のガロア群テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2)/Q のガロア群
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])  # x² - 2
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        
        galois_group = GaloisGroup.from_extension(extension, base_field)
        
        assert galois_group.order() == 2
        assert galois_group.is_abelian()
        assert galois_group.is_cyclic()
        
        # ガロア群の要素
        identity = galois_group.identity()
        conjugation = galois_group.conjugation_automorphism()
        
        assert identity.order() == 1
        assert conjugation.order() == 2
        assert conjugation * conjugation == identity

    def test_galois_group_cubic_extension(self) -> None:
        """3次拡大のガロア群テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # x³ - 2 の分解体のガロア群
        polynomial = poly_ring.from_coefficients([-2, 0, 0, 1])  # x³ - 2
        
        # 完全な分解体のガロア群は S₃ と同型
        galois_group = GaloisGroup.from_splitting_field(polynomial.polynomial, base_field)
        
        assert galois_group.order() == 6
        assert not galois_group.is_abelian()
        assert galois_group.is_isomorphic_to(SymmetricGroup(3))

    def test_galois_group_automorphisms(self) -> None:
        """ガロア群の自己同型写像テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2, √3)/Q のガロア群
        # これは Klein 4-群と同型
        sqrt2_poly = poly_ring.from_coefficients([-2, 0, 1])
        q_sqrt2 = SimpleExtension(base_field, sqrt2_poly, "sqrt2")
        
        poly_ring_ext = PolynomialRing(q_sqrt2, "y")
        sqrt3_poly = poly_ring_ext.from_coefficients([-3, 0, 1])
        q_sqrt2_sqrt3 = SimpleExtension(q_sqrt2, sqrt3_poly, "sqrt3")
        
        galois_group = GaloisGroup.from_extension(q_sqrt2_sqrt3, base_field)
        
        assert galois_group.order() == 4
        assert galois_group.is_abelian()
        assert not galois_group.is_cyclic()  # Klein 4-群
        
        # 4つの自己同型写像
        automorphisms = galois_group.automorphisms()
        assert len(automorphisms) == 4

    def test_fundamental_theorem_galois_theory(self) -> None:
        """ガロア理論の基本定理テスト"""
        base_field = RationalField()
        poly_ring = PolynomialRing(base_field, "x")
        
        # Q(√2)/Q
        minimal_poly = poly_ring.from_coefficients([-2, 0, 1])
        extension = SimpleExtension(base_field, minimal_poly, "sqrt2")
        galois_group = GaloisGroup.from_extension(extension, base_field)
        
        # 拡大次数 = ガロア群の位数
        assert extension.degree() == galois_group.order()
        
        # 部分群と中間体の対応
        subgroups = galois_group.all_subgroups()
        intermediate_fields = galois_group.intermediate_fields()
        
        assert len(subgroups) == len(intermediate_fields)


class TestGroupAction:
    """群の作用のテスト"""

    def test_group_action_on_set(self) -> None:
        """集合への群作用テスト"""
        s3 = SymmetricGroup(3)
        set_elements = {1, 2, 3}
        
        # S₃ の {1,2,3} への自然な作用
        action = GroupAction(s3, set_elements)
        
        # 作用の公理
        identity = s3.identity()
        g = s3.element_from_transposition(1, 2)
        h = s3.element_from_cycle([1, 2, 3])
        
        # e・x = x
        for x in set_elements:
            assert action.act(identity, x) == x
        
        # (gh)・x = g・(h・x)
        for x in set_elements:
            lhs = action.act(g * h, x)
            rhs = action.act(g, action.act(h, x))
            assert lhs == rhs

    def test_orbit_and_stabilizer(self) -> None:
        """軌道と固定化群のテスト"""
        s4 = SymmetricGroup(4)
        set_elements = {1, 2, 3, 4}
        action = GroupAction(s4, set_elements)
        
        # 1の軌道は {1,2,3,4} 全体
        orbit_1 = action.orbit(1)
        assert orbit_1 == {1, 2, 3, 4}
        
        # 1の固定化群
        stabilizer_1 = action.stabilizer(1)
        assert stabilizer_1.order() == 6  # S₃と同型
        
        # 軌道・固定化群定理: |G| = |Orbit| × |Stabilizer|
        assert s4.order() == len(orbit_1) * stabilizer_1.order()

    def test_burnside_lemma(self) -> None:
        """バーンサイドの補題テスト"""
        # D₃ の正三角形の頂点への作用
        d3 = DihedralGroup(3)
        vertices = {1, 2, 3}
        action = GroupAction(d3, vertices)
        
        # 軌道の数を計算
        num_orbits = action.count_orbits()
        assert num_orbits == 1  # 推移的作用


class TestGroupHomomorphism:
    """群準同型写像のテスト"""

    def test_group_homomorphism_creation(self) -> None:
        """群準同型写像の作成テスト"""
        c6 = CyclicGroup(6)
        c2 = CyclicGroup(2)
        
        # Z/6Z → Z/2Z の準同型 (x ↦ x mod 2)
        def homomorphism_function(x):
            return c2.element(x.value % 2)
        
        phi = GroupHomomorphism(c6, c2, homomorphism_function)
        
        # 準同型の性質
        g1 = c6.element(2)
        g2 = c6.element(3)
        
        assert phi(g1 * g2) == phi(g1) * phi(g2)

    def test_kernel_and_image(self) -> None:
        """核と像のテスト"""
        c8 = CyclicGroup(8)
        c4 = CyclicGroup(4)
        
        # Z/8Z → Z/4Z の自然な準同型
        def natural_map(x):
            return c4.element(x.value % 4)
        
        phi = GroupHomomorphism(c8, c4, natural_map)
        
        kernel = phi.kernel()
        image = phi.image()
        
        assert kernel.order() == 2
        assert image.order() == 4
        
        # 準同型定理: |G| = |ker(φ)| × |im(φ)|
        assert c8.order() == kernel.order() * image.order()

    def test_group_isomorphism(self) -> None:
        """群同型写像のテスト"""
        c4 = CyclicGroup(4)
        z4_additive = CyclicGroup(4)  # 加法群として
        
        # 位数4の巡回群は同型
        iso = GroupIsomorphism.find_isomorphism(c4, z4_additive)
        assert iso is not None
        assert iso.is_bijective()
        assert iso.preserves_operation()


class TestFiniteGroupProperties:
    """有限群の性質のテスト"""

    def test_lagrange_theorem(self) -> None:
        """ラグランジュの定理テスト"""
        s3 = SymmetricGroup(3)  # S₄の代わりにS₃を使用
        
        # すべての部分群について、位数が群の位数を割る
        for subgroup in s3.all_subgroups():
            assert s3.order() % subgroup.order() == 0

    def test_cauchy_theorem(self) -> None:
        """コーシーの定理テスト（素数位数の元の存在）"""
        # 位数12の群には位数2と位数3の元が存在
        c12 = CyclicGroup(12)
        
        # 位数2の元
        has_order_2 = any(g.order() == 2 for g in c12.elements())
        assert has_order_2
        
        # 位数3の元
        has_order_3 = any(g.order() == 3 for g in c12.elements())
        assert has_order_3

    def test_sylow_theorems(self) -> None:
        """シローの定理テスト（基本的な場合）"""
        # S₃ の場合（位数6 = 2 × 3）
        s3 = SymmetricGroup(3)
        
        # 2-シロー部分群（位数2）
        sylow_2_subgroups = s3.sylow_subgroups(2)
        assert len(sylow_2_subgroups) == 3  # S₃ には3つの位数2の部分群
        
        # 3-シロー部分群（位数3）
        sylow_3_subgroups = s3.sylow_subgroups(3)
        assert len(sylow_3_subgroups) == 1  # S₃ には1つの位数3の部分群

    def test_group_classification_small_orders(self) -> None:
        """小さい位数の群の分類テスト"""
        # 位数4の群は2種類: Z/4Z と Klein 4-群
        groups_order_4 = FiniteGroup.all_groups_of_order(4)
        assert len(groups_order_4) == 2
        
        # 一方は巡回群、もう一方は非巡回群
        cyclic_count = sum(1 for g in groups_order_4 if g.is_cyclic())
        assert cyclic_count == 1


class TestGroupException:
    """群論の例外処理テスト"""

    def test_invalid_group_operation(self) -> None:
        """無効な群演算のエラーテスト"""
        c3 = CyclicGroup(3)
        c4 = CyclicGroup(4)
        
        g1 = c3.generator()
        g2 = c4.generator()
        
        # 異なる群の要素同士の演算はエラー
        with pytest.raises(GroupException, match="異なる群の要素同士の演算はできません"):
            g1 * g2

    def test_invalid_permutation_error(self) -> None:
        """無効な置換のエラーテスト"""
        s3 = SymmetricGroup(3)
        
        # 範囲外の要素を含む置換
        with pytest.raises(GroupException, match="無効な置換"):
            s3.element_from_cycle([1, 2, 4])  # 4は範囲外

    def test_non_homomorphism_error(self) -> None:
        """準同型でない写像のエラーテスト"""
        c4 = CyclicGroup(4)
        c2 = CyclicGroup(2)
        
        # 準同型でない関数
        def bad_function(x):
            return c2.element((x.value + 1) % 2)  # これは準同型でない
        
        with pytest.raises(GroupException, match="準同型写像ではありません"):
            GroupHomomorphism(c4, c2, bad_function)


class TestAdvancedGroupTheory:
    """高度な群論のテスト"""

    def test_semidirect_product(self) -> None:
        """半直積のテスト"""
        # D₄ ≅ Z/4Z ⋊ Z/2Z (実装後にテスト)
        pass

    def test_group_extensions(self) -> None:
        """群の拡大のテスト"""
        # 群の拡大理論（実装後にテスト）
        pass

    def test_representation_theory_basics(self) -> None:
        """表現論の基礎テスト"""
        # 群の表現（実装後にテスト）
        pass

    def test_character_theory(self) -> None:
        """指標論のテスト"""
        # 群の指標（実装後にテスト）
        pass 