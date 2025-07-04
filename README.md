pushテスト

# 5次方程式解の存在判定 - ガロア理論による実装

このプロジェクトは、ガロア理論を用いて5次方程式の解の存在を判定するPythonライブラリです。テスト駆動開発（TDD）とオブジェクト指向設計を採用し、数学的に厳密で保守可能な実装を目指しています。

## 🎯 プロジェクトの目的

- 5次方程式の根号による解の存在を判定
- ガロア理論の実践的な実装
- 抽象代数学の概念をPythonで表現
- 教育的価値の高いコードベースの構築

## 📁 プロジェクト構造

```
galois_theory/
├── src/galois_theory/          # メインパッケージ
│   ├── __init__.py
│   ├── algebraic_structures.py # 群、環、体の実装
│   ├── polynomials.py          # 多項式環
│   ├── field_extensions.py     # 体拡大
│   ├── galois_groups.py        # ガロア群
│   └── solvability.py          # 可解性判定
├── tests/                      # テストスイート
│   ├── __init__.py
│   ├── test_algebraic_structures.py
│   ├── test_polynomials.py
│   ├── test_field_extensions.py
│   ├── test_galois_groups.py
│   ├── test_solvability.py
│   └── test_integration.py
├── doc/                        # ドキュメント
│   ├── galois_theory_implementation_guide.md
│   ├── implementation_methodology.md
│   └── phase*.md
├── requirements.txt            # 依存関係
├── pyproject.toml             # プロジェクト設定
└── README.md                  # このファイル
```

## 🚀 セットアップ

### 1. 環境準備

```bash
# Python 3.9以上が必要
python --version

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール

```bash
# 基本依存関係
pip install -r requirements.txt

# または開発依存関係も含めて
pip install -e ".[dev]"
```

## 🧪 テスト実行

### 基本的なテスト実行

```bash
# 全てのテストを実行
pytest

# 詳細な出力で実行
pytest -v

# カバレッジ付きで実行
pytest --cov=src/galois_theory --cov-report=html
```

### 特定のテストカテゴリ

```bash
# 単体テストのみ
pytest -m unit

# 統合テストのみ
pytest -m integration

# 重いテストを除外
pytest -m "not slow"
```

### テスト結果の確認

```bash
# カバレッジレポートをブラウザで確認
open htmlcov/index.html  # Mac
# または
start htmlcov/index.html # Windows
```

## 🔧 開発ツール

### コード品質チェック

```bash
# 型チェック
mypy src/galois_theory

# コードフォーマット
black src/ tests/

# インポート整理
isort src/ tests/

# リント
flake8 src/ tests/
```

### 全体的な品質チェック

```bash
# 一括実行スクリプト（作成予定）
./scripts/check_quality.sh
```

## 📚 実装フェーズ

このプロジェクトは段階的に実装されます：

1. **Phase 1**: 抽象代数学基礎 - 群、環、体
2. **Phase 2**: 多項式環 - 多項式の操作と性質
3. **Phase 3**: 体拡大 - 拡大体の理論
4. **Phase 4**: 群論 - 群の性質と同型
5. **Phase 5**: ガロア群 - ガロア群の計算
6. **Phase 6**: 可解性判定 - 最終的な判定アルゴリズム

詳細は `doc/` ディレクトリの各フェーズドキュメントを参照してください。

## 🎓 理論的背景

このプロジェクトは以下の数学的概念に基づいています：

- **抽象代数学**: 群、環、体の理論
- **ガロア理論**: 体拡大とガロア群
- **群論**: 可解群と組成列
- **多項式理論**: 既約多項式と分解体

## 🤝 貢献

1. Fork このリポジトリ
2. Feature ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Request を作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 👤 作者

- **asato99w** - *初期作成* - [GitHub](https://github.com/asato99w)

## 🙏 謝辞

- ガロア理論の教育的リソースを提供してくださった数学コミュニティ
- オープンソース数学ライブラリの開発者の皆様