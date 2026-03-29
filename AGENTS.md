# fuga-memory: Agent Guidelines

このドキュメントは、LLMエージェント（Claude Code, Gemini CLI, Copilot CLI等）に向けた共通の開発指針です。

## 技術スタック

- **言語**: Python 3.13（uv管理）
- **MCP**: fastmcp
- **DB**: SQLite + FTS5（trigram）+ sqlite-vec
- **埋め込み**: sentence-transformers + cl-nagoya/ruri-v3-310m（ONNXバックエンド）
- **並行処理**: ThreadPoolExecutor（推論・DB操作）+ asyncio（MCPリクエスト受付）
- **検索統合**: RRF（k=60）+ 時間減衰（半減期30日）

## 開発コマンド

```bash
# セットアップ
uv sync

# テスト実行（カバレッジ付き）
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# 型チェック
uv run mypy src/

# MCPサーバー起動
uv run fuga-memory serve

# ファイルから記憶を保存
uv run fuga-memory save --file <path> --session-id <id>

# CLI検索
uv run fuga-memory search <query>

# 記憶の削除
uv run fuga-memory delete <id>
```

## ディレクトリ構造

```
src/fuga_memory/
├── cli.py          # CLIエントリーポイント（click）
├── server.py       # MCPサーバー定義（fastmcp）
├── config.py       # 設定（DBパス、モデル名、スレッド数等）
├── warnings.py     # ライブラリ警告抑制ユーティリティ
├── db/
│   ├── connection.py   # SQLite接続管理（WAL、sqlite-vec拡張）
│   ├── schema.py       # DDL（memoriesテーブル、FTS5、vec仮想テーブル）
│   └── repository.py   # CRUD操作
├── embedding/
│   ├── loader.py       # バックグラウンドモデルロード
│   ├── encoder.py      # テキスト→768次元ベクトル
│   └── onnx_cache.py   # ONNXモデルのローカルキャッシュ管理
└── search/
    ├── fts.py      # FTS5検索
    ├── vector.py   # ベクトル検索
    ├── fusion.py   # RRF統合
    └── decay.py    # 時間減衰スコア計算

tests/
├── unit/           # ユニットテスト（in-memory DB、モックモデル）
├── integration/    # 統合テスト（パイプライン全体）
└── e2e/            # E2Eテスト（MCPプロトコルレベル）
```

## 開発ルール

- **TDD**: テストを先に書く（テストと実装は完全分離）
- **カバレッジ**: 80%以上必須
- **コミット**: Conventional Commits（`feat:`, `fix:`, `refactor:`, `docs:`, `test:`）
- **ブランチ**: GitHub Flow（feature/* → main PR）
- **PRマージ**: CIグリーン確認後、必ずユーザーの明示的な承認を得てからマージする（自動マージ禁止）
- **ファイルサイズ**: 200-400行目安、最大800行
- **型ヒント**: 全関数に必須

## 設計メモ

- **埋め込み次元**: 768（ruri-v3-310m）
- **プレフィックス規則**: 保存時=`検索文書: `、クエリ時=`検索クエリ: `
- **WALモード**: 並行読み取り対応、書き込みはシリアライズ
- **時間減衰**: `score * 0.5^(age_days/30)`
- **RRF**: `score = sum(1/(k+rank_i)) * decay`（k=60）
