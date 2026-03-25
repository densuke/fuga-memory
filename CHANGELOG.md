# Changelog

fuga-memory の更新履歴です。[Semantic Versioning](https://semver.org/lang/ja/) に準拠しています。

---

## [Unreleased]

---

## 2026-03-25

### feat: CLI の timestamp 表示をローカルタイムに変換 (#8)

DB は UTC で格納しているものの、CLI 表示時に変換していなかったため
UTC と JST の混同が起きやすい状態でした。

- `_to_localtime()` ヘルパーを `cli.py` に追加
  - UTC ISO 8601 文字列（`...Z`）をシステムのローカルタイムに変換
  - 標準ライブラリのみ使用（外部依存なし）
  - 出力例: `2026-03-25T00:12:33Z` → `2026-03-25 09:12:33 JST`
- `search` コマンドの `created_at` 表示に適用
- ユニットテスト 3 件を追加

### security: セキュリティ脆弱性の修正 (#7)

セキュリティレビューで検出された脆弱性を修正。

**HIGH**
- FTS5 クエリのサニタイズ処理を追加（`search/fts.py`）
  - 特殊文字（`"()*.:{}`）と大文字演算子（`AND`/`OR`/`NOT`/`NEAR`）を除去
  - `OperationalError` を FTS5 構文エラーに限定してフォールバック
- `top_k` / `limit` の上限バリデーションを追加（`MAX_TOP_K=100`, `MAX_LIMIT=200`）

**MEDIUM**
- `save_memory` に `content` の空文字列・100,000 文字超過チェックを追加
- `search_memory` に `query` の 4,096 文字超過チェックを追加
- `embedding_dim` の DDL f-string 展開前に正の整数かを検証
- `model_name` のパストラバーサル対策（正規表現 + `..` 禁止、`re.ASCII`）
- `db_path` に `.resolve()` を追加してパストラバーサルを防止
- セキュリティイベントのログ記録を追加（`logging` モジュール）
- stdin/ファイル読み込みを 1MB 上限・バイナリ実測チェックに変更
- `db/repository.py` の `search_fts` を安全版に委譲

**LOW**
- CI GitHub Actions を SHA ハッシュでピン留め（サプライチェーン攻撃対策）

### fix: sentencepiece を依存関係に追加

`sentence-transformers` 実行時に `sentencepiece` が必要なケースがあったため明示的に追加。

### feat: 設定ファイル対応と環境変数の全フィールド補完 (#6)

- TOML 設定ファイルの読み込みに対応（優先度: デフォルト < 設定ファイル < 環境変数）
- 設定ファイル探索パスを OS 別に定義
  - macOS: `~/Library/Application Support/fuga-memory/config.toml`
  - Linux: `$XDG_CONFIG_HOME/fuga-memory/config.toml`
  - 共通: `~/.fuga-memory.toml`
- 環境変数に全フィールドを対応（`FUGA_MEMORY_*` プレフィックス）
- `config.toml.example` / `.env.example` を追加
- テスト 53 件追加、カバレッジ 97% 以上

---

## 2026-03-24

### feat: Phase 4 — MCP サーバー + CLI の実装 (#5)

- `server.py`: fastmcp を使った MCP サーバー定義
  - `save_memory` / `search_memory` / `list_sessions` の 3 ツールを公開
  - グローバル依存（DB 接続・エンコーダ・設定）の遅延初期化
  - テストでインメモリ DB・モックエンコーダを注入可能な設計
- `cli.py`: Click を使った CLI エントリーポイント
  - `serve` / `search` / `save` の 3 コマンドを実装
  - `save` は引数・`--stdin`・`--file` の 3 入力方式に対応
- 統合テスト・E2E テスト追加、全テスト通過

### fix: Phase 3 Gemini レビュー指摘を対応（検索層） (#4)

- `search_fts`: `rowid` を明示した JOIN クエリに修正
- `reciprocal_rank_fusion`: `decay_score` を積算する方向に修正
- `search_vector`: `distance` を降順でなく昇順（近い順）に修正
- テストの期待値・アサーションを仕様に合わせて修正

### feat: Phase 3 — 検索層の実装 (#3)

- `search/decay.py`: 時間減衰スコア計算（半減期ベースの指数関数）
- `search/fts.py`: FTS5 全文検索モジュール関数
- `search/vector.py`: sqlite-vec KNN ベクトル検索モジュール関数
- `search/fusion.py`: RRF（Reciprocal Rank Fusion）+ 時間減衰によるスコア統合
- TDD: 41 件のテスト（RED→GREEN）、全 116 件パス、カバレッジ 97.56%

### feat: Phase 2 — 埋め込み層の実装 (#2)

- `exceptions.py`: `FugaMemoryError` / `ModelLoadError` を追加
- `embedding/encoder.py`: `Encoder` プロトコルと `RuriEncoder` を実装
  - sentence-transformers ONNX バックエンドで cl-nagoya/ruri-v3-310m を使用
  - 保存時プレフィックス「検索文書: 」を自動付与
  - 768 次元 float リストを返す
- `embedding/loader.py`: `ModelLoader` を実装
  - `ThreadPoolExecutor` によるバックグラウンドロード
  - スレッドセーフなダブルチェックロッキング
  - ロード失敗時は `ModelLoadError` を送出
- 72 テスト全パス、カバレッジ 96.38%

### feat: Phase 1 — DB 層の実装 (#1)

- `config.py`: 設定クラス（デフォルト値、環境変数オーバーライド対応）
- `db/connection.py`: WAL モード + sqlite-vec 拡張ロード
- `db/schema.py`: `memories` テーブル、FTS5（trigram トークナイザ）、vec 仮想テーブルの DDL
- `db/repository.py`: `save` / `search_fts` / `search_vector` / `list_sessions` の CRUD 操作
- ユニットテスト 35 件、カバレッジ 98.78%

### chore: initial project setup

- `pyproject.toml` でプロジェクト構成を定義
- 依存関係: fastmcp, sentence-transformers, sqlite-vec, click
- 開発依存: pytest, ruff, mypy, coverage
- pre-commit フックに ruff format / lint + mypy チェックを設定
