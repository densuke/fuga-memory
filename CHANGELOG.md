# Changelog

fuga-memory の更新履歴です。[Semantic Versioning](https://semver.org/lang/ja/) に準拠したバージョン管理へ今後移行する予定です。

---

## [Unreleased]

### feat: 記憶の削除機能（delete command/tool）を追加 (#14)

不要な記憶や重複データを整理・枝刈りするための機能を追加しました。AI レビュアー（Gemini/Copilot）の指摘に基づき、パフォーマンスと原子性を最適化しています。

- **リポジトリ層**: `MemoryRepository.delete_memory()` を追加。
  - エンコーダ不要の `@staticmethod` として実装。
  - `SELECT` を排し `DELETE` 直後の `rowcount` で判定することで原子性と効率を向上。
  - `memories`, `memories_fts`, `memories_vec` の 3 テーブルから一貫して削除。
- **サーバー層 (MCP)**: `delete_memory` ツールを追加。
  - `memory_id < 1` のバリデーションを追加。
  - 削除時に重いモデル（エンコーダ）をロードしないように最適化。
- **CLI層**: `fuga-memory delete <ID>` コマンドを追加。
  - `memory_id` の入力範囲を 1 以上に制限（`click.IntRange`）。
  - 削除失敗（見つからない）やエラー時に非ゼロ（exit code 1）で終了し、stderr にエラーを出力するよう改善。
- **テスト**: 全レイヤーでユニットテストを追加し、マルチプラットフォーム（Windows/Linux/macOS）での動作を検証済み。

---

## [0.4.2] - 2026-03-28

### docs: ドキュメント更新（v0.4.0/v0.4.1 の変更を反映）

- `CLAUDE.md`: ディレクトリ構造に `warnings.py`・`embedding/onnx_cache.py` を追加
- `CLAUDE.md`: Stop フックのコマンドを README に合わせて統一
  - `${sessionId}` → `${CLAUDE_SESSION_ID:-unknown}`
  - timeout 30 → 60
  - `--source claude_code` を追加
- `README.md`: 環境変数表に v0.4.0 追加分を補完
  - `FUGA_MEMORY_DAEMON_PORT`、`FUGA_MEMORY_DAEMON_IDLE_TIMEOUT`
  - `FUGA_MEMORY_ONNX_CACHE_DIR`、`FUGA_MEMORY_DEBUG`
- `README.md`: CLI リファレンスに `--debug` フラグの使用例を追加
- `README.md`: データの場所テーブルに ONNX キャッシュパスを追加
- `README.md`: 初回起動の説明を更新（ONNX キャッシュで2回目以降が高速になった旨）

---

## [0.4.1] - 2026-03-28

### fix: FTS5 検索クエリのハイフン処理

- `search/fts.py`: `_sanitize_fts_query()` でハイフン（`-`）を除去するよう修正
  - `fuga-memory` のような語中ハイフンが FTS5 の否定演算子として解釈され、
    `no such column: memory` エラーが発生していた問題を修正
- `tests/unit/test_search_fts.py`: ハイフンを含むクエリのテストを 2 件追加

---

## [0.4.0] - 2026-03-28

### feat: 起動ログ改善・ONNXモデルのローカルキャッシュを実装

初回起動時に大量のライブラリ警告が出力される問題を解決するため、
警告抑制機構と ONNX モデルのローカルキャッシュ保存を実装しました。

#### Phase 1: 警告抑制・進捗メッセージ

- `src/fuga_memory/warnings.py` を新規追加
  - `suppress_warnings()`: PyTorch/ONNXRuntime/HuggingFace 系ライブラリの警告を抑制
  - `is_debug_mode()`: `FUGA_MEMORY_DEBUG` 環境変数でデバッグモード判定
- `cli.py` に `--debug` フラグを追加（デバッグモード時は警告を抑制しない）
- `embedding/loader.py` に初回モデルロード時の進捗メッセージを追加
  - `モデルを初期化中... (初回のみ、数十秒かかります)` を stderr に表示
- `Config` に `debug` フィールドを追加

#### Phase 2: ONNX モデルキャッシュ

- `src/fuga_memory/embedding/onnx_cache.py` を新規追加
  - `get_onnx_cache_dir()`: モデル名からキャッシュパスを生成
  - `is_cached()`: キャッシュ済み ONNX モデルの存在確認
  - `export_and_cache()`: SentenceTransformer を ONNX 形式でローカルに保存
  - `load_cached_model()`: キャッシュ済み ONNX モデルを高速ロード
- `embedding/encoder.py` の `RuriEncoder` に `cache_dir` パラメータを追加
  - キャッシュあり → 高速ローカルロード
  - キャッシュなし → エクスポート・保存してからロード
  - エクスポート失敗時 → 直接ロードにフォールバック
- `embedding/loader.py` の `ModelLoader` に `cache_dir` パラメータを追加
- `server.py`・`daemon/server.py` が `onnx_cache_dir` 設定を参照するよう更新
- `Config` に `onnx_cache_dir` フィールドを追加
  - デフォルト: `~/.local/share/fuga-memory/onnx_cache`
  - 設定ファイル: `onnx_cache_dir`
  - 環境変数: `FUGA_MEMORY_ONNX_CACHE_DIR`

これにより、初回起動時のみ ONNX エクスポート（数十秒）が発生し、
2回目以降はキャッシュから高速ロードされるようになります。

---

## [0.3.0] - 2026-03-28

### feat: バックグラウンドデーモン経由の非同期 save (#12)

Claude Code の Stop フックで `fuga-memory save --stdin` がブロッキングになる問題を解決するため、
軽量ローカル HTTP デーモンをバックグラウンド自動起動し、即座に 202 を返す仕組みを実装しました。

#### 新機能

- `src/fuga_memory/daemon/` パッケージを新規追加
  - `server.py`: `ThreadingHTTPServer` + watchdog（アイドルタイムアウト 10 分で自動停止）
    - `/save` (POST): 202 即返却、`ThreadPoolExecutor`（max 4 workers）でバックグラウンド処理
    - `/health` (GET): `{"app": "fuga-memory", "pending": <int>}` を返す
    - `/shutdown` (POST): graceful shutdown
    - プロセス内エンコーダキャッシュ（リクエストごとのモデルロードを排除）
    - `Content-Length` 上限 100,000 バイトのバリデーション
  - `client.py`: `ensure_daemon_running()` + `send_save_request()`（202 即返）
  - `_process.py`: クロスプラットフォームでデタッチプロセス起動（Windows/POSIX）
- `Config` に `daemon_port=18520`、`daemon_idle_timeout=600` を追加
- `cli.py` の save コマンドをデーモン経由に変更（失敗時は従来の直接実行にフォールバック）
- CLAUDE.md にデーモン設定例・動作説明・管理コマンドを追記

#### バグ修正

- Windows: `close_fds=True` + DEVNULL リダイレクトの組み合わせが `ValueError` になるバグを修正
- `initialize_schema` / `MemoryRepository` に `config.embedding_dim` を渡すよう修正
- FileHandler 重複追加防止（複数回 DaemonServer 生成時の FD リーク解消）
- `server_close()` を `serve_forever()` 終了後に呼び出し（ソケット即時解放）

#### テスト

- `tests/unit/test_daemon_process.py`、`test_daemon_client.py`、`test_daemon_server.py` を追加
- `tests/integration/test_daemon_integration.py` を追加
- CI マトリクスに Windows を追加（クロスプラットフォーム検証）

---

## [0.2.0] - 2026-03-27

### docs: Gemini CLI 利用ガイドの日本語化

`GEMINI.md` の内容を日本語に翻訳。Gemini CLI ユーザーが設定・利用方法を母国語で確認できるように改善しました。

### docs: セキュリティ監査レポートを追加 (#11)

プロジェクト全体のセキュリティ監査を実施し、結果を `SECURITY_AUDIT.md` にまとめました。

- **SQL インジェクション**: パラメータ化クエリと FTS5 サニタイズにより安全であることを確認
- **パストラバーサル**: モデル名バリデーション等により安全であることを確認
- **リソース制限**: 入力文字数・ファイルサイズの上限設定を確認
- **データ保護**: SQLite 非暗号化の現状と SQLCipher 等の推奨事項を記載
- 監査日のプレースホルダ化（レビュー指摘対応）

### docs: Gemini CLI 向けセットアップガイドを追加 (#10)

Gemini CLI で `fuga-memory` を MCP サーバーとして利用するための専用ドキュメント `GEMINI.md` を追加。

- `~/.gemini/settings.json` への登録方法を明示
- `save_memory` / `search_memory` / `list_sessions` の Gemini CLI での活用例を記載
- セットアップ手順の明確化（レビュー指摘対応: `uv sync` の実行場所を明示）
- `README.md` のセットアップガイド見出しを日本語化

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
  - 特殊文字（`"()*.:{}^`）を除去し、大文字演算子（`AND`/`OR`/`NOT`/`NEAR`）は小文字化して演算子として解釈されないようにサニタイズ
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
