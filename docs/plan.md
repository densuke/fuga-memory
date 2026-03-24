# fuga-memory 実装プラン

## 概要

Claude Code（および他のLLM）向けの長期記憶MCPサーバー。

## アーキテクチャ

```
保存経路：
  Claude Code  → Stop hook → CLI → SQLite
  他LLM        → MCP save_memory ツール → SQLite

検索経路（全LLM共通）：
  LLM → MCP search_memory → FTS5+ベクトル(Ruri v3) → RRF → 結果返却
```

## 技術選定

| 項目 | 選択 | 理由 |
|------|------|------|
| 言語 | Python 3.13 | sentence-transformers との親和性、実装速度 |
| MCP | fastmcp | シンプルなAPI、asyncio対応 |
| DB | SQLite + sqlite-vec | 外部依存なし、単一ファイル |
| 全文検索 | FTS5 + trigram | 形態素解析不要、日本語対応 |
| 埋め込み | ruri-v3-310m (ONNX) | CPU動作、日本語特化、軽量 |
| 並行処理 | ThreadPoolExecutor + asyncio | 推論・DB操作の並列化 |

## 実装フェーズ

### Phase 0: プロジェクト基盤（完了）
- [x] ディレクトリ作成、git init
- [x] pyproject.toml（uv + hatchling）
- [x] CLAUDE.md
- [x] GitHub リポジトリ作成・初回push
- [x] GitHub Actions CI

### Phase 1: DB層
- [ ] config.py（設定クラス）
- [ ] db/connection.py（WAL、sqlite-vec拡張ロード）
- [ ] db/schema.py（DDL定義）
- [ ] db/repository.py（CRUD操作）
- [ ] ユニットテスト（in-memory DB使用）

### Phase 2: 埋め込み層
- [ ] embedding/loader.py（バックグラウンドロード）
- [ ] embedding/encoder.py（テキスト→768次元ベクトル）
- [ ] ユニットテスト（モックモデル使用）

### Phase 3: 検索層
- [ ] search/decay.py（時間減衰スコア計算）
- [ ] search/fts.py（FTS5検索）
- [ ] search/vector.py（ベクトル検索）
- [ ] search/fusion.py（RRF統合）
- [ ] ユニットテスト + 統合テスト

### Phase 4: MCP + CLI
- [ ] server.py（fastmcp、3ツール）
- [ ] cli.py（click、serve/save/search）
- [ ] E2Eテスト

## スキーマ設計

```sql
CREATE TABLE memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL  -- ISO 8601
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='trigram'
);

-- sqlite-vec 仮想テーブル（Phase 1でDDL定義、Phase 2で接続）
CREATE VIRTUAL TABLE memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[768]
);
```

## 検索アルゴリズム

### RRF計算式
```
score(d) = sum(1 / (k + rank_i)) * decay(d.created_at)
k = 60（標準値）
```

### 時間減衰
```
decay(t) = 0.5 ^ (age_days / 30)
age_days = (now - created_at).days
```

## リスク対策

| リスク | 対策 |
|--------|------|
| sqlite-vec Python 3.13互換性 | Phase 0で検証済み（v0.1.7 OK） |
| torch のパッケージサイズ | ONNXバックエンドで回避 |
| Stop hookの環境変数仕様 | フォールバックとしてタイムスタンプID生成 |
| FTS5日本語精度 | trigramトークナイザー採用 |
