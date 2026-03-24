# fuga-memory

Claude Code（および他の LLM）向け長期記憶 MCP サーバー。

会話内容を SQLite に保存し、FTS5（全文検索）+ ベクトル検索（ruri-v3-310m）のハイブリッド検索で関連する記憶を呼び出します。

## 特徴

- **外部依存なし**: 全データを SQLite 単一ファイルに格納
- **ハイブリッド検索**: FTS5 キーワード検索 + ベクトル検索を RRF で統合
- **時間減衰**: 古い記憶のスコアを半減期 30 日で段階的に低下
- **軽量推論**: ONNX バックエンドで ruri-v3-310m を CPU で動作
- **MCP 対応**: Claude Code / Gemini / Copilot など複数 LLM で共有可能
- **自動保存**: Claude Code の Stop フックでセッション終了時に自動保存

---

## クイックスタート

### 1. インストール

```bash
git clone https://github.com/densuke/fuga-memory
cd fuga-memory
uv sync
```

> **初回起動時**: ruri-v3-310m モデルを自動ダウンロードします（約 600MB）。ダウンロードはバックグラウンドで行われ、完了まで最初の保存・検索リクエストがブロックされます。

### 2. 設定ファイルを配置（任意）

デフォルト設定のまま使う場合はスキップできます。カスタマイズしたい場合はテンプレートをコピーします。

```bash
# macOS
mkdir -p ~/Library/Application\ Support/fuga-memory
cp config.toml.example ~/Library/Application\ Support/fuga-memory/config.toml

# Linux
mkdir -p ~/.config/fuga-memory
cp config.toml.example ~/.config/fuga-memory/config.toml
```

### 3. Claude Code に登録

`~/.claude/settings.json` を開き、`mcpServers` に追加します（`/path/to/fuga-memory` は実際のパスに変更してください）。

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/fuga-memory", "fuga-memory", "serve"]
    }
  }
}
```

### 4. Stop フックで自動保存（任意）

セッション終了時に会話内容を自動保存するには `hooks` セクションも追加します。

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/fuga-memory", "fuga-memory", "serve"]
    }
  },
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --project /path/to/fuga-memory fuga-memory save --stdin --session-id \"${CLAUDE_SESSION_ID:-unknown}\" --source claude_code",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

### 5. 動作確認

Claude Code を再起動すると MCP ツールが有効になります。Claude に話しかけてみてください。

```
あなたはfuga-memoryの search_memory ツールを使えます。
「Python の asyncio について」と検索してみてください。
```

---

## MCP サーバーの仕組み

`fuga-memory serve` は **stdio transport** で動作します。

```
Claude Code ←─ stdin/stdout ─→ fuga-memory serve (子プロセス)
```

**ポイント:**

- Claude Code が `mcpServers` の設定を読み、**必要に応じて自動でプロセスを起動・停止**します
- 常駐サーバーを手動で立ち上げておく必要はありません
- HTTP ポートも使用しません

### Stop フックでの自動保存の仕組み

```
セッション終了
    ↓
Claude Code が Stop フックを実行
    ↓
fuga-memory save --stdin --session-id <id>  （短命な1回限りのプロセス）
    ↓
SQLite に保存完了
```

Stop フックは MCP サーバーとは独立して動作します。フックが走るときに MCP サーバーが起動している必要はありません。

---

## Claude Code での使い方

MCP ツールが有効になると、以下の 3 つのツールが使えるようになります。

### save_memory — 記憶を保存

```
save_memory(content="今日Pythonのasyncioを勉強した", session_id="my-session")
```

| 引数 | 型 | 説明 |
|------|----|------|
| `content` | str | 保存するテキスト（必須） |
| `session_id` | str | セッション識別子（必須） |
| `source` | str | ソース識別子（デフォルト: `"manual"`） |

### search_memory — 記憶を検索

```
search_memory(query="Pythonの非同期処理", top_k=5)
```

| 引数 | 型 | 説明 |
|------|----|------|
| `query` | str | 検索クエリ（必須） |
| `top_k` | int | 返す最大件数（デフォルト: 5） |

返り値: `[{"id", "score", "content", "session_id", "source", "created_at"}, ...]`（score 降順）

### list_sessions — セッション一覧

```
list_sessions(limit=20)
```

返り値: `[{"session_id", "memory_count", "last_updated"}, ...]`

---

## CLI リファレンス

MCP 経由ではなく、コマンドラインから直接操作できます。

### serve — MCP サーバーを起動

```bash
uv run fuga-memory serve
```

通常は手動で起動する必要はありません。Claude Code が自動で管理します。
他の MCP クライアント（Gemini CLI 等）と接続する場合や動作確認時に使用します。

### search — 記憶を検索

```bash
uv run fuga-memory search "Rustのlifetimeについて"
uv run fuga-memory search "Python" --top-k 10
```

### save — 記憶を保存

3 種類の入力方式があります。

```bash
# 引数として直接渡す
uv run fuga-memory save "今日学んだこと" --session-id my-session

# ファイルから読み込む
uv run fuga-memory save --file notes.txt --session-id my-session

# 標準入力から読み込む（パイプ）
echo "パイプで渡す内容" | uv run fuga-memory save --stdin --session-id my-session
cat transcript.txt | uv run fuga-memory save --stdin --session-id my-session
```

---

## 設定

### 設定ファイル（推奨）

以下の順で探索し、最初に見つかったものを使用します。

| 優先度 | OS | パス |
|--------|-----|------|
| 1 | macOS | `~/Library/Application Support/fuga-memory/config.toml` |
| 2 | Linux / 共通 | `$XDG_CONFIG_HOME/fuga-memory/config.toml`（未設定時: `~/.config/fuga-memory/config.toml`） |
| 3 | 共通 | `~/.fuga-memory.toml` |

**テンプレートから作成:**

```bash
cp config.toml.example ~/.config/fuga-memory/config.toml  # Linux
cp config.toml.example ~/Library/Application\ Support/fuga-memory/config.toml  # macOS
```

**設定例:**

```toml
[fuga-memory]
db_path = "~/.local/share/fuga-memory/memories.db"
decay_halflife_days = 14   # 記憶の半減期を2週間に変更
default_top_k = 10
```

### 環境変数

設定ファイルより優先されます。Docker / CI など、ファイル配置が難しい環境向けです。

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `FUGA_MEMORY_DB_PATH` | `~/.local/share/fuga-memory/memories.db` | DB ファイルパス |
| `FUGA_MEMORY_MODEL_NAME` | `cl-nagoya/ruri-v3-310m` | 埋め込みモデル |
| `FUGA_MEMORY_THREAD_WORKERS` | CPU 数 ÷ 2 | 推論スレッド数 |
| `FUGA_MEMORY_RRF_K` | `60` | RRF の k パラメータ |
| `FUGA_MEMORY_DECAY_HALFLIFE_DAYS` | `30` | 時間減衰の半減期（日） |
| `FUGA_MEMORY_DEFAULT_TOP_K` | `5` | デフォルト検索件数 |

詳細は `.env.example` を参照してください。

### 優先順位

```
デフォルト値  <  設定ファイル  <  環境変数
```

---

## データの場所

| 項目 | デフォルトパス |
|------|--------------|
| DB ファイル | `~/.local/share/fuga-memory/memories.db` |
| モデルキャッシュ | `~/.cache/huggingface/` |

DB ファイルは SQLite 単一ファイルです。バックアップは `cp memories.db memories.db.bak` で行えます。

---

## 技術スタック

- Python 3.13, uv, fastmcp
- SQLite + FTS5（trigram トークナイザ）+ sqlite-vec
- sentence-transformers + cl-nagoya/ruri-v3-310m（ONNX バックエンド）
- ThreadPoolExecutor + asyncio

## ライセンス

MIT
