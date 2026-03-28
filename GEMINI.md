# fuga-memory: Gemini CLI 利用ガイド

このガイドでは、[Gemini CLI](https://github.com/google/gemini-cli) で長期記憶 MCP サーバー **fuga-memory** を使用する方法について説明します。

## 1. セットアップ

### 前提条件

- [uv](https://docs.astral.sh/uv/) がインストールされていること。
- `fuga-memory` リポジトリをクローン済みであること。
- **クローンした `fuga-memory` ディレクトリ内で `uv sync` を実行済みであること。**

### 設定

Gemini CLI の設定ファイル（通常は `~/.gemini/settings.json`）に `fuga-memory` を追加します。

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/path/to/fuga-memory", // ← クローン先の絶対パスに変更（例: /Users/yourname/src/fuga-memory）
        "fuga-memory",
        "serve"
      ]
    }
  }
}
```

`/path/to/fuga-memory` は、クローンしたリポジトリの絶対パスに置き換えてください。

## 2. 利用可能なツール

設定が完了すると、Gemini CLI から以下のツールが利用可能になります。

### `save_memory`
新しい記憶をデータベースに保存します。

- **`content`** (string, 必須): 記憶するテキスト内容。
- **`session_id`** (string, 必須): 現在のセッションを一意に識別する ID。
- **`source`** (string, 任意): 記憶のソース（デフォルトは `"manual"`）。

**例:**
> 「データ分析には Python を使うのが好みだと覚えておいて」
> （Gemini が `save_memory(content="ユーザーはデータ分析に Python を使うことを好む", session_id="...")` を呼び出します）

### `search_memory`
ハイブリッド検索（全文検索 + ベクトル検索）を使用して、関連する記憶を検索します。

- **`query`** (string, 必須): 検索ワードまたは質問。
- **`top_k`** (integer, 任意): 返す結果の最大件数（デフォルトは 5）。

**例:**
> 「データ分析に関する私の好みは何？」
> （Gemini が `search_memory(query="データ分析の好み")` を呼び出します）

### `list_sessions`
記憶が保存されているセッションの一覧を表示します。

- **`limit`** (integer, 任意): 表示するセッションの最大件数（デフォルトは 20）。

## 3. セッションベースの記憶管理

`fuga-memory` は `session_id` ごとに記憶を整理するように設計されています。Gemini CLI を使用する場合、セッション ID を使用して特定のプロジェクトや会話のコンテキストを追跡できます。

### 自動コンテキスト注入

セッションの開始時に、関連する記憶を検索するように Gemini に指示できます。

> 「現在のプロジェクトに関連するコンテキストを記憶から検索して」

### セッション識別

Gemini CLI は、エージェントが MCP ツールを呼び出す際に自動的に `session_id` を処理しますが、カスタムスクリプトを実行する場合などは、セッション固有の環境変数を利用することも可能です。

## 4. Gemini CLI ユーザーへのヒント

- **具体的に記述する**: `save_memory` で提供するコンテキストが具体的なほど、ハイブリッド検索の精度が向上します。
- **ハイブリッド検索の活用**: `fuga-memory` はキーワード一致（FTS5）と意味理解（ベクトル検索）を組み合わせています。全く同じ言葉を使わなくても、関連情報を柔軟に取得できます。
- **時間減衰システム**: `fuga-memory` には時間減衰機能（デフォルトの半減期は 30 日）があります。古い記憶は、クエリに対する関連性が非常に高い場合を除き、自然にスコアが低下します。
