# Gemini CLI Usage Guide for fuga-memory

This guide explains how to use **fuga-memory** as a Long-Term Memory MCP server specifically for [Gemini CLI](https://github.com/google/gemini-cli).

## 1. Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed.
- `fuga-memory` repository cloned.
- **Run `uv sync` inside the cloned `fuga-memory` directory.**

### Configuration

Add `fuga-memory` to your Gemini CLI settings file located at `~/.gemini/settings.json`.

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/path/to/fuga-memory",
        "fuga-memory",
        "serve"
      ]
    }
  }
}
```

Replace `/path/to/fuga-memory` with the absolute path to your cloned repository.

## 2. Available Tools

Once configured, Gemini CLI will have access to the following tools:

### `save_memory`
Saves a new memory to the database.

- **`content`** (string, required): The text content to remember.
- **`session_id`** (string, required): A unique identifier for the current session.
- **`source`** (string, optional): Where the memory came from (defaults to `"manual"`).

**Example:**
> Gemini, remember that I prefer using Python for data analysis.
> (Gemini will call `save_memory(content="User prefers using Python for data analysis.", session_id="...")`)

### `search_memory`
Searches for relevant memories using hybrid search (Full-Text Search + Vector Search).

- **`query`** (string, required): The search term or question.
- **`top_k`** (integer, optional): Maximum number of results to return (defaults to 5).

**Example:**
> What are my preferences for data analysis?
> (Gemini will call `search_memory(query="data analysis preferences")`)

### `list_sessions`
Lists the sessions that have stored memories.

- **`limit`** (integer, optional): Maximum number of sessions to list (defaults to 20).

## 3. Session-Based Memory Management

`fuga-memory` is designed to organize memories by `session_id`. When using Gemini CLI, you can use the session ID to keep track of context within a specific project or conversation.

### Automated Context Injection

You can instruct Gemini to always look up relevant memories at the start of a session:

> "Search my memories for any context related to the current project."

### Session Identification

Gemini CLI provides session-specific environment variables that can be used if you are running custom scripts, though typically the agent handles the `session_id` automatically when calling the MCP tools.

## 4. Tips for Gemini CLI Users

- **Be Descriptive**: The more context you provide in `save_memory`, the better the hybrid search will perform.
- **Hybrid Search Power**: `fuga-memory` combines keyword matching (FTS5) with semantic understanding (Vector Search), making it very robust for retrieving information even if you don't use the exact same words.
- **Decay System**: Remember that `fuga-memory` has a time-decay feature (default half-life of 30 days). Older memories will naturally have lower scores unless they are highly relevant to your query.
