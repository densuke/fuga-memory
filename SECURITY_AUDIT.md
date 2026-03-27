# Security Audit Report: fuga-memory

Date: 2026-03-27
Status: Completed

## 1. Executive Summary
A comprehensive security audit of the `fuga-memory` project was conducted. The project demonstrates strong security practices for a local CLI/MCP tool, including the use of parameterized queries, input validation, and secure handling of model names.

## 2. Findings

### 2.1 SQL Injection (Low Risk)
- **Observation**: All database interactions in `src/fuga_memory/db/repository.py` and `src/fuga_memory/db/schema.py` use parameterized queries (`?` placeholders).
- **FTS5 Handling**: Full-Text Search (FTS5) queries are sanitized in `src/fuga_memory/search/fts.py` to remove special characters and logic operators, preventing syntax errors and unexpected behavior.
- **Verdict**: Safe.

### 2.2 Path Traversal (Low Risk)
- **Model Name**: The `model_name` configuration is validated against a strict ASCII regex in `src/fuga_memory/config.py`, preventing path traversal during model loading.
- **File Input**: The `fuga-memory save --file` option allows reading any file the user has permission to. While this is expected behavior for a CLI tool, users should be aware that an LLM with access to this tool could be tricked into reading sensitive files if not properly supervised.
- **Database Path**: The `db_path` is configurable. An attacker with access to the configuration could redirect the database to sensitive locations, but this requires existing filesystem access.
- **Verdict**: Safe for intended use cases.

### 2.3 Data Security (Medium Risk)
- **Encryption**: The SQLite database is stored as a plain file without encryption. Any user or process with read access to the filesystem can access the stored memories.
- **Logging**: The application uses standard logging and does not appear to log the actual content of memories or queries, only metadata and errors.
- **Verdict**: Acceptable for local personal use; encryption would be required for shared or multi-user environments.

### 2.4 Denial of Service / Resource Exhaustion (Low Risk)
- **Input Limits**: Strict limits are enforced in `src/fuga_memory/server.py`:
  - `content`: Max 100,000 characters.
  - `query`: Max 4,096 characters.
  - `top_k`: Max 100.
  - `limit`: Max 200.
- **File Size**: Stdin and file inputs are limited to 1MB in `src/fuga_memory/cli.py`.
- **Verdict**: Safe.

### 2.5 Dependencies (Low Risk)
- **Status**: The project uses standard, well-maintained libraries like `fastmcp`, `sentence-transformers`, and `sqlite-vec`.
- **Verdict**: Low risk, provided dependencies are kept up to date via `uv`.

## 3. Recommendations
1. **Encrypted Storage (Optional)**: For users handling highly sensitive information, consider supporting [SQLCipher](https://www.zetetic.net/sqlcipher/) or similar for database encryption.
2. **Audit Logging**: For sensitive environments, implement an audit log of which sessions and users are accessing or modifying memories.
3. **Model Verification**: Implement checksum verification for the downloaded model files to ensure integrity.

## 4. Conclusion
The `fuga-memory` project is well-architected with security in mind. It is suitable for its intended use as a personal long-term memory assistant for LLMs.
