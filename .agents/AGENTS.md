# ESPHome Documentation Semantic Search MCP Server - Workspace Rules

This document outlines workspace guidelines and structural overview of the ESPHome search MCP server.

## Codebase Architecture
- **`config.py`**: Reads variables from the environment and loads `.env` locally. Use this module to reference paths (`DB_PATH`, `REPO_PATH`) or LiteLLM credentials.
- **`db.py`**: Manages the SQLite database and executes vectorized similarity calculations via `numpy`. Do not use external vector databases unless explicitly asked by the user.
- **`indexer.py`**: Handles git sync operations, markdown document parsing, yaml frontmatter extraction, text chunking by headers, and LiteLLM embedding API requests.
- **`server.py`**: FastAPI app running the SSE transport. Includes a lifespan background thread runner for scheduled synchronization.
- **`Dockerfile` / `docker-compose.yml`**: Mounts stateful directories to the `/data/` container path.

## Development Constraints
1. **Thread Safety & Async safety**:
   - SQLite is synchronous, and embedding calculations/file parsing are resource heavy.
   - When calling `run_indexing()` or `get_embeddings()` from an async context in `server.py`, always offload the synchronous function calls to a worker thread using `asyncio.to_thread` or `run_in_executor` to prevent blocking the async event loop.
2. **Path Traversal Security**:
   - Any tools retrieving raw files (e.g. `get_doc_page`) must resolve path strings relative to `config.REPO_PATH` and call `is_relative_to` to prevent access to host files.
3. **LiteLLM Compatibility**:
   - Always construction LiteLLM request payloads using standard OpenAI compatibility (`POST /v1/embeddings`, containing JSON body with `model` and `input`).
4. **SQLite Serialization**:
   - When writing vector embeddings into the `chunks` table, serialize float32 NumPy arrays to binary bytes using `.tobytes()`.
   - When reading, convert them back using `np.frombuffer(blob, dtype=np.float32)`. Ensure you check for empty/null blobs.
