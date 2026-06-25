# ESPHome Documentation Semantic Search MCP Server

This repository contains a Model Context Protocol (MCP) server that provides semantic vector search and raw page retrieval for the official ESPHome documentation (`esphome.io`).

It uses **LiteLLM** (or any OpenAI-compatible embeddings API) to generate text embeddings, indexes them into a local **SQLite** database, and exposes tools to search and fetch documents dynamically.

---

## Features
- **Semantic Search**: Vector similarity search (cosine similarity using NumPy) over chunked documentation pages.
- **Dual Transport Protocol Support**: Compatible with both modern **Streamable HTTP** (GET/POST/DELETE to `/sse`) and legacy **SSE** (GET `/sse` and POST `/messages/`) transport specs. Fully tested with Google's Antigravity (`agy`) CLI and other MCP hosts.
- **Auto-Sync & Indexing**: Automatically clones the ESPHome docs repository and schedules periodic incremental sync pulls (default is 24 hours).
- **Docker Ready**: Packaged for easy deployment with `docker-compose`, mounting persistent state for both the SQLite database and local Git repository clone.

---

## Repository Structure
- **`config.py`**: Resolves configuration from environment variables or a local `.env` file. Sets default paths, credentials, and configurations.
- **`db.py`**: Handles SQLite database schema creation, indexing inserts, cascade-deletion of outdated chunks, and vectorized cosine similarity calculations using `numpy`.
- **`indexer.py`**: Coordinates cloning and pulling the git repository, parsing frontmatter and body of Markdown/MDX files, chunking text, and fetching embeddings from the API.
- **`server.py`**: The FastAPI/Starlette web server. Includes background synchronization loops, exposes standard HTTP endpoints, and implements the MCP tools.

---

## Exposed MCP Tools

### 1. `search_docs(query: str, limit: int = 5) -> str`
Performs a semantic similarity search over all chunked documentation files using the configured embeddings API.
- **Arguments**:
  - `query` (str): Search term or phrase.
  - `limit` (int, optional): Maximum matching chunks to return (default: `5`).
- **Returns**: A formatted string containing the top similarity matches, the relative file paths, sections, similarity scores, and content snippets.

### 2. `get_doc_page(path: str) -> str`
Retrieves the raw Markdown/MDX contents of a specific documentation file from the repository (e.g. `components/sensor/wifi.rst` or custom component guides).
- **Arguments**:
  - `path` (str): Relative file path in the repository (e.g., `src/content/docs/components/sensor/wifi.mdx`).
- **Security**: Validates that the path does not traverse outside the clone directory boundary.

### 3. `sync_docs() -> str`
Manually triggers a Git pull on the documentation repository and runs an incremental index update, parsing modified files and requesting embeddings for new sections.
- **Returns**: Statistics on added, updated, and deleted files.

### 4. `get_index_status() -> str`
Queries the database to return high-level index statistics.
- **Returns**: Total number of indexed documents, total chunks created, and the timestamp of the last index update.

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

### Options (`.env`)
- **`LITELLM_API_BASE`**: Endpoint URL for the LiteLLM or OpenAI-compatible embeddings server (e.g., `http://10.100.0.50:4000/v1`).
- **`LITELLM_API_KEY`**: Your API authentication key.
- **`LITELLM_MODEL`**: Name of the embedding model to use (e.g., `text-embedding-qwen3-embedding-0.6b`).
- **`DATA_DIR`**: Local path where data is stored (SQLite file `db.sqlite` and the Git repository clone `esphome_docs_repo`). Defaults to `./data`.
- **`SYNC_INTERVAL_HOURS`**: Frequency of scheduled git sync and indexing checks (set to `0` to disable auto-sync).
- **`ESPHOME_DOCS_REPO_URL`**: Git repository clone URL (defaults to `https://github.com/esphome/esphome.io.git`).

---

## How to Run

### Option 1: Via Docker Compose (Recommended)
This runs the server in an isolated container and maps `./data` in the project root to `/data` in the container for persistence.

```bash
docker-compose up --build -d
```

The server will start listening at `http://localhost:8000`.

### Option 2: Running Locally
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Uvicorn application server:
   ```bash
   python server.py
   ```

---

## Connecting with Antigravity (`agy`)

Add the server to your `~/.gemini/antigravity-cli/settings.json` (or `.agents/mcp_config.json` depending on configuration):

```json
{
  "mcpServers": {
    "esphome-doc-search": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Since the server supports modern Streamable HTTP protocols and has DNS rebinding protection disabled by default, the CLI can successfully connect to the `/sse` route from any host on the network.
