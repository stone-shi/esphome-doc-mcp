from pathlib import Path
from unittest import mock

import pytest
import config
import db
import indexer
import server as server_module


@pytest.fixture(autouse=True)
def disable_auto_sync(monkeypatch):
    """Prevent the server from starting background sync during tests."""
    monkeypatch.setattr(config, "SYNC_INTERVAL_HOURS", 0)


@pytest.fixture
def temp_repo(tmp_path, monkeypatch):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    monkeypatch.setattr(config, "REPO_PATH", repo_path)
    return repo_path


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return db_path


class TestGetDocPage:
    def test_get_doc_page_success(self, temp_repo):
        (temp_repo / "page.mdx").write_text("# Page\n\nContent.")

        result = server_module.get_doc_page("page.mdx")

        assert "# Page" in result
        assert "Content." in result

    def test_get_doc_page_empty_path(self):
        result = server_module.get_doc_page("  ")
        assert "Path cannot be empty" in result

    def test_get_doc_page_path_traversal(self, temp_repo):
        result = server_module.get_doc_page("../outside.txt")
        assert "Access denied" in result

    def test_get_doc_page_not_found(self, temp_repo):
        result = server_module.get_doc_page("missing.md")
        assert "not found" in result

    def test_get_doc_page_read_error(self, temp_repo):
        (temp_repo / "page.mdx").write_text("content")

        with mock.patch("builtins.open", side_effect=OSError("disk error")):
            result = server_module.get_doc_page("page.mdx")

        assert "Error reading document file" in result


class TestGetIndexStatus:
    def test_get_index_status(self, temp_db):
        conn = db.get_connection(temp_db)
        db.store_document(conn, "doc.md", "Doc", "hash")
        db.store_chunk(conn, 1, 0, "Header", "content", [0.1, 0.2])
        conn.close()

        result = server_module.get_index_status()

        assert "Total Documents Indexed: 1" in result
        assert "Total Chunks Created: 1" in result

    def test_get_index_status_db_error(self, temp_db, monkeypatch):
        monkeypatch.setattr(db, "get_index_stats", lambda conn: (_ for _ in ()).throw(RuntimeError("db corrupt")))
        result = server_module.get_index_status()
        assert "Error reading database index status" in result


class TestSearchDocs:
    @pytest.mark.asyncio
    async def test_search_docs_empty_query(self):
        result = await server_module.search_docs("  ")
        assert "valid query" in result

    @pytest.mark.asyncio
    async def test_search_docs_no_results(self, temp_db, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "get_embeddings",
            lambda texts, api_base, model, api_key: [[1.0, 0.0]],
        )

        result = await server_module.search_docs("query")

        assert "No matching documentation found" in result

    @pytest.mark.asyncio
    async def test_search_docs_returns_formatted_results(self, temp_db, monkeypatch):
        conn = db.get_connection(temp_db)
        doc_id = db.store_document(conn, "doc.md", "My Doc", "hash")
        db.store_chunk(conn, doc_id, 0, "Section", "content", [1.0, 0.0])
        conn.close()

        monkeypatch.setattr(
            indexer,
            "get_embeddings",
            lambda texts, api_base, model, api_key: [[1.0, 0.0]],
        )

        result = await server_module.search_docs("query")

        assert "Match #1" in result
        assert "doc.md" in result
        assert "My Doc" in result
        assert "Section" in result

    @pytest.mark.asyncio
    async def test_search_docs_empty_embedding_result(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "get_embeddings",
            lambda texts, api_base, model, api_key: [],
        )

        result = await server_module.search_docs("query")
        assert "Failed to generate query embedding" in result

    @pytest.mark.asyncio
    async def test_search_docs_embedding_error(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "get_embeddings",
            lambda texts, api_base, model, api_key: (_ for _ in ()).throw(RuntimeError("API down")),
        )

        result = await server_module.search_docs("query")
        assert "Error executing semantic search" in result


class TestSyncDocs:
    @pytest.mark.asyncio
    async def test_sync_docs_success(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "run_indexing",
            lambda: {"added": 1, "updated": 2, "deleted": 3},
        )

        result = await server_module.sync_docs()

        assert "Added documents: 1" in result
        assert "Updated documents: 2" in result
        assert "Deleted documents: 3" in result

    @pytest.mark.asyncio
    async def test_sync_docs_failure(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "run_indexing",
            lambda: {"error": "repository failed"},
        )

        result = await server_module.sync_docs()

        assert "Synchronization failed" in result
        assert "repository failed" in result

    @pytest.mark.asyncio
    async def test_sync_docs_exception(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "run_indexing",
            lambda: (_ for _ in ()).throw(RuntimeError("crash")),
        )

        result = await server_module.sync_docs()
        assert "Error running manual synchronization" in result


class TestHTTPRoutes:
    def test_sse_route_registered(self):
        """Verify /sse and /messages routes are registered on the FastAPI app."""
        routes = [r.path for r in server_module.fastapi_app.routes]
        assert "/sse" in routes
        assert "/messages" in routes or "/messages/" in routes
