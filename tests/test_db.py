import sqlite3
import numpy as np
import pytest

import db as db_module


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    db_module.init_db(db_path)
    connection = db_module.get_connection(db_path)
    yield connection
    connection.close()


class TestDatabase:
    def test_init_db_creates_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        db_module.init_db(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "documents" in tables
        assert "chunks" in tables

    def test_get_all_documents_empty(self, conn):
        docs = db_module.get_all_documents(conn)
        assert docs == {}

    def test_store_and_get_document(self, conn):
        doc_id = db_module.store_document(conn, "path/to/doc.md", "My Doc", "abc123")
        assert doc_id is not None

        docs = db_module.get_all_documents(conn)
        assert "path/to/doc.md" in docs
        assert docs["path/to/doc.md"]["id"] == doc_id
        assert docs["path/to/doc.md"]["file_hash"] == "abc123"

    def test_store_document_updates_existing(self, conn):
        db_module.store_document(conn, "path/to/doc.md", "My Doc", "abc123")
        db_module.store_document(conn, "path/to/doc.md", "My Doc Updated", "def456")

        docs = db_module.get_all_documents(conn)
        assert len(docs) == 1
        assert docs["path/to/doc.md"]["file_hash"] == "def456"

    def test_delete_document_by_path(self, conn):
        db_module.store_document(conn, "path/to/doc.md", "My Doc", "abc123")
        db_module.delete_document_by_path(conn, "path/to/doc.md")

        docs = db_module.get_all_documents(conn)
        assert "path/to/doc.md" not in docs

    def test_delete_document_cascades_to_chunks(self, conn):
        doc_id = db_module.store_document(conn, "path/to/doc.md", "My Doc", "abc123")
        db_module.store_chunk(conn, doc_id, 0, "Header", "content", [0.1, 0.2, 0.3])
        db_module.delete_document_by_path(conn, "path/to/doc.md")

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 0

    def test_store_chunk_embedding_blob(self, conn):
        doc_id = db_module.store_document(conn, "path/to/doc.md", "My Doc", "abc123")
        embedding = [0.1, 0.2, 0.3]
        db_module.store_chunk(conn, doc_id, 0, "Header", "content", embedding)

        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM chunks WHERE document_id = ?", (doc_id,))
        row = cursor.fetchone()
        restored = np.frombuffer(row[0], dtype=np.float32)
        np.testing.assert_array_almost_equal(restored, np.array(embedding, dtype=np.float32))

    def test_get_index_stats_empty(self, conn):
        stats = db_module.get_index_stats(conn)
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0
        assert stats["last_indexed"] == "Never"

    def test_get_index_stats_with_data(self, conn):
        db_module.store_document(conn, "doc1.md", "Doc 1", "hash1")
        doc_id = db_module.store_document(conn, "doc2.md", "Doc 2", "hash2")
        db_module.store_chunk(conn, doc_id, 0, "Header", "content", [0.1, 0.2])

        stats = db_module.get_index_stats(conn)
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 1
        assert stats["last_indexed"] != "Never"

    def test_search_similar_chunks_empty_db(self, conn):
        results = db_module.search_similar_chunks(conn, [0.1, 0.2, 0.3])
        assert results == []

    def test_search_similar_chunks_returns_sorted_results(self, conn):
        doc_id = db_module.store_document(conn, "doc.md", "Doc", "hash")
        db_module.store_chunk(conn, doc_id, 0, "A", "first", [1.0, 0.0, 0.0])
        db_module.store_chunk(conn, doc_id, 1, "B", "second", [0.0, 1.0, 0.0])
        db_module.store_chunk(conn, doc_id, 2, "C", "third", [0.5, 0.5, 0.0])

        results = db_module.search_similar_chunks(conn, [0.0, 1.0, 0.0], limit=2)

        assert len(results) == 2
        assert results[0]["header"] == "B"
        assert results[1]["header"] == "C"
        assert results[0]["score"] > results[1]["score"]

    def test_search_similar_chunks_zero_query_vector(self, conn):
        doc_id = db_module.store_document(conn, "doc.md", "Doc", "hash")
        db_module.store_chunk(conn, doc_id, 0, "A", "first", [1.0, 0.0])

        results = db_module.search_similar_chunks(conn, [0.0, 0.0], limit=1)

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-9)

    def test_search_skips_null_embedding(self, conn):
        """Chunks with NULL embeddings are skipped; returns empty if no valid embeddings."""
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("INSERT INTO documents (filepath, title, file_hash) VALUES ('d.md', 'D', 'h')")
        doc_id = conn.execute("SELECT id FROM documents WHERE filepath='d.md'").fetchone()[0]
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, header, content, embedding) VALUES (?, 0, 'H', 'C', NULL)",
            (doc_id,),
        )
        conn.commit()

        results = db_module.search_similar_chunks(conn, [0.1, 0.2])
        assert results == []

    def test_search_mixed_valid_and_null_embeddings(self, conn):
        """Valid embeddings are found even when NULL entries exist alongside."""
        doc_id = db_module.store_document(conn, "doc.md", "Doc", "hash")
        db_module.store_chunk(conn, doc_id, 0, "Valid", "content", [1.0, 0.0])
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, header, content, embedding) VALUES (?, 1, 'Null', 'content', NULL)",
            (doc_id,),
        )
        conn.commit()

        results = db_module.search_similar_chunks(conn, [1.0, 0.0])
        assert len(results) == 1
        assert results[0]["header"] == "Valid"
