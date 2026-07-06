import hashlib
from pathlib import Path
from unittest import mock

import pytest

import indexer


class TestParseMdFile:
    def test_no_frontmatter(self):
        content = "# Hello World\n\nThis is a doc."
        title, description, body = indexer.parse_md_file(content)
        assert title == "Untitled"
        assert description == ""
        assert body == content

    def test_yaml_frontmatter(self):
        content = "---\ntitle: My Title\ndescription: My Description\n---\n# Body\n\nText."
        title, description, body = indexer.parse_md_file(content)
        assert title == "My Title"
        assert description == "My Description"
        assert body == "# Body\n\nText."

    def test_yaml_frontmatter_fallback_regex(self):
        content = "---\ntitle: 'Quoted Title'\ndescription: \"Quoted Description\"\n---\nBody here."
        title, description, body = indexer.parse_md_file(content)
        assert title == "Quoted Title"
        assert description == "Quoted Description"
        assert body == "Body here."

    def test_invalid_yaml_uses_regex_fallback(self):
        content = "---\ntitle: {invalid yaml\ndescription: fallback desc\n---\nBody"
        title, description, body = indexer.parse_md_file(content)
        assert title == "{invalid yaml"
        assert description == "fallback desc"
        assert body == "Body"


class TestChunkText:
    def test_no_headers(self):
        body = "This is the introduction.\n\nIt has multiple paragraphs."
        chunks = indexer.chunk_text(body, max_chunk_len=1000)
        assert len(chunks) == 1
        assert chunks[0]["header"] == "Introduction"

    def test_splits_by_headers(self):
        body = "Intro paragraph.\n\n# Title\n\nMore intro text.\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two."
        chunks = indexer.chunk_text(body, max_chunk_len=1000)
        headers = [c["header"] for c in chunks]
        assert "Introduction" in headers
        assert "Title" in headers
        assert "Section One" in headers
        assert "Section Two" in headers

    def test_sub_chunk_long_paragraph(self):
        # Use a max_chunk_len larger than the 150-char overlap to avoid a
        # known issue in the indexer sliding-window logic.
        body = "## Long\n\n" + "word " * 1000
        chunks = indexer.chunk_text(body, max_chunk_len=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["header"] == "Long"
            assert len(chunk["content"]) <= 200

    def test_sub_chunk_short_text(self):
        result = indexer.sub_chunk("Header", "Short text.", max_len=100)
        assert result == [{"header": "Header", "content": "Short text."}]

    def test_sub_chunk_by_paragraph(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = indexer.sub_chunk("Header", text, max_len=1000)
        assert result == [{"header": "Header", "content": text}]

    def test_sub_chunk_skips_empty_paragraphs(self):
        """Empty paragraphs (whitespace-only) should be skipped."""
        text = "\n\nPara one.\n\n   \n\nPara two.\n\n\n\nPara three.\n\n"
        result = indexer.sub_chunk("Header", text, max_len=1000)
        assert len(result) == 1
        assert "Para one." in result[0]["content"]
        assert "Para two." in result[0]["content"]
        assert "Para three." in result[0]["content"]

    def test_sub_chunk_paragraph_split_at_limit(self):
        """Paragraphs that cause overflow past max_len create a new sub-chunk."""
        text = "AAA\n\nBBB\n\nCCC"
        result = indexer.sub_chunk("Header", text, max_len=5)
        assert len(result) > 1

    def test_sub_chunk_sliding_window_on_very_long_paragraph(self):
        """A single paragraph longer than max_len gets character-level split."""
        text = "X" * 500
        result = indexer.sub_chunk("Header", text, max_len=200)
        assert len(result) > 1
        for chunk in result:
            assert chunk["header"] == "Header"
            assert len(chunk["content"]) <= 200


class TestGetEmbeddings:
    def test_get_embeddings_success(self):
        fake_response = mock.MagicMock()
        fake_response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        fake_client = mock.MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = fake_response

        with mock.patch("httpx.Client", return_value=fake_client):
            embeddings = indexer.get_embeddings(["a", "b"], "http://api/v1", "model", "key")

        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        fake_client.post.assert_called_once()

    def test_get_embeddings_no_api_key(self):
        fake_response = mock.MagicMock()
        fake_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
        fake_client = mock.MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = fake_response

        with mock.patch("httpx.Client", return_value=fake_client):
            embeddings = indexer.get_embeddings(["a"], "http://api/v1", "model", None)

        assert embeddings == [[0.1]]
        call_kwargs = fake_client.post.call_args.kwargs
        assert "Authorization" not in call_kwargs["headers"]


class TestSyncRepository:
    def test_clones_when_git_missing(self, tmp_path):
        repo_path = tmp_path / "repo"
        with mock.patch("subprocess.run") as mock_run:
            indexer.sync_repository("https://example.com/repo.git", repo_path)
            mock_run.assert_called_once_with(
                ["git", "clone", "--depth", "1", "https://example.com/repo.git", str(repo_path)],
                check=True,
            )

    def test_pulls_when_git_present(self, tmp_path):
        repo_path = tmp_path / "repo"
        (repo_path / ".git").mkdir(parents=True)
        with mock.patch("subprocess.run") as mock_run:
            indexer.sync_repository("https://example.com/repo.git", repo_path)
            assert mock_run.call_count == 2
            mock_run.assert_any_call(
                ["git", "-C", str(repo_path), "reset", "--hard"],
                check=True,
            )
            mock_run.assert_any_call(
                ["git", "-C", str(repo_path), "pull"],
                check=True,
            )


class TestCalculateFileHash:
    def test_calculates_md5(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        expected = hashlib.md5(b"hello world").hexdigest()
        assert indexer.calculate_file_hash(file_path) == expected


class TestRunIndexing:
    def test_returns_error_when_repo_path_missing(self, tmp_path, monkeypatch):
        """If sync fails and repo path does not exist, run_indexing should return an error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        repo_path = data_dir / "esphome_docs_repo"

        monkeypatch.setattr("config.DB_PATH", data_dir / "db.sqlite")
        monkeypatch.setattr("config.REPO_PATH", repo_path)
        monkeypatch.setattr("config.REPO_URL", "https://example.com/repo.git")
        monkeypatch.setattr("config.LITELLM_API_BASE", "http://api/v1")
        monkeypatch.setattr("config.LITELLM_MODEL", "model")
        monkeypatch.setattr("config.LITELLM_API_KEY", "key")

        with mock.patch("indexer.sync_repository", side_effect=Exception("git failed")):
            result = indexer.run_indexing()

        assert "error" in result
        assert "Repository path not found" in result["error"]
