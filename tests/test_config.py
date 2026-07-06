import os
import importlib
from pathlib import Path
import pytest

import config as config_module


class TestConfig:
    def test_default_data_dir_created(self, tmp_path, monkeypatch):
        """DATA_DIR should be created and resolve to an absolute path."""
        data_dir = tmp_path / "test_data"
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.delenv("ESPHOME_DOCS_REPO_URL", raising=False)

        importlib.reload(config_module)

        assert config_module.DATA_DIR == data_dir.resolve()
        assert config_module.DATA_DIR.exists()
        assert config_module.DB_PATH == config_module.DATA_DIR / "db.sqlite"
        assert config_module.REPO_PATH == config_module.DATA_DIR / "esphome_docs_repo"

    def test_repo_url_from_env(self, monkeypatch):
        """REPO_URL should respect the environment variable."""
        monkeypatch.setenv("ESPHOME_DOCS_REPO_URL", "https://example.com/repo.git")

        importlib.reload(config_module)

        assert config_module.REPO_URL == "https://example.com/repo.git"

    def test_litellm_api_base_trailing_slash_removed(self, monkeypatch):
        """LITELLM_API_BASE should have a trailing slash stripped."""
        monkeypatch.setenv("LITELLM_API_BASE", "http://example.com/v1/")

        importlib.reload(config_module)

        assert config_module.LITELLM_API_BASE == "http://example.com/v1"

    def test_litellm_api_base_no_trailing_slash(self, monkeypatch):
        """LITELLM_API_BASE without trailing slash should remain unchanged."""
        monkeypatch.setenv("LITELLM_API_BASE", "http://example.com/v1")

        importlib.reload(config_module)

        assert config_module.LITELLM_API_BASE == "http://example.com/v1"

    def test_sync_interval_hours_parses_int(self, monkeypatch):
        """SYNC_INTERVAL_HOURS should parse a valid integer."""
        monkeypatch.setenv("SYNC_INTERVAL_HOURS", "12")

        importlib.reload(config_module)

        assert config_module.SYNC_INTERVAL_HOURS == 12

    def test_sync_interval_hours_invalid_defaults_to_24(self, monkeypatch):
        """SYNC_INTERVAL_HOURS should fall back to 24 on invalid input."""
        monkeypatch.setenv("SYNC_INTERVAL_HOURS", "not-a-number")

        importlib.reload(config_module)

        assert config_module.SYNC_INTERVAL_HOURS == 24

    def test_default_litellm_values(self, monkeypatch):
        """Default LiteLLM settings should be present."""
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("LITELLM_MODEL", raising=False)

        importlib.reload(config_module)

        assert config_module.LITELLM_API_BASE == "http://10.100.0.50:4000/v1"
        assert config_module.LITELLM_API_KEY == "sk-emOgH32VfC7TK_knr5BTHQ"
        assert config_module.LITELLM_MODEL == "text-embedding-qwen3-embedding-0.6b"
