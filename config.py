import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Data directory configuration
# Inside container, this will be /data.
# Locally, it will default to ./data.
DATA_DIR_STR = os.getenv("DATA_DIR", "./data")
DATA_DIR = Path(DATA_DIR_STR).resolve()

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = DATA_DIR / "db.sqlite"

# Git repository path
REPO_PATH = DATA_DIR / "esphome_docs_repo"

# Repo URL
REPO_URL = os.getenv("ESPHOME_DOCS_REPO_URL", "https://github.com/esphome/esphome.io.git")

# LiteLLM Configuration
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://10.100.0.50:4000/v1")
# Normalize base URL (strip trailing slash for endpoint construction)
if LITELLM_API_BASE.endswith("/"):
    LITELLM_API_BASE = LITELLM_API_BASE[:-1]

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-emOgH32VfC7TK_knr5BTHQ")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "text-embedding-qwen3-embedding-0.6b")

# Scheduler Config
try:
    SYNC_INTERVAL_HOURS = int(os.getenv("SYNC_INTERVAL_HOURS", "24"))
except ValueError:
    SYNC_INTERVAL_HOURS = 24
