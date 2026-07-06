#!/bin/bash
set -e

# Change directory to the location of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

echo "========================================="
echo " ESPHome Doc MCP Test Runner            "
echo "========================================="
echo "Project directory: $SCRIPT_DIR"
echo "Virtual environment: $VENV_DIR"
echo ""

# Ensure Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH." >&2
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Using $PYTHON_VERSION"
echo ""

# Create virtual environment if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
    echo ""
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify we are using the venv Python
VENV_PYTHON=$(which python)
echo "Active Python: $VENV_PYTHON"
echo ""

# Upgrade pip to avoid install issues
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install project dependencies
echo "Installing dependencies from $REQUIREMENTS..."
pip install -r "$REQUIREMENTS"

echo ""

# Run the test suite
echo "Running pytest..."
pytest -v

echo ""
echo "========================================="
echo " TESTS COMPLETED SUCCESSFULLY           "
echo "========================================="
