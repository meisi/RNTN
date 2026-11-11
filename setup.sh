#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo 'This legacy runtime requires x86_64 Linux (including Ubuntu under WSL2).'
  echo 'For macOS or ARM hardware, request a separate compatible port before training.'
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first; see README.md, then run this script again.'
  exit 1
fi
export UV_CACHE_DIR="$bundle_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$bundle_dir/.python"
export UV_PYTHON_BIN_DIR="$bundle_dir/.bin"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_BIN_DIR"
uv python install 3.8.20
if [ ! -x "$bundle_dir/.venv/bin/python" ]; then
  uv venv --python "$UV_PYTHON_INSTALL_DIR/cpython-3.8-linux-x86_64-gnu/bin/python3.8" "$bundle_dir/.venv"
fi
uv pip install --python "$bundle_dir/.venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r "$bundle_dir/requirements.txt"
