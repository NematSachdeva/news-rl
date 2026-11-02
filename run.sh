#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-run}"

case "$cmd" in
  setup)
    echo "==> Setting up environment..."
    PYTHON=$(which python3.12 2>/dev/null || which python3)
    if [ ! -d ".venv" ]; then
      "$PYTHON" -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip setuptools wheel
    .venv/bin/pip install -r final/requirements-dev.txt
    echo "==> Setup finished."
    ;;
  ingest)
    echo "==> Ingesting real news from NewsData.io..."
    (cd final && ../.venv/bin/python -m news_ingest.run)
    ;;
  ingest-force)
    echo "==> Force ingesting real news from NewsData.io..."
    (cd final && ../.venv/bin/python -m news_ingest.run --force)
    ;;
  test)
    echo "==> Running tests..."
    (cd final && ../.venv/bin/pytest -v)
    ;;
  eval)
    echo "==> Running evaluation..."
    (cd final && ../.venv/bin/python evaluation.py --users 5 --rounds 15)
    ;;
  run)
    echo "==> Starting NeuroFeed on http://127.0.0.1:8000 (UI: /app/, Docs: /docs)..."
    (cd final && ../.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000)
    ;;
  all)
    "$0" setup
    "$0" ingest
    "$0" test
    echo "==> Ready! Run './run.sh' to start."
    ;;
  *)
    echo "Usage: $0 {setup|ingest|ingest-force|run|test|eval|all}"
    exit 1
    ;;
esac
