"""
tests/test_no_icons.py — Runs the check_no_icons scanner as a pytest test.

This ensures the frontend never regresses to containing emoji, SVG icons,
icon-font classes, or icon library imports. The scanner is the authoritative
source of truth; this file just provides a pytest entry point so `make test`
catches violations automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from scripts/ at the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_no_icons import test_no_icons_in_frontend  # noqa: F401
# Importing the function is all pytest needs — it discovers test_ names.

