#!/usr/bin/env python3
"""
check_no_icons.py — Scan NeuroFeed frontend files and fail if any
icon-like content is found.

Checks for:
  1. Emoji characters (Unicode ranges: emoticons, pictographs, symbols)
  2. <svg> elements
  3. Icon-font class patterns (fa-, bi-, material-icons, lucide-, etc.)
  4. Icon-library script imports (fontawesome, feather, bootstrap-icons, etc.)
  5. Arrow/cross/pictograph symbols used as UI text (→ ↗ × ✕ ⏱ 🎯 etc.)
     — a curated list of the specific offenders found in this codebase

Usage:
  python scripts/check_no_icons.py              # exits 1 on violation
  python -m pytest scripts/check_no_icons.py    # can also run as pytest

Or via Makefile:
  make check-icons
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

# ── Target directory ──────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
FRONTEND    = REPO_ROOT / "final" / "frontend"
EXTENSIONS  = {".html", ".js", ".css"}

# ── Patterns to flag ──────────────────────────────────────────────────────────

# Emoji ranges (broad coverage; false positives are extremely rare in HTML/JS)
# Covers: Emoticons & Transport, Supplemental, Miscellaneous Symbols &
#         Pictographs, Dingbats, Enclosed Alphanumeric Supplement, etc.
EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FFFF"   # Supplemental Multilingual Plane emoji block
    r"\u2600-\u27BF"            # Misc Symbols, Dingbats, arrows-B, etc.
    r"\u2300-\u23FF"            # Misc Technical (⏱ lives here)
    r"\u2700-\u27BF"            # Dingbats
    r"]"
)

# Allow-list: Unicode codepoints that are typographic (not pictographic icons)
# · (middle dot, separator), — (em dash), … (ellipsis) are fine.
# Only emoji/pictographs should be flagged — narrow the check with a denylist
# of the specific known offenders too.
KNOWN_ICON_CHARS = frozenset([
    "→", "↗", "←", "↙", "↑", "↓",    # directional arrows used as icons
    "×", "✕", "✗", "✘",               # cross/close icons
    "⏱", "🎯", "🔥", "🗞", "🗞️",      # specific emoji found in this codebase
    "❤", "♥", "♡",                    # heart icons
    "★", "☆",                          # star icons
    "⚡", "🧐", "😌", "😤", "🥱",
    "🏃", "😊", "😰", "🎯",
])

SVG_RE = re.compile(r"<svg\b", re.IGNORECASE)

# Icon-font class patterns (e.g. class="fa-home", class="bi bi-house")
ICON_CLASS_RE = re.compile(
    r'class\s*=\s*["\'][^"\']*\b('
    r'fa-|bi\s+bi-|material-icons|lucide-|ph-|tabler-icon|heroicon'
    r')[^"\']*["\']',
    re.IGNORECASE,
)

# Icon-library script/link imports
ICON_IMPORT_RE = re.compile(
    r'(fontawesome|bootstrap-icons|feather|heroicons|phosphor|lucide|'
    r'remixicon|tabler-icons|ionicons|material-icons)'
    r'[^"\']*\.(js|css)',
    re.IGNORECASE,
)

# ── Comment stripping ─────────────────────────────────────────────────────────
# Strip JS/HTML/CSS comments before scanning for emoji so that commented-out
# old code doesn't trigger false positives.

def strip_comments(text: str, ext: str) -> str:
    if ext in (".js", ".css"):
        # /* ... */ block comments
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        # // line comments (JS only)
        if ext == ".js":
            text = re.sub(r"//[^\n]*", " ", text)
    elif ext == ".html":
        # <!-- ... --> HTML comments
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    return text


# ── Violation type ────────────────────────────────────────────────────────────

class Violation(NamedTuple):
    file:    str
    line:    int
    kind:    str
    snippet: str


def _lineno(text: str, match_start: int) -> int:
    return text[:match_start].count("\n") + 1


def scan_file(path: Path) -> list[Violation]:
    raw   = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_comments(raw, path.suffix.lower())
    rel   = str(path.relative_to(REPO_ROOT))
    violations: list[Violation] = []

    # 1 — emoji (scanned on stripped text)
    for m in EMOJI_RE.finditer(clean):
        ch = m.group()
        violations.append(Violation(
            file=rel, line=_lineno(clean, m.start()),
            kind="emoji", snippet=repr(ch),
        ))

    # 2 — known icon characters (in stripped text, raw too for the HTML entities)
    for i, line_text in enumerate(clean.splitlines(), start=1):
        for ch in KNOWN_ICON_CHARS:
            if ch in line_text:
                violations.append(Violation(
                    file=rel, line=i,
                    kind="icon-char", snippet=repr(ch) + " in: " + line_text.strip()[:80],
                ))

    # 3 — <svg> elements (in raw text, before comment stripping, because svgs
    #     are never in comments in this codebase)
    for m in SVG_RE.finditer(raw):
        violations.append(Violation(
            file=rel, line=_lineno(raw, m.start()),
            kind="svg-element", snippet="<svg> found",
        ))

    # 4 — icon-font class patterns
    for m in ICON_CLASS_RE.finditer(clean):
        violations.append(Violation(
            file=rel, line=_lineno(clean, m.start()),
            kind="icon-font-class", snippet=m.group()[:100],
        ))

    # 5 — icon-library imports
    for m in ICON_IMPORT_RE.finditer(raw):
        violations.append(Violation(
            file=rel, line=_lineno(raw, m.start()),
            kind="icon-library-import", snippet=m.group()[:100],
        ))

    # Deduplicate (same line, same kind, same snippet prefix)
    seen: set[tuple] = set()
    unique: list[Violation] = []
    for v in violations:
        key = (v.file, v.line, v.kind, v.snippet[:30])
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def run_check() -> int:
    """Return 0 if clean, 1 if violations found."""
    all_violations: list[Violation] = []
    scanned = 0
    for path in sorted(FRONTEND.rglob("*")):
        if path.suffix.lower() in EXTENSIONS and path.is_file():
            scanned += 1
            all_violations.extend(scan_file(path))

    if not all_violations:
        print(f"check_no_icons: OK — scanned {scanned} files, no violations found.")
        return 0

    print(f"\ncheck_no_icons: FAIL — {len(all_violations)} violation(s) in {scanned} files\n")
    current_file = None
    for v in sorted(all_violations, key=lambda x: (x.file, x.line)):
        if v.file != current_file:
            print(f"  {v.file}")
            current_file = v.file
        print(f"    line {v.line:>4}  [{v.kind}]  {v.snippet}")
    print()
    return 1


# ── pytest integration ────────────────────────────────────────────────────────
def test_no_icons_in_frontend():
    """Pytest-runnable version of check_no_icons."""
    all_violations: list[Violation] = []
    for path in sorted(FRONTEND.rglob("*")):
        if path.suffix.lower() in EXTENSIONS and path.is_file():
            all_violations.extend(scan_file(path))

    if all_violations:
        lines = [f"\nFrontend icon violations ({len(all_violations)} found):"]
        for v in sorted(all_violations, key=lambda x: (x.file, x.line)):
            lines.append(f"  {v.file}:{v.line}  [{v.kind}]  {v.snippet}")
        raise AssertionError("\n".join(lines))


if __name__ == "__main__":
    sys.exit(run_check())

