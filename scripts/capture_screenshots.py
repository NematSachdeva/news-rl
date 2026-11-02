#!/usr/bin/env python3
"""
capture_screenshots.py — Capture screenshots of NeuroFeed UI in:
- Onboarding (light & dark, desktop & phone)
- Feed (light & dark, desktop & phone)
- Empty state (light & dark, desktop & phone)

Uses Google Chrome headless without requiring additional package installations.
"""

import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "final" / "frontend"
SCREENSHOTS_DIR = REPO_ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

VIEWPORTS = {
    "desktop": (1280, 840),
    "phone": (390, 844),
}

THEMES = ["light", "dark"]

INDEX_HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

def build_html(state: str, theme: str) -> str:
    html = INDEX_HTML
    # Insert theme attribute
    html = html.replace('<html lang="en">', f'<html lang="en" data-theme="{theme}">')
    # Fix stylesheet path for temp file
    css_path = (FRONTEND_DIR / "styles.css").resolve().as_uri()
    html = html.replace('href="styles.css"', f'href="{css_path}"')
    # Remove app scripts so it doesn't trigger auth redirects during static capture
    html = html.replace('<script src="config.js"></script>', '')
    html = html.replace('<script src="app.js"></script>', '')

    # State modifications
    if state == "onboarding":
        # Onboarding is visible by default, feed is hidden
        pass
    elif state == "feed":
        # Hide onboarding, show feed and sidebar
        html = html.replace('class="panel onboarding" id="onboarding"', 'class="panel onboarding hidden" id="onboarding"')
        html = html.replace('class="feed hidden" id="feed"', 'class="feed" id="feed"')
        html = html.replace('class="sidebar hidden" id="sidebar"', 'class="sidebar" id="sidebar"')
        # Populate dummy news and categories
        html = html.replace('<span id="category" class="hero-badge">Category</span>', '<span id="category" class="hero-badge">Technology</span>')
        html = html.replace('<h1 id="headline">Ready when you are.</h1>', '<h1 id="headline">DeepMind Unveils Next-Generation Autonomous Agent Architecture</h1>')
        html = html.replace('<p id="abstract">Open your feed to get your first recommendation.</p>', '<p id="abstract">New reinforcement learning agents blend dueling deep Q-networks with Bayesian bandits to dynamically balance exploratory catalog discovery and high-precision relevance.</p>')
        html = html.replace('<span class="byline" id="byline">NEUROFEED DESK</span>', '<span class="byline" id="byline">MIT TECHNOLOGY REVIEW</span>')
        html = html.replace('<span id="readTimer" class="byline-timer">just now</span>', '<span id="readTimer" class="byline-timer">2 hours ago</span>')
        html = html.replace('<span id="statReadTime">– min read</span>', '<span id="statReadTime">4 min read</span>')
        html = html.replace('<span id="statMatch">– match</span>', '<span id="statMatch">96% match</span>')
        html = html.replace('class="read-link hidden"', 'class="read-link"')
        # Populate mood chips
        mood_html = '<button class="mood-chip active">Curious</button><button class="mood-chip">Focused</button><button class="mood-chip">Relaxed</button><button class="mood-chip">Energetic</button>'
        html = html.replace('<div class="mood-chips" id="moodChips"></div>', f'<div class="mood-chips" id="moodChips">{mood_html}</div>')
        # Sidebar categories
        cats = ['For You', 'Technology', 'Science', 'Business', 'World', 'Health']
        nav_html = ''.join([f'<a href="#" class="sidebar-link{" active" if c == "For You" else ""}">{c}</a>' for c in cats])
        html = html.replace('<nav class="sidebar-nav" id="sidebarCategories"></nav>', f'<nav class="sidebar-nav" id="sidebarCategories">{nav_html}</nav>')
    elif state == "empty":
        # Hide onboarding, show feed
        html = html.replace('class="panel onboarding" id="onboarding"', 'class="panel onboarding hidden" id="onboarding"')
        html = html.replace('class="feed hidden" id="feed"', 'class="feed" id="feed"')
        html = html.replace('<span id="category" class="hero-badge">Category</span>', '<span id="category" class="hero-badge">Notice</span>')
        html = html.replace('<h1 id="headline">Ready when you are.</h1>', '<h1 id="headline">No news loaded yet</h1>')
        html = html.replace('<p id="abstract">Open your feed to get your first recommendation.</p>', '<p id="abstract">No real articles found in the local catalog. Add NEWSDATA_API_KEY to your .env file and run "make ingest" to load real news.</p>')
        html = html.replace('<span class="byline" id="byline">NEUROFEED DESK</span>', '<span class="byline" id="byline">SYSTEM DESK</span>')
        html = html.replace('<span id="readTimer" class="byline-timer">just now</span>', '<span id="readTimer" class="byline-timer"></span>')
        html = html.replace('class="actions-row"', 'class="actions-row hidden"')
        html = html.replace('id="catalogStatus" class="catalog-status-badge">Updated: Just now</span>', 'id="catalogStatus" class="catalog-status-badge empty">Catalog empty • Run "make ingest"</span>')
    return html

def capture():
    if not os.path.exists(CHROME_BIN):
        print(f"Chrome binary not found at {CHROME_BIN}; skipping screenshots.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        for state in ["onboarding", "feed", "empty"]:
            for theme in THEMES:
                html_content = build_html(state, theme)
                html_file = Path(tmpdir) / f"{state}_{theme}.html"
                html_file.write_text(html_content, encoding="utf-8")

                for vp_name, (w, h) in VIEWPORTS.items():
                    out_name = f"{state}_{theme}_{vp_name}.png"
                    out_path = SCREENSHOTS_DIR / out_name
                    cmd = [
                        CHROME_BIN,
                        "--headless=new",
                        "--disable-gpu",
                        f"--screenshot={out_path}",
                        f"--window-size={w},{h}",
                        f"file://{html_file}",
                    ]
                    print(f"Capturing {out_name} ({w}x{h})...")
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    if out_path.exists():
                        print(f"  Saved {out_path.name} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    capture()
