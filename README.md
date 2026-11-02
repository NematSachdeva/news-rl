# NeuroFeed — Adaptive Local-First News Recommender

> Based on the open-source base project [News-Recommender](https://github.com/Dracula-5/News-Recommender) by Dheeraj.

NeuroFeed is a local-first, interactive news feed recommendation engine built with FastAPI, PyTorch, and SQLite. It adapts its feed recommendations after **every single user interaction** (clicks, likes, dislikes, dwell time, scroll depth, and optional webcam-based attention) using a hybrid reinforcement learning and contextual bandit architecture.

The UI is designed as an editorial-style live feed that reads cleanly like a digital newspaper, while running continuous online learning under the hood.

---

## Intended Architecture & How It Works

### 1. Hybrid Cold-Start & Policy Transition
- **Cold-Start Phase (`interaction_count < 5`)**: For a user's first few interactions, recommendations lean primarily on the user's declared category interests (from onboarding) and trending articles (`pref: 0.20`, `sim: 0.18`, `trend: 0.18`), keeping untrained RL/bandit weights low (`q: 0.08`, `ts: 0.00`).
- **Post-Cold-Start Phase (`interaction_count >= 5`)**: The system shifts priority to learned behavior: the Dueling Double DQN Q-value becomes the dominant scoring component (`q: 0.20`), bandit exploration activates (`ts: 0.05`), while reliance on static declared preferences drops (`pref: 0.09`).
- **Internal Bandits (`HybridPolicy`)**: Combines Discounted Thompson Sampling (Beta priors with decay `0.995`) with DDQN Q-values, enabling quick recovery and discovery of emerging category affinities.

### 2. Multi-Signal Reward Function
Interactions yield an immediate reward in approximately `[-3.55, +4.25]`:
- **Explicit Likes/Dislikes**: $\pm 2.0$ base signal.
- **Dwell Time**: $+1.2$ for long reads ($> 6\text{s}$), $-1.2$ for quick skips ($< 2\text{s}$).
- **Content Similarity**: $+0.5 \times \text{NLP similarity}$.
- **Scroll Depth**: $+0.4$ for deep scrolls ($> 70\%$), $-0.2$ for immediate shallow skips ($< 15\%$).
- **Mood-Congruence**: $\pm 0.15$ adjustment reinforcing or penalizing alignment with active decayed mood state.

### 3. Online DDQN Training
- Interactions are logged online and pushed to an experience replay buffer (capacity 1,000 transitions).
- While the replay buffer has fewer than 16 samples, the agent observes without gradient steps.
- Once $\ge 16$ transitions exist, every interaction triggers an online mini-batch update (batch size 16) with target network synchronization every 20 steps and epsilon exploration decay.
- Checkpoints are saved periodically to local disk in a background executor pool.

### 4. Attention & Physical Signals (Graceful Fallback)
- Optional server-side webcam attention tracking (OpenCV + MediaPipe FaceMesh) measures eye openness, gaze, and head movement.
- When `NEUROFEED_ENABLE_WEBCAM_ATTENTION=false` (default) or when no camera device is available, the system falls back gracefully to neutral attention ($0.5$), ensuring zero disruption.

---

## Project Structure

```
.
├── Makefile                     # One-command setup, seed, ingest, run, test, eval
├── run.sh                       # Shell script alternative to make
├── .env.example                 # Central environment variable template
├── .python-version              # Pinned Python version (3.12.6)
├── requirements-lock.txt        # Reproducible frozen pip environment
├── docs/
│   └── architecture.md          # Comprehensive recommendation & RL flow docs
├── deploy/                      # Deployment configs (preserved, not in active use)
│   ├── README.md                # Deployment notes
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-entrypoint.sh
│   ├── render.yaml
│   ├── netlify.toml
│   └── vercel.json
└── final/                       # Main application codebase
    ├── backend/
    │   └── main.py              # FastAPI server & route handlers
    ├── frontend/                # Vanilla JS / HTML / CSS newspaper UI
    ├── data/                    # SQLite databases & disk caches
    ├── models/                  # DDQN PyTorch checkpoint storage (.pt)
    ├── news_ingest/             # NewsData.io provider, deduplication & classification
    │   ├── base.py              # Provider interface & NormalizedArticle dataclass
    │   ├── newsdata.py          # NewsData.io provider adapter with disk caching
    │   ├── category_mapping.py  # Explicit mapper & TF-IDF fallback classifier
    │   ├── category_mapping.json# Canonical category dictionary & rule table
    │   ├── dedupe.py            # URL hash & title Jaccard deduplication
    │   ├── credit_tracker.py    # Daily credit quota tracker (SQLite-backed)
    │   ├── storage.py           # Ingest pipeline coordinator & candidate store
    │   ├── scheduler.py         # Background periodic ingestion loop
    │   └── run.py               # Ingestion CLI entry point
    ├── scripts/
    │   └── seed_demo_data.py    # Synthetic dataset generator & SQLite seeder
    ├── tests/                   # Pytest test suite (121 passing tests)
    │   └── test_news_ingest.py  # NewsData ingest, dedupe, mapping & quota tests
    ├── recommender.py           # Core ranking, scoring & candidate pools
    ├── ddqn_agent.py            # Dueling Double DQN agent & PyTorch model
    ├── rl_policies.py           # Thompson Sampling, UCB1, HybridPolicy
    ├── graph_memory.py          # NetworkX interest graph with decay
    ├── nlp_engine.py            # BM25 + TF-IDF + LSA ensemble
    ├── reranking.py             # MMR diversity re-ranking
    ├── webcam_attention.py      # OpenCV + MediaPipe attention tracker
    ├── config.py                # Pydantic-settings configuration
    └── evaluation.py            # Simulated-user offline evaluation harness
```

---

## Quick Start (Local Setup)

### Prerequisites
- macOS or Linux
- Python 3.12 (pinned via `.python-version`)

### One-Command Setup & Run

```bash
# 1. Setup virtualenv and install all dependencies
make setup

# 2. Seed synthetic articles and demo user profiles
make seed

# 3. Start the FastAPI recommender server
make run
```

Alternatively, use `./run.sh`:
```bash
./run.sh all   # runs setup, seed, and runs the test suite
./run.sh run   # starts the development server
```

Open in your browser:
- **Interactive Feed UI**: [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/)
- **Interactive API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Real News Ingestion (NewsData.io)

NeuroFeed seamlessly ingests live, real-world news from [NewsData.io](https://newsdata.io/) into the candidate pool (`is_live=0`) so the DDQN agent, Thompson Sampling bandits, and interest graph learn directly from your real reading habits.

### Ingestion Features
- **Provider Adapter Architecture (`final/news_ingest/`)**: Extensible provider interface (`BaseNewsProvider`) with clean normalization, pagination via `nextPage`, exponential backoff on 429/5xx, and local disk caching in `final/data/cache/newsdata_raw/` so re-runs never burn API credits.
- **Credit Tracker (`final/news_ingest/credit_tracker.py`)**: SQLite-backed tracker enforcing the 200 daily credit limit (default budget: 180 credits) with UTC midnight resets.
- **Category Mapper & Fallback Classifier (`final/news_ingest/category_mapping.py`)**: Maps external NewsData.io categories to NeuroFeed's 10 internal categories via explicit rules in `category_mapping.json`. For low-confidence or unmapped items, an automated TF-IDF + Logistic Regression fallback classifier infers the single best category.
- **Article Deduplication (`final/news_ingest/dedupe.py`)**: Multi-stage deduplication by exact canonical URL hash and token-level title Jaccard similarity ($0.75$ threshold).
- **Online NLP Refit**: Following any ingestion batch, the TF-IDF, BM25, and LSA index automatically refits across the expanded corpus and performs an atomic pointer swap without dropping incoming recommendation requests.

### Ingestion Commands
```bash
# Ingest latest news using configured settings
make ingest

# Or via run.sh
./run.sh ingest

# Or directly with the CLI for custom options:
cd final && ../.venv/bin/python -m news_ingest.run --provider newsdata --countries in,us,gb --categories top,world,technology,business,sports --max 50
```

---

## Testing & Evaluation

### Run Test Suite
```bash
make test
# or: cd final && ../.venv/bin/pytest -v
```
All 121 tests cover API endpoints, real news ingestion, deduplication, category mapping, authentication, cache, graph memory, reading history, live news, mood affinities, rate limiting, recommender scoring, and bandit policies.

### Run Recommender Evaluation Harness
Simulates 5 distinct users reading recommendations across 15 interaction rounds:
```bash
make eval
# or: cd final && ../.venv/bin/python evaluation.py --users 5 --rounds 15
```

Observed run results:
- **Precision@K (early rounds)**: 0.607
- **Precision@K (late rounds)**: 0.657
- **Learning Delta**: $+0.051$ (confirms precision increases within a session as RL learns).

---

## Configuration Reference

All settings can be configured via environment variables or a `.env` file at the root:

| Variable | Default | Description |
|---|---|---|
| `NEUROFEED_ENV` | `development` | Runtime environment (`development`, `production`, `test`) |
| `NEUROFEED_DB_PATH` | `final/data/news_recommender.sqlite` | SQLite database file path |
| `NEUROFEED_AUTH_DB_PATH` | `final/data/auth.sqlite` | User credentials SQLite database path |
| `NEUROFEED_MODEL_DIR` | `final/models` | Directory for persisting PyTorch DDQN checkpoints |
| `NEUROFEED_ENABLE_WEBCAM_ATTENTION` | `false` | Enable OpenCV/MediaPipe webcam attention tracker |
| `NEUROFEED_COLD_START_INTERACTION_THRESHOLD` | `5` | Interaction count threshold before RL dominates scoring |
| `NEUROFEED_MMR_LAMBDA` | `0.75` | Diversity trade-off factor ($1.0$ = relevance, $0.0$ = diversity) |
| `NEWSDATA_API_KEY` | `""` | NewsData.io API key for real news ingestion |
| `NEWSDATA_DAILY_CREDIT_LIMIT` | `180` | Safety ceiling for daily NewsData API credit usage |
| `NEWSDATA_CATEGORIES` | `top,world,technology,business,sports` | Default comma-separated categories to fetch |
| `NEWSDATA_COUNTRIES` | `in,us,gb` | Default comma-separated country codes |
| `NEUROFEED_NEWS_API_KEY` | `""` | NewsAPI.org API key for Live News bulletin (optional) |
