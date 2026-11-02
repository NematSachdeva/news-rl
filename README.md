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
├── Makefile                     # One-command setup, seed, run, test, eval
├── run.sh                       # Shell script alternative to make
├── .env.example                 # Central environment variable template
├── .python-version              # Pinned Python version (3.12.6)
├── requirements-lock.txt        # Reproducible frozen pip environment
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
    ├── data/                    # SQLite databases & demo CSV data
    ├── models/                  # DDQN PyTorch checkpoint storage (.pt)
    ├── scripts/
    │   └── seed_demo_data.py    # Synthetic dataset generator & SQLite seeder
    ├── tests/                   # Pytest test suite (114 passing tests)
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

## Testing & Evaluation

### Run Test Suite
```bash
make test
# or: cd final && ../.venv/bin/pytest -v
```
All 114 tests cover API endpoints, authentication, cache, graph memory, reading history, live news, mood affinities, rate limiting, recommender scoring, and bandit policies.

### Run Recommender Evaluation Harness
Simulates 5 distinct users reading recommendations across 15 interaction rounds:
```bash
make eval
# or: cd final && ../.venv/bin/python evaluation.py --users 5 --rounds 15
```

Observed run results:
- **Precision@K (early rounds)**: 0.651
- **Precision@K (late rounds)**: 0.696
- **Learning Delta**: $+0.044$ (confirms precision increases within a session as RL learns).

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
| `NEUROFEED_NEWS_API_KEY` | `""` | NewsAPI.org API key for Live News bulletin (optional) |
