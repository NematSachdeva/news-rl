.PHONY: help setup seed ingest run test eval clean all

PYTHON ?= $(shell which python3.12 2>/dev/null || which python3)
VENV ?= .venv
VENV_PYTHON = $(VENV)/bin/python
VENV_PIP = $(VENV)/bin/pip
VENV_PYTEST = $(VENV)/bin/pytest
VENV_UVICORN = $(VENV)/bin/uvicorn

help:
	@echo "NeuroFeed - Local-First News Recommender"
	@echo "=========================================="
	@echo "Available commands:"
	@echo "  make setup     - Create virtualenv and install dependencies"
	@echo "  make seed      - Seed synthetic news & user data into SQLite"
	@echo "  make ingest    - Ingest real news from NewsData.io into SQLite"
	@echo "  make run       - Start FastAPI recommender server on http://127.0.0.1:8000"
	@echo "  make test      - Run pytest test suite"
	@echo "  make eval      - Run evaluation harness (5 users x 15 rounds)"
	@echo "  make clean     - Remove cache and compiled files"
	@echo "  make all       - Run setup, seed, and test in one command"

setup:
	@echo "Setting up virtual environment with $(PYTHON)..."
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@$(VENV_PIP) install -r final/requirements-dev.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

seed:
	@echo "Seeding demo dataset into SQLite..."
	@cd final && ../$(VENV_PYTHON) scripts/seed_demo_data.py --import

ingest:
	@echo "Ingesting real news from NewsData.io..."
	@cd final && ../$(VENV_PYTHON) -m news_ingest.run

run:
	@echo "Starting NeuroFeed server on http://127.0.0.1:8000..."
	@echo "Open http://127.0.0.1:8000/app/ for the feed UI, or http://127.0.0.1:8000/docs for Swagger."
	@cd final && ../$(VENV_UVICORN) backend.main:app --reload --host 127.0.0.1 --port 8000

test:
	@echo "Running pytest test suite..."
	@cd final && ../$(VENV_PYTEST) -v

eval:
	@echo "Running recommender evaluation..."
	@cd final && ../$(VENV_PYTHON) evaluation.py --users 5 --rounds 15

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache final/.pytest_cache final/.coverage .coverage 2>/dev/null || true

all: setup seed test
	@echo ""
	@echo "All steps completed successfully!"
	@echo "To start the application, run: make run"
