SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY      ?= python
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

export PYTHONPATH := $(PWD)

ART_DIR := artifacts
MODEL   := $(ART_DIR)/sentiment_tfidf_logreg.joblib
VECT    := $(ART_DIR)/tfidf_vectorizer.joblib
MATRIX  := $(ART_DIR)/tfidf_matrix.npz
CORPUS  := $(ART_DIR)/corpus.joblib

.PHONY: help all train index api api-dev ui lint format test clean

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sed 's/:.*## /  /' | sort

all: train index api-dev ## train + build index + start API (dev)

train: ## Train baseline sentiment model (TF-IDF + Logistic Regression)
	@echo ">> Training sentiment baseline model..."
	$(PY) -m src.models.train_baseline
	@test -f "$(MODEL)" || { echo "Model not found at $(MODEL)"; exit 1; }

index: ## Build TF-IDF index for recommender
	@echo ">> Building TF-IDF index..."
	$(PY) -m src.recommender.build_index
	@test -f "$(VECT)" -a -f "$(MATRIX)" -a -f "$(CORPUS)" || { echo "Index artifacts missing"; exit 1; }

api: ## Start FastAPI (prod-ish, no reload, binds to 127.0.0.1)
	@echo ">> Starting FastAPI server (no reload)..."
	$(UVICORN) src.api.main:app --host 127.0.0.1 --port 8000

api-dev: ## Start FastAPI (dev, reload, binds to 0.0.0.0)
	@echo ">> Starting FastAPI server (dev, reload)..."
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Start Streamlit UI
	@echo ">> Starting Streamlit UI..."
	$(STREAMLIT) run src/app/streamlit_app.py

lint: ## Ruff lint (auto-fix)
	ruff check . --fix

format: ## Black format
	black .

test: ## Run test suite
	pytest -q

clean: ## Remove local artifacts and caches (not versioned)
	rm -rf $(ART_DIR) data .ruff_cache __pycache__ */__pycache__ .pytest_cache .mypy_cache
