# ===== EN Sentiment + Recommender — Makefile =====
# Usage examples:
#   make help
#   make train
#   make eval
#   make predict TEXT="this was surprisingly good!"
#   make inspect
#   make clean
#
# Optional sampling (faster iterations):
#   make train TRAIN_N=5000 TEST_N=2000
#   make eval TEST_N=2000

# Python interpreter (use your venv's python if needed)
PY ?= python

# Paths
ARTIFACTS := artifacts
MODEL := $(ARTIFACTS)/sentiment_tfidf_logreg.joblib

# Sampling knobs (0 = disabled)
TRAIN_N ?= 0
TEST_N ?= 0

.PHONY: help setup train eval predict inspect clean

help:
	@echo "Targets:"
	@echo "  make train                - Train baseline (TF-IDF + LogReg)"
	@echo "  make eval                 - Evaluate baseline (ROC-AUC, CM, report)"
	@echo "  make predict TEXT='... '  - Predict label for a raw text"
	@echo "  make inspect              - Show top positive/negative features"
	@echo "  make clean                - Remove artifacts/"
	@echo
	@echo "Sampling (optional):"
	@echo "  make train TRAIN_N=5000 TEST_N=2000"
	@echo "  make eval  TEST_N=2000"

setup:
	@mkdir -p $(ARTIFACTS)
	@echo "OK: artifacts/ ready"

train: setup
	BASELINE_TRAIN_SAMPLE=$(TRAIN_N) BASELINE_TEST_SAMPLE=$(TEST_N) \
	$(PY) -m src.models.train_baseline

eval: setup
	BASELINE_TEST_SAMPLE=$(TEST_N) \
	$(PY) -m src.models.eval_baseline

# Use: make predict TEXT="this was surprisingly good!"
predict:
ifndef TEXT
	$(error Provide TEXT='your text' e.g. make predict TEXT="great movie")
endif
	$(PY) -m src.models.predict "$(TEXT)"

inspect:
	$(PY) -m src.models.inspect_baseline

clean:
	@rm -rf $(ARTIFACTS)
	@echo "Removed: $(ARTIFACTS)/"
