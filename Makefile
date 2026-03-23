# ============================================================
# Demand Forecasting — Production Makefile
# ============================================================

.PHONY: install train serve clean help

CONFIG ?= config/config.yaml

# ── Install dependencies ─────────────────────────────────────
install:
	pip install -r requirements.txt --break-system-packages -q

# ── Full training pipeline ───────────────────────────────────
train: install
	@echo "══════════════════════════════════════════════════════════"
	@echo "  7-Eleven Demand Forecasting — Training Pipeline"
	@echo "══════════════════════════════════════════════════════════"
	python -m src.models.train $(CONFIG)

# ── Serve API ────────────────────────────────────────────────
serve: install
	@echo "Starting API server..."
	python -m src.api.serve

# ── Clean generated files ────────────────────────────────────
clean:
	rm -rf data/interim data/processed models/artifacts models/plots mlruns
	@echo "Cleaned."

# ── Help ─────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make install   — Install Python dependencies"
	@echo "  make train     — Run full training pipeline (preprocess → features → validate → optuna → cv → train → export)"
	@echo "  make serve     — Start FastAPI prediction server"
	@echo "  make clean     — Remove generated data, models, and plots"
