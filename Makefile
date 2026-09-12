PY ?= python3
MODEL ?= qwen2.5-vl-3b

.PHONY: setup budget smoke bench ui test clean

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
	@echo "Done. Activate with: source .venv/bin/activate"

## Prove we are inside the 6B constraint (reads real checkpoint headers)
budget:
	$(PY) -m glance.cli budget --model $(MODEL)

## Generate synthetic UI screens with known ground truth
data:
	$(PY) -m glance.cli gen-data --n 40 --out assets/generated

## End-to-end sanity run on one generated screen
smoke:
	$(PY) -m glance.cli describe assets/generated/ui_000.png --report out/smoke.html

## Full accuracy + latency + memory benchmark
bench:
	$(PY) -m glance.cli bench --n 40 --out runs/

ui:
	$(PY) -m glance.cli ui

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf out runs assets/generated __pycache__ .pytest_cache