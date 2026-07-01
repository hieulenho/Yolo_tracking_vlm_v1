.PHONY: help install dev-install format lint test clean train-players eval inference

help:
	@echo "Available commands:"
	@echo "  make install         - Install production dependencies"
	@echo "  make dev-install     - Install with dev dependencies"
	@echo "  make format          - Format code with ruff"
	@echo "  make lint            - Lint code"
	@echo "  make test            - Run tests"
	@echo "  make train-players   - Train YOLOv8m player detector"
	@echo "  make eval            - Evaluate player detector"
	@echo "  make inference VIDEO=path/to/video.mp4"
	@echo "  make clean           - Remove cache files"

install:
	pip install -r requirements.txt

dev-install:
	pip install -e ".[dev]"
	pre-commit install

format:
	ruff format src scripts tests
	ruff check --fix src scripts tests

lint:
	ruff check src scripts tests
	mypy src

test:
	pytest tests/ --cov=src --cov-report=term-missing

train-players:
	python scripts/train_players_yolo.py --data configs/data/players.yaml --model yolov8m.pt --epochs 100

eval:
	python scripts/evaluate.py --weights models/players_best.pt --data configs/data/players.yaml

inference:
	python scripts/run_inference.py --video $(VIDEO) --output output.mp4

clean:
	powershell -NoProfile -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force; Remove-Item -Recurse -Force .pytest_cache,.ruff_cache,.mypy_cache -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Filter *.pyc | Remove-Item -Force"
