.PHONY: install lint typecheck test run qdrant-up

install:
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy

test:
	pytest

run:
	uvicorn app.main:app --reload

qdrant-up:
	docker compose up -d qdrant
