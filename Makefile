.PHONY: install lint typecheck test run qdrant-up qdrant-up-local

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

# Docker-free alternative: run the standalone Qdrant server binary instead
# (download from https://github.com/qdrant/qdrant/releases, Windows asset
# is qdrant-x86_64-pc-windows-msvc.zip). Needed because QDRANT_STORAGE_PATH
# in .env is unset, so the app connects to QDRANT_HOST:QDRANT_PORT instead
# of opening an embedded on-disk store — embedded mode only allows a single
# process to hold the storage folder at a time, which caused 500s whenever
# the API was accidentally started twice.
qdrant-up-local:
	"C:/Users/ZehraMey/qdrant/qdrant.exe"
