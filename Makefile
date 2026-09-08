.PHONY: setup build smoke data train validate evaluate test benchmark

setup:
	git submodule update --init --recursive
	uv sync --frozen

build:
	cargo build --release --locked

smoke:
	uv run --frozen python -m scripts.smoke_model
	uv run --frozen python -m scripts.zk

data:
	uv run --frozen python -m scripts.download_data

train:
	uv run --frozen python -m scripts.train

validate:
	uv run --frozen python -m scripts.validate

evaluate:
	uv run --frozen python -m scripts.evaluate

test:
	uv run --frozen pytest -q

benchmark:
	uv run --frozen python -m scripts.benchmark
