.PHONY: setup build smoke data train train-comparison validate-neural optimize-neural benchmark-neural validate evaluate test calibrate optimize benchmark

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

train-comparison:
	uv run --frozen python -m scripts.train_comparison

validate:
	uv run --frozen python -m scripts.validate

evaluate:
	uv run --frozen python -m scripts.evaluate

test:
	uv run --frozen pytest -q

benchmark:
	uv run --frozen python -m scripts.benchmark

calibrate:
	uv run --frozen python -m scripts.calibrate

optimize:
	uv run --frozen python -m scripts.optimize

validate-neural:
	uv run --frozen python -m scripts.validate --model neural

optimize-neural:
	uv run --frozen python -m scripts.optimize --model neural

benchmark-neural:
	uv run --frozen python -m scripts.benchmark --model neural
