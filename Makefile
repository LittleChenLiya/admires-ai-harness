.PHONY: install format format-check lint typecheck test test-runtime test-security test-e2e coverage web-check check demo eval dev

install:
	uv sync --all-extras
	pnpm --dir web install --frozen-lockfile

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests
	pnpm --dir web format

format-check:
	uv run ruff format --check src tests
	pnpm --dir web format:check

lint:
	uv run ruff check src tests
	pnpm --dir web lint

typecheck:
	uv run mypy src
	pnpm --dir web typecheck

test:
	uv run python -m pytest tests -m "not e2e"
	pnpm --dir web test

test-runtime:
	uv run python -m pytest tests/unit/runtime --cov=admires_harness.runtime --cov-branch --cov-report=term-missing --cov-fail-under=85

test-security:
	uv run python -m pytest tests -m security

test-e2e:
	pnpm --dir web test:e2e

coverage:
	uv run python -m pytest tests --cov=admires_harness --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=0

web-check:
	pnpm --dir web format:check
	pnpm --dir web lint
	pnpm --dir web typecheck
	pnpm --dir web test
	pnpm --dir web build

check: format-check lint typecheck test test-runtime web-check

demo:
	uv run admire demo all

eval:
	uv run admire evaluate evaluation/reference.jsonl

dev:
	uv run admire dev
