-include .env
export

install:
	uv sync

dev:
	uv run python -m crm_agent

test:
	uv run pytest

eval:
	uv run python -m crm_agent.eval

.PHONY: install dev test eval
