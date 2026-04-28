# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`. Use `src/main.py` as the entrypoint that wires the LangChain agent, OpenAI chat model, and search/tool integrations. Put reusable stock utilities in `src/tools/`; today `src/tools/stock.py` contains LangChain `@tool` functions for price and moving-average lookups. Keep environment-specific values in `.env`, dependency metadata in `pyproject.toml`, and lockfile updates in `uv.lock`.

## Build, Test, and Development Commands
- `uv sync` — preferred setup when `uv` is installed; creates a reproducible environment from `pyproject.toml` and `uv.lock`.
- `.venv/bin/python src/main.py` — run the local agent with the repository virtualenv.
- `.venv/bin/python -m compileall src` — lightweight verification that all Python modules parse correctly.
- `source .venv/bin/activate` — activate the checked-in virtualenv before running ad hoc scripts or REPL checks.

## Coding Style & Naming Conventions
Target Python 3.12+ and use 4-space indentation. Follow PEP 8 naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes. Group imports as standard library, third-party, then local modules. No formatter or linter is configured yet, so keep changes Black-compatible and easy to scan in review. Keep LangChain tool functions small, deterministic, and documented: the `@tool` docstring should clearly describe when the agent should call it. Prefer explicit return strings and avoid hidden side effects.

## Testing Guidelines
There is no committed `tests/` suite yet; add one for any non-trivial feature. Place tests under `tests/` using `test_*.py` filenames. Mock OpenAI, DuckDuckGo, and live stock-data calls so tests stay deterministic and offline-friendly. Before opening a PR, at minimum run `.venv/bin/python -m compileall src`; if you add tests, run `.venv/bin/python -m pytest`.

## Commit & Pull Request Guidelines
This repository has no commit history yet, so start cleanly: use an imperative subject line and keep commits focused. Follow the repo’s Lore commit format when committing, including useful trailers such as `Constraint:`, `Confidence:`, and `Tested:`. PRs should explain the user-visible change, list environment/config updates, and include verification evidence (command output or sample agent responses). Link related issues when applicable.

## Security & Configuration Tips
Keep secrets only in `.env`; `OPENAI_API_KEY` is required for `src/main.py`. Never commit `.env`, `.venv/`, generated caches, or OMX runtime state.
