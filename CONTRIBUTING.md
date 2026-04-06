# Contributing to VoiceBridge

Thanks for helping improve VoiceBridge. This document describes how to set up your environment, run checks locally, and propose changes.

## Before you start

1. Follow **[INSTALL.md](INSTALL.md)** to clone the repo, run the backend, and run the frontend.
2. **Do not commit secrets.** Keep API keys and tokens in local `src/.env` (and related env files) only; those files must stay out of version control.

---

## Branching and pull requests

- **Base branch:** `main`
- **Workflow:** Create a feature branch from `main`, push it, and open a pull request into `main`.
- **PR description:** Summarize what changed and why, note any new dependencies or config, and mention how you tested (manual steps or automated tests).

Use a short, descriptive branch name, for example:

- `fix/session-state-race`
- `feat/websocket-reconnect`

---

## Project layout (high level)

| Area | Path |
|------|------|
| Backend (FastAPI, core logic) | `src/` |
| React UI / extension | `src/presentation/` |
| Automated tests | `test/` |
| Documentation | `docs/` |

When editing Python under `src/`, imports assume `src` is on **`PYTHONPATH`** (see CI and pytest below).

---

## Python: lint and tests (match CI)

GitHub Actions runs on pushes and pull requests to `main` when Python files or `src/requirements.txt` change. Locally, mirror that workflow from the **repository root** (`voicebridge/`, not only `src/`).

### Install tools

Use the same virtual environment you use for the app, then from repo root:

```bash
pip install pylint pytest
pip install -r src/requirements.txt
```

### Lint (pylint)

From the repo root (same as CI, best in Git Bash on Windows):

```bash
pylint **/*.py --fail-under=8.0
```

If your shell does not expand `**`, use explicit paths instead, for example:

```bash
pylint src test --fail-under=8.0
```

CI currently allows the lint step to report without failing the job; still aim to keep new code clean and fix easy warnings.

### Tests (pytest)

`PYTHONPATH` must include `src` so modules resolve the same way as in CI:

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "src"
pytest --rootdir=.
```

**macOS / Linux:**

```bash
PYTHONPATH=src pytest --rootdir=.
```

Add or update tests under `test/` when you change behavior users or other modules rely on.

---

## Frontend (React)

From `src/presentation/`:

```bash
npm install
npm test
npm run build
```

Use `npm start` during development. See `src/presentation/README.md` for the extension build flow.

---

## Style and review expectations

- Prefer **small, focused** changes with clear commit messages.
- **Match existing patterns** in the file you touch (naming, error handling, logging).
- If you introduce a new dependency, update `src/requirements.txt` or `src/presentation/package.json` and mention it in the PR.

---

## Getting help

- Feel free to open up any issues or email one of the original team members!
