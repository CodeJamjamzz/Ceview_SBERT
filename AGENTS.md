# AGENTS.md

Welcome. When working in this repository, follow these rules:

## Read Order
At the start of every session, read the context files in this exact order:
1. `AGENTS.md` - This file.
2. `BLOCKERS.md` - What not to do / what is out of scope.
3. `docs/model.md` - Understand what the model is and its contract.
4. `docs/integration.md` - How the backend calls this model.
5. `docs/training.md` - How this model is trained.
6. `experiments/latest.json` - What happened during the last training run.
7. `tasks/today.md` - What you need to do today.

## Hard Rules
- **No training in the IDE**: Training is done externally via Google Colab. Code changes happen here, but execution of heavy training tasks should be assumed to happen in Colab.
- **Never retrain on the test set**.
- **Confidence ≠ Correctness**: Do not assume high confidence means the model is correct.
- **Keep it lean**: Do not add unnecessary directories (like `api/`, `architecture/`) unless explicitly requested.
- **Task-based Commits**: Only commit a completed task from `tasks/today.md` if the user explicitly adds the text `(commit)` to the task description. When a commit is required, ensure the commit message strictly follows the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`).
