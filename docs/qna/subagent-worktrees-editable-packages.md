# Why Do Editable Python Installs Behave Differently Inside Subagent Git Worktrees?

## Context
When adopting subagent-driven development with the `using-git-worktrees` skill, subagents operate within isolated working trees (e.g. `.system_generated/worktrees/subagent-Implement-Task-1-...`). During test and CLI execution, subagents frequently had to prefix commands with `PYTHONPATH=.` or experienced issues where newly created modules were not found by Python, whereas in the primary root repository `uv run` functioned seamlessly without `PYTHONPATH`.

## Answer

### 1. Mechanism of Editable Package Installations
When a package is installed in editable mode using `uv pip install -e .` (or `pip install -e .`), the build backend creates an editable `.pth` file or direct link in the virtual environment's `site-packages` directory:

```
.venv/lib/python3.12/site-packages/__editable___lewm_mlx_0_1_0_finder.py
# points to: <repository root>
```

This absolute filesystem pointer informs the Python import system to resolve any imports of `lewm_mlx.*` by looking directly inside the repository root where the editable install was created.

### 2. The Git Worktree Discrepancy
When a subagent creates an isolated git worktree:
1. The worktree creates a separate directory checkout at:
   `.system_generated/worktrees/<task-branch>/`
2. The subagent continues to use the existing parent virtual environment (`.venv`).
3. If the subagent creates a new module (e.g., `lewm_mlx/dataset.py`) inside its worktree, executing `uv run pytest` causes Python to search `site-packages`, following the `.pth` pointer back to the **primary root directory**, *not* the worktree directory.
4. As a result, Python either reports `ModuleNotFoundError: No module named 'lewm_mlx.dataset'` (because the file does not exist yet in the parent directory) or imports stale code from the parent branch instead of the worktree's uncommitted changes.

### 3. Resolution Strategies

| Approach | Where Used | Behavior |
|---|---|---|
| **`PYTHONPATH=.`** | Isolated Subagent Worktrees | Injects the worktree's local directory at the head of `sys.path` (index 0), overriding `site-packages` and the parent `.pth` link. |
| **`uv run` (Native)** | Primary Root Workspace | Fully resolved by the `.pth` link directly. `PYTHONPATH=.` is completely redundant and should be omitted. |
| **Worktree Editable Re-install** | Heavy Multi-Package Trees | Executing `uv pip install -e .` inside the worktree updates the `.pth` pointer to the worktree, but modifies the shared virtual environment. |

### Best Practice for Agentic Orchestration
When dispatching subagents to git worktrees, prompt instructions should explicitly advise subagents to run tests with `PYTHONPATH=. uv run ...`. Once code is merged back to `main` in the primary workspace, documentation and user commands should omit `PYTHONPATH=.` to maintain clean, idiomatic execution standards.
