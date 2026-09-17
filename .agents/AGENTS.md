# Developer Agent Guidelines: Le World Model in MLX Workspace

This document defines the behavior, documentation standards, and workflow automation rules for AI agents operating in the `lewm-mlx` repository.

---

## 📖 1. Documentation & Commenting Standards

All agents must adhere to the high standards of Google DeepMind when writing source code or documentation:

### Code Commenting (DeepMind Standard)
* **Docstrings**: Every module, class, and public function must have a clear Google-style docstring outlining its purpose, arguments, and return types.
* **Shape Annotations**: For all tensor manipulations (e.g., in `model.py`, `loss.py`, `dataset.py`), explicitly annotate intermediate shapes in comments (e.g., `# [B, H, W, C]` or `# [B, H*W+1]`).
* **Mathematical Rationale**: Document the underlying math equations (e.g., PUCT selections, MSE scaling, Tromp-Taylor scoring) inside the code comments using LaTeX math notation where applicable.
* **No Clutter**: Do not over-comment obvious lines of code. Focus on architectural design choices, layout changes (e.g., NHWC channels-last), and boundary conditions.

### Project Documentation
* **Academic Tone**: Write documentation using clear, evidence-based, and mathematically rigorous language. Avoid marketing speak. Focus on experimental setups, observations, and theoretical proofs.
* **Relative Links Only**: All links between documentation files must use relative paths (e.g., `[MLX Porting Walkthrough](../sessions/20260629/mlx_port.md)`). Never use absolute path schemes (`/Users/username/` or `file:///`).

---

## 🤖 2. Automated Chore & Findings Workflows

To reduce manual tracking chores and maintain a comprehensive history of the project's evolution, agents must execute the following automated logging tasks:

### Context Switch & Pivot Logging
* **When to Log**: Whenever a significant bug is discovered, a training run collapses, or the agent must pivot priority (e.g., switching from ensembling to value decoupling).
* **Where to Log**: Append a short summary to `docs/findings/chronological.md` or create a new session/walkthrough record describing:
  1. The anomaly or priority shift.
  2. The scientific hypothesis and proposed fix.
  3. The outcome of the change.
* **Commit Integration**: Ensure commit messages act as summaries, but are backed by these detailed logs inside the repository.

### Q&A Log Rule
Whenever the user says *"wow good to know, can you save it to QnA?"* (or similar requests to save a key insight/concept to the QnA folder):
1. **Identify the Concept**: Extract the core concept and technical explanation from the recent context.
2. **File Location**: Save inside the `docs/qna/` directory.
3. **Naming Convention**: Use lowercase kebab-case naming (e.g., `docs/qna/replay-buffer-sampling.md`).
4. **Layout**:
   ```markdown
   # [Descriptive Question/Title]

   ## Context
   [Brief context of when/why this question arose during training/development]

   ## Answer
   [Detailed technical explanation, including code snippets, math formulas, and architectural design choices]
   ```

---

## 🔒 3. Safe Execution Constraints & Environment Setup

* **Automatic Approvals**: Proposals containing `[WIP]` or `WIP` tags in their titles must never be executed automatically. The agent must wait for human feedback.
* **Timer Safety**: Never use shell-based `sleep` tasks for background polling; use the system `schedule` timer tool to manage sleep cycles asynchronously.
* **Python Environment**: Use `uv run` for executing python scripts and tests. With an editable install (`uv pip install -e .`), `PYTHONPATH=.` is not required in the project root workspace.

---

## 🔌 4. Superpowers Skills Adaptation Policy

The Superpowers plugin expects design specs and plans at `docs/superpowers/specs/` and `docs/superpowers/plans/`, whereas the repository's canonical context structure mandates designs in `docs/design/` and plans in `docs/plans/`.

To maintain a single source of truth without duplicating documents:
1. **Canonical Store**: All design specifications live in `docs/design/` and all implementation plans live in `docs/plans/`.
2. **Git Ignored**: `docs/superpowers/` is git-ignored and must never be tracked in the git index.
3. **Session-Scoped Symlinks**: While a Superpowers skill (`brainstorming`, `writing-plans`, `subagent-driven-development`) is actively running and needs its expected paths, create relative symlinks from `docs/superpowers/` to the canonical stores:
   - `docs/superpowers/specs/<YYYY-MM-DD-topic-design>.md` $\to$ `../../design/<topic>.md`
   - `docs/superpowers/plans/<YYYY-MM-DD-topic>.md` $\to$ `../../plans/<YYYY-MM-DD-topic>.md`
   Symlink names must satisfy Superpowers naming conventions (e.g., prefixed with `YYYY-MM-DD`). The canonical documents stay in `docs/design/` and `docs/plans/`; the symlinks are scaffolding for the skill session only.
4. **Remove When the Session Ends**: Delete every symlink created for a skill session as soon as that session finishes, and before handing the checkout to another agent or committing. A repository that carries the aliases at rest cannot pass `agents drift`, which reports `docs/superpowers/specs/<YYYY-MM-DD-topic-design>.md` as a misplaced design document.
