# Repository Guidelines

## Project Structure & Module Organization

- `peft/` contains the editable local PEFT implementation. GPart's core lives in `peft/tuners/gpart/` (`config.py`, `layer.py`, `model.py`, and `fastfood.py`).
- `src/configs/` defines dataclass-based experiment and adapter configuration. Add public GPart options to `src/configs/adapter_configs/gpart.py` and forward them in `src/utils/adapter_utils.py`.
- `src/scripts/` contains task entry points: `glue/`, `vision/`, and `math/`. Shared training behavior is in `src/utils/`.
- `src/tests/` holds pytest tests. Put focused GPart behavior tests in files named `test_gpart_<feature>.py`.
- `assets/` contains README media. Treat `data/`, `experiments/`, and `mlruns/` as local/generated data; do not commit outputs unless explicitly intended.

## Build, Test, and Development Commands

Use Python 3.12 and the repository-managed environment:

```bash
uv sync                         # create/update the local environment
uv run pytest -q                # run the full test suite
uv run pytest -q src/tests/test_gpart_fastfood.py
uv run python src/scripts/glue/eval_roberta_glue.py --help
```

Use targeted tests while iterating, then run the full suite before a pull request. Tests marked `slow` can be excluded with `uv run pytest -m "not slow"`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints for helpers, and concise docstrings for public APIs. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and descriptive test names such as `test_save_reload_runtime_state`. Keep GPart deterministic: configure seeds explicitly and avoid consuming global RNG state. `black` and `isort` are available through the `dev` extra; format only files you changed.

## Testing Guidelines

Add regression coverage for changes to projection math, checkpoint loading, adapter lifecycle, or config validation. Prefer small CPU-only models and assert deterministic values, shapes, gradients, save/reload behavior, and invalid-config errors where relevant. Run `git diff --check` before committing.

## Open-Source Branch Policy

For `open-source`, backport reusable library behavior, public scripts, documentation, and tests. Exclude Slurm files, cluster launchers, campaign scripts, internal profilers, MLflow data, result CSVs, rebuttals, planning notes, unpublished PDFs, and confidential assets unless explicitly approved. When syncing from `main`, use a dedicated branch and focused commits; never merge `main` wholesale. Preserve existing open-source deletions when source commits conflict with removed internal files.

`open-source` is the source of truth for releases to `SamsungLabs/GPart`. Publish it through a dedicated integration branch and a public-repository PR: fetch the public default branch, inspect its ancestry and full diff, run `uv run pytest -q`, and review for restricted paths before creating the PR. Do not copy files manually or push directly to public `main`; an initial history-alignment force push requires explicit approval.

## Commit & Pull Request Guidelines

Use imperative Conventional Commit-style subjects, for example `feat(gpart): add Fastfood projection` or `fix(vision): persist classifier`. Keep commits focused; avoid bundling results, Slurm campaigns, or rebuttal artifacts with public library changes. PR descriptions should state scope, excluded artifacts, tests run, and any model/dataset requirements. Include CLI examples when adding a user-facing script or configuration option.
