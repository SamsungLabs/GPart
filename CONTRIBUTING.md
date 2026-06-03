# Contributing to GPart

Thank you for your interest in contributing to GPart.

This repository uses a **fork-based workflow** for external contributions. Contributors work in their own fork, create topic branches for each change, and open pull requests to the canonical repository. This helps keep `main` stable and ensures all changes are reviewed before merging. 

---

## Workflow Overview

The contribution flow is:

```text
Canonical repo (protected main)
        ^
        |  Pull Request
        |
Your fork -> feature/experiment branch
```

In practice:

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Add the canonical repository as `upstream`.
4. Create a new branch from the latest `upstream/main`.
5. Make your changes and commit them.
6. Push the branch to your fork.
7. Open a pull request to the canonical repository.

---

## Getting Started

### 1. Fork the repository

Use the **Fork** button on GitHub to create your own copy of the repository under your account.

### 2. Clone your fork

```bash
git clone https://github.com/<YOUR_USERNAME>/GPart.git
cd GPart
```

### 3. Add the canonical repository as `upstream`

This allows you to keep your fork synchronized with the main repository.

```bash
git remote add upstream https://github.com/SamsungLabs/GPart.git
git remote -v
```

You should see something like:

```bash
origin    https://github.com/<YOUR_USERNAME>/GPart.git (fetch)
origin    https://github.com/<YOUR_USERNAME>/GPart.git (push)
upstream  https://github.com/SamsungLabs/GPart.git (fetch)
upstream  https://github.com/SamsungLabs/GPart.git (push)
```

GitHub recommends configuring an `upstream` remote for fork-based work so your local clone and fork can be synced with the original repository.

### 4. Install dependencies

```bash
uv sync
source .venv/bin/activate
```
---

## Branches

Always create a new branch for each contribution. Do not work directly on `main`.

### Branch naming

Use short, descriptive branch names with one of the following prefixes:

| Prefix        | Use case                                              | Example                          |
| ------------- | ----------------------------------------------------- | -------------------------------- |
| `feature/`    | New functionality                                     | `feature/new-gpart-variant`      |
| `experiment/` | Experimental or exploratory work                      | `experiment/sst2-ablation`       |
| `fix/`        | Bug fix                                               | `fix/dataloader-shuffle`         |
| `docs/`       | Documentation only                                    | `docs/update-contributing-guide` |
| `refactor/`   | Internal refactoring with no intended behavior change | `refactor/config-registry`       |

### Create a branch

Create your branch from the latest `upstream/main`:

```bash
git fetch upstream
git checkout -b experiment/my-new-experiment upstream/main
```

This keeps your work based on the current canonical history and avoids unnecessary merge commits in your local `main`. Syncing and branching from the upstream default branch is a standard fork-based workflow.

---

## Making Changes

Keep changes focused and easy to review.

### General guidelines

- Make one logical change per pull request.
- Avoid mixing refactors, formatting changes, and new functionality in the same PR.
- Update documentation when you add or change behavior.
- Follow the existing project structure and naming patterns.
- Do not commit large artifacts, temporary outputs, checkpoints, or private data unless the repository explicitly expects them.

### Commit messages

Use clear and descriptive commit messages. A simple conventional style works well:

```text
feat: add config for new gpart adapter
fix: correct dataset split handling
docs: clarify fork sync workflow
refactor: simplify tuner registry initialization
```

If the change is substantial, add a short body explaining the motivation or major details.

---

## Syncing Your Fork

Keep your fork and local branches up to date with the canonical repository.

### Update your local `main`

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

GitHub documents syncing a fork by incorporating changes from the upstream default branch into your local and forked repository. [web:42]

### Update your working branch

```bash
git checkout experiment/my-new-experiment
git fetch upstream
git merge upstream/main
```

If your branch is already part of an open pull request, merging `upstream/main` is usually the safest option because it avoids rewriting branch history. GitHub’s fork workflow is centered on branch-based pull requests, and preserving shared branch history can reduce confusion during review.

---

## Pull Requests

All changes must be submitted through a pull request to the canonical repository.

### Before opening a PR

Please make sure that:

- [ ] Your branch is based on the latest `upstream/main`.
- [ ] Your code runs locally without errors.
- [ ] You formatted your code with [Black](https://github.com/psf/black) (`black .`).
- [ ] You tested your changes.
- [ ] You updated documentation for any new behavior, flags, configs, or APIs.
- [ ] You did not include unrelated changes.
- [ ] You checked whether a similar PR or issue already exists.

If the repository has automated checks, make sure they pass before requesting review. Repositories with contributor guidance, PR templates, and CI-supported workflows are generally easier to maintain and review.

### Open the PR

1. Push your branch to your fork:

```bash
git push origin experiment/my-new-experiment
```

2. Open a pull request from your fork to the canonical repository.
3. Target the `main` branch unless the repository specifies otherwise.
4. Fill in the PR template completely.
5. Write a clear title and description explaining:
   - what changed,
   - why the change is needed,
   - how it was tested,
   - any limitations or follow-up work.

### PR scope

Please keep PRs small and well scoped. Research on GitHub workflows has found that well-scoped and clearly described work is associated with a higher likelihood of successful integration. [web:30]

### During review

- Respond to review comments directly in the PR.
- Push follow-up commits to the same branch.
- Keep discussion in the PR so the review history stays visible.
- Do not open a new PR for the same change unless requested.

### After merge

Once your PR is merged, update your fork:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

You can then delete your feature branch locally and on GitHub if you no longer need it.

---

## Adding a New Adapter or Experiment

GPart is a research codebase, so contributions should be easy to understand and reproduce.

When adding a new adapter, tuner, or experiment:

1. Add configuration files in the appropriate config directory.
2. Register new components where required by the codebase.
3. Add or update training and evaluation scripts if needed.
4. Document all new options, expected inputs, and usage.
5. Include a minimal example command when possible.

Please also include enough detail for others to run or inspect the contribution:

- dataset or benchmark used,
- relevant config name,
- seed or reproducibility notes if applicable,
- expected output files or logs,
- whether results are sanity checks or full experiments.

For project-specific structure, follow the relevant sections in `README.md`.

---

## Issues and Proposals

For small fixes and documentation updates, you can usually open a pull request directly.

For larger contributions, such as:
- a new method,
- a major refactor,
- a broad experimental framework change,
- a change that affects public APIs or repository structure,

please open an issue or discussion first. This helps align the proposal with project goals and can save time for both contributors and maintainers. Clear contribution guidance and issue/PR templates are commonly recommended elements of healthy open-source repositories.

---

## Code Style and Repository Hygiene

Please keep the repository clean:

- Do not commit secrets, tokens, or credentials.
- Do not commit generated files unless they are intentional project artifacts.
- Avoid unnecessary file renames in the same PR as functional changes.
- Add comments to improve clarity.

### Code Formatting

This project uses [Black](https://github.com/psf/black) as the code formatter. **Always format your code with Black before opening a pull request.**

```bash
# Format all Python files
black .

# Format specific files
black src/path/to/file.py
```

If the repository defines formatting, linting, or testing commands, run them before opening a PR.

---

## Questions

If something is unclear, open an issue before starting work.

Thank you for contributing to GPart.