## Description

<!-- Provide a clear and concise description of the changes in this PR. -->

## Type of Change

<!-- Mark the relevant option with an `x`. -->

- [ ] 🧪 New experiment or adapter
- [ ] ✨ New feature (non-adapter)
- [ ] 🐛 Bug fix
- [ ] 📝 Documentation update
- [ ] 🔧 Refactoring
- [ ] ⚙️ Config or build system change

## What's Changed

<!-- List the key changes made in this PR. -->

-

## Testing

<!-- Describe how you tested your changes. -->

- [ ] Code runs without import errors
- [ ] Training/eval script tested locally
- [ ] Results are reproducible (if applicable)

**Test command used:**

```bash
# e.g., python src/scripts/glue/finetune_roberta_glue.py --adapter_type gpart --tasks sst2
```

## Checklist

- [ ] My branch is up to date with `upstream/main`
- [ ] I've followed the existing code style and patterns
- [ ] New adapter configs are registered in `src/configs/adapter_configs/__init__.py` (if applicable)
- [ ] New tuner implementations follow the existing pattern in `src/peft/tuners/` (if applicable)
- [ ] I've documented any new configuration options or adapter types
- [ ] No large data files or model checkpoints are included

## Related Issues

<!-- Link any related issues here, e.g., "Closes #12" or "Related to #5". -->

## Additional Notes

<!-- Any other context, screenshots, or information that would be helpful for the reviewer. -->
