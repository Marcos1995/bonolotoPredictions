---
name: review
description: Review git diff for correctness, tests, secrets, and waste. Fix critical issues only.
disable-model-invocation: true
---

# Review

Review the current git diff against the default branch. Do not re-read the whole repo.

## Check

- Correctness vs the request; edge cases; error handling
- Tests exist for new behavior (or a single self-check)
- No secrets, injection, auth bypass, or unsafe eval
- No dead code, speculative features, or duplicate helpers

## Do

1. Look at the change set only: `git diff <base> -- . ':!graphify-out'` / `git log`
2. Fix Critical issues in place
3. Do not expand scope
4. Reply max 5 lines: `HECHO` or `FALLO` + what you found/fixed
