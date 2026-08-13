---
name: ship
description: >
  Commit and push finished work. Use when the user asks to ship, commit,
  push, or after completing an implementation task that should land on the
  remote.
---

# Ship

1. `git status` / `git diff` — confirm intentional changes only (no secrets).
2. `git add -A` (or only relevant paths).
3. Commit with a short descriptive message (why, not essay).
4. `git push -u origin HEAD` (or default branch remote).
5. Report: commit hash + remote ref. If push fails, say why briefly.
