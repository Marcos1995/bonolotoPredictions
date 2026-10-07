---
name: debug
description: Use on any bug, failing test, build error or unexpected behavior, before proposing a fix. Root cause first, one change at a time.
---

# Debug

Condensed from obra/superpowers `systematic-debugging`. Iron law: no fix without the root cause.

## 1. Root cause

- Read the full error and stack trace: file, line, code. It often says the answer.
- Reproduce it reliably with the exact steps. Not reproducible: gather data, don't guess.
- Check what changed: `git diff`, `git log -5`, deps, config, env.
- Several components (bot → SDK → repo → GitHub): log what enters and exits each boundary once, find where it breaks, then dig only there.
- Trace the bad value back to where it originates; fix there, not at the symptom.

## 2. Compare

- Find similar code in the repo that works and list every difference, however small.
- Using a library or pattern: read its official docs for the installed version fully before applying it.

## 3. Hypothesis

- Write one: "X is the cause because Y". Test it with the smallest possible change, one variable at a time.
- Wrong: form a new hypothesis. Never stack fixes on top of a failed one.

## 4. Fix

- Add a failing check first (test or one-off script) that reproduces the bug.
- One fix for the root cause. No "while I'm here" refactors.
- Verify: the check passes, nothing else broke, the original symptom is gone.
- 3 failed fixes = design problem, not bad luck. Stop, reply `FALLO` with the evidence and the design question; don't try a 4th.

## Red flags → back to step 1

"Quick fix for now", "let's try X and see", several changes at once, fixing without a repro, "probably X".
