---
name: fix-bug
description: >
  Root-cause bug fixing. Use when the user reports an error, broken
  behavior, stack trace, or screenshot of a failure.
---

# Fix bug

1. Reproduce from the report (logs, stack, screenshot notes).
2. Trace the real flow; grep all callers of the failing function.
3. Fix the shared root cause once (Ponytail ladder).
4. Add ONE small runnable check if the logic is non-trivial.
5. Ship (commit + push) unless told not to.
6. Reply: cause → fix → commit.
