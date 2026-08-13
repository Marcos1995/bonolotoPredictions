---
name: ponytail-review
description: >
  Review for over-engineering only. Lists what to delete/shrink (stdlib,
  native, yagni). Use for "review for bloat", "what can we delete", or
  /ponytail-review. Not for security/correctness review.
---

# Ponytail review

Hunt complexity. One line per finding.

Format: `file:L12-38: tag: what to cut. Replacement.`

Tags: `delete` | `stdlib` | `native` | `yagni` | `shrink`

End with `net: -N lines possible.` or `Lean already. Ship.`

Do not apply fixes here — only list them. Never flag the single smoke/assert check as bloat.
