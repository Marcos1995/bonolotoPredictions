---
name: jev
description: >
  Build features that need programmable common sense (routing, ranking,
  extraction, scoring, verification) or replace an LLM prompt-and-parse step
  with a typed decision. Judgments run on local Laya, same primitives as
  TypeSafe's Jev (choice, score, noul). Do not call the TypeSafe API.
---

# Build judgments with Laya

Jev is TypeSafe's cloud System One model (paid API). This setup does not use it.
**Laya** is the local engine (Apache 2.0, 0 €, MCP `laya` already on every run).
Same job: state in, typed answers and probabilities out. No generated text.
Code owns the workflow.

Do not add `typesafe`, `typesafe_sdk`, `api.typesafe.ai`, or `TYPESAFE_API_KEY`.
[docs.typesafe.ai](https://docs.typesafe.ai/llms.txt) is only for how to shape questions
and which cookbook fits. The call is always Laya.

Skill `laya` is the agent's own stack decision (`docs/DECISIONES.md`).
This skill is for product code.

## Call

```python
from laya import Router

router = Router()
result = router.predict(state, questions, model="multilingual")
```

`state`: a string, or named JSON with the text the judgment needs.
`questions`: `{id: {type, instructions, criteria?}}`. Ids are for your code; put the meaning in `instructions`.

| Need | type | Read |
| --- | --- | --- |
| One of a defined set | `choice` | `choice`, `probabilities`, `answer_confidence` |
| Yes or no | `noul` | `noul` = P(yes). Near 0.5 is uncertain, not a medium score |
| Degree on ordered levels | `score` | `score` plus per-level probabilities |

```python
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": {
            "billing": "invoices, refunds",
            "technical": "bugs, outages",
            "other": "none of those",
        },
    },
    "urgent": {"type": "noul", "instructions": "Does this block the user right now?"},
    "severity": {
        "type": "score",
        "instructions": "How severe is the impact?",
        "criteria": ["cosmetic", "degraded", "blocked"],
    },
}
```

To try a judgment during a run, MCP `laya_predict` with `model: "multilingual"` and the same `state` and `questions`. Shipped code uses `Router`, not the MCP.

`model="multilingual"` covers Spanish and 100+ languages. Long text: `max_len=8192` (useful up to ~4000 tokens; check past that). More than ~20 choice options: narrow the list in code first, or MCP `laya_shortlist`.

## Design

Keep rules, math, lookups, and side effects in code. One narrow judgment per question. Add a no-match option when nothing may fit. Independent questions over the same state go in one `predict`. A second call only when an earlier answer changes the state or the options.

`answer_confidence` says how peaked the distribution is, not whether the app may act. Thresholds live in the app. Ignore uncertainty on branches you do not use.

When the shape is not obvious, read the closest cookbook on docs.typesafe.ai (routing, extraction, rerank, guardrails) and implement that decomposition with Laya.
