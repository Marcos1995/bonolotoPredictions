---
name: laya
description: Non-trivial choice (stack, library, data model, architecture, scope) or "qué es mejor"/"decide". The local Laya model picks the option, not you. Logs one line in docs/DECISIONES.md.
---

# Laya

Laya is an open-source decision model running locally (MCP server `laya`, 0 €). You frame the question with facts; Laya chooses. Do not score the options yourself.

## Steps

1. Read `docs/DECISIONES.md` (if present). Reuse a past decision if it still applies; say so.
2. Frame one question. List 2-4 concrete options (always include "lo más simple que funcione"). For each, facts from the repo: extra cost, deps/files added, fit with `PROJECT.md`, how easy to undo.
3. Call the MCP tool `laya_predict` with `model: "multilingual"`:
   - `state`: `{"pregunta": "...", "contexto": "<stack>; prioridades: 0 € extra, simple, encaja, reversible", "opciones": {"<id>": "<facts>"}}`
   - `questions`: `{"mejor": {"type": "choice", "instructions": "¿Qué opción es mejor para este proyecto según el contexto?", "criteria": {"<id>": "<short label>"}}}`
4. Decision = `answers.mejor.choice`, probability `p` = `answers.mejor.answer_confidence`. Confidence: `p ≥ 0.6` alta, `0.4-0.6` media, `< 0.4` baja.
   - Override Laya only if its pick breaks a hard rule (paid API, impossible with the stack); say why.
   - `baja` → note what data would change the call.
   - Tool missing or failing → pick the most reversible option, confidence `baja`, and write `sin Laya`.
5. Append one line to `docs/DECISIONES.md` (create it with a `# Decisiones` header if missing):
   `YYYY-MM-DD · <decisión> · <por qué> · descartado: <opciones> · laya <p> · confianza: <alta|media|baja>`
6. Continue the task with the chosen option. Do not ask the user (headless).

## Reply (decision-only requests)

Max 4 lines:

```
DECISIÓN: <opción> (Laya <p>, confianza <alta|media|baja>)
Por qué: <1 línea>
Descartado: <opción — motivo>
Siguiente paso: <1 línea>
```
