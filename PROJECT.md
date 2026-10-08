<!-- managed-by-telegram-cursor-bot:agent-kit -->
# Contexto del proyecto

## Produccion
- URL: https://Marcos1995.github.io/bonolotoPredictions/
- Deploy: GitHub Pages desde `main` en `/`

## Stack
- Python 3.11, pandas, sqlite local (no se publica). Página estática: `index.html` + `data/bonoloto.json`.

## Comandos utiles
- Instalar: `py -3.11 -m pip install pandas`
- Test: `py -3.11 analyze_web.py`
- Dev: abrir `index.html` (o `py -3.11 -m http.server`)

## Notas para el agente
- El análisis público es Bonoloto 6/49. Cada boleto se juzga solo con sorteos anteriores. Lo fiable es la forma (3 y 3), no un número caliente o frío.
- No tocar `predictions.sqlite` ni subirlo.
- Ponytail siempre activo (ver AGENTS.md)

## Estado
- 2026-10-09: los calientes se parten en grupos de 3 y se juega cada pareja. 9 números son 3 apuestas (1,50 €) y 12 son 6 (3 €). En la página, las tres ventanas que más cobran.


