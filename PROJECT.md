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
- 2026-10-09: las tablas se filtran por caliente, frío o retrasado, por sorteos mirados y por apuestas. Cada plazo, de 3 meses a 5 años, se parte en tramos y se ve qué combinación gana en cada uno.


