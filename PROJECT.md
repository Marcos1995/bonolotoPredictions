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
- El análisis público es Bonoloto 6/49. Las reglas no miran el futuro. Laya eligió abrir con los calientes; el extra hay que decirlo en aciertos, no como premio.
- No tocar `predictions.sqlite` ni subirlo.
- Ponytail siempre activo (ver AGENTS.md)

## Estado
- 2026-10-07: página con histórico, pares/impares, seguidos, frecuencias y backtest de seis reglas contra el azar. El JSON lo regenera `.github/workflows/analisis.yml`.


