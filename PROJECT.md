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
- 2026-10-14: el criterio sigue siendo un boleto con los 6 calientes de los últimos 120 sorteos. Los tramos son meses naturales. En todo el periodo, el rinde es premios/gasto: 20 apuestas solo compensan si no rinden menos. La regularidad es la mediana mensual de ese rinde. La tabla tiene premios, gasto, saldo con el 5, sin 6 y sin 5.


