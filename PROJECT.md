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
- 2026-10-09: barrido del último año (365 sorteos, desde 2025-10-09), solo pasado. Calientes, fríos y retrasados; ventanas de 20 a 400 de 20 en 20; grupos 6, 8, 9, 10 y 12; 1, 5, 10 o 20 apuestas. Rentable solo con mediana positiva y al menos 14 de 20 ventanas en positivo. Resultado: ninguna de las 51 reglas. La menos mala es 1 apuesta de 6 calientes, mediana −150,50 €, 0/20. Hay un 6 real (12 calientes, ventana 400, 10 o 20 apuestas, 2026-05-05) y no sostiene la mediana. No predice el siguiente sorteo.


