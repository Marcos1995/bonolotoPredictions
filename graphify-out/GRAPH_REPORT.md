# Graph Report - bonolotoPredictions  (2026-11-15)

## Corpus Check
- 35 files · ~116,159 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 3, .mdc 2, .sqlite 1)

## Summary
- 429 nodes · 648 edges · 33 communities (25 shown, 8 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6e048c92`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- edgeHunt.py
- commonFunctions.py
- webScraper.py
- LotteryPatternAnalyzer
- PatternOptimizer
- MultiStrategyPredictor
- analyze_web.py
- MaximumHitsPredictor
- OverdueBacktester
- tablas/SKILL.md
- db
- LotteryBacktester
- Debug
- predictData
- Contexto del proyecto
- Dashboard brief (for the Stitch prompt)
- Verify (UI)
- Web design
- Agent rules
- Laya
- Review
- bonolotoPredictions
- DESIGN.md
- DECISIONES.md
- comb
- load
- _topk
- cargar_premios
- shape_check
- build

## God Nodes (most connected - your core abstractions)
1. `LotteryPatternAnalyzer` - 16 edges
2. `MultiStrategyPredictor` - 15 edges
3. `barrido()` - 14 edges
4. `PatternOptimizer` - 14 edges
5. `build()` - 13 edges
6. `MaximumHitsPredictor` - 12 edges
7. `BonolotoScraper` - 12 edges
8. `run_series()` - 11 edges
9. `hunt()` - 9 edges
10. `OverdueBacktester` - 9 edges

## Surprising Connections (you probably didn't know these)
- `load()` --calls--> `load_rows()`  [EXTRACTED]
  analyze_web.py → raffles.py
- `noche()` --calls--> `nextBonolotoDate()`  [EXTRACTED]
  analyze_web.py → commonFunctions.py
- `build()` --calls--> `nextBonolotoDate()`  [EXTRACTED]
  analyze_web.py → commonFunctions.py
- `generate_best_prediction()` --calls--> `nextBonolotoDate()`  [EXTRACTED]
  bestPrediction.py → commonFunctions.py
- `generate_best_prediction()` --calls--> `db`  [EXTRACTED]
  bestPrediction.py → sqliteClass.py

## Import Cycles
- None detected.

## Communities (33 total, 8 thin omitted)

### Community 0 - "edgeHunt.py"
Cohesion: 0.05
Nodes (49): nextDrawDate(), Next date after last_date whose weekday is in weekdays (Mon=0)., Build judgments with Laya, Call, Design, borda(), _ensemble_members(), expected_hits() (+41 more)

### Community 1 - "commonFunctions.py"
Cohesion: 0.11
Nodes (20): Best Prediction Generator ========================= Comprehensive analysis…, collections, colorama, Pick k unique ints from a value_counts-like Series, then fill low..high., topUniqueNumbers(), datetime, Maximum Hits Predictor ====================== Predicts lottery numbers by…, Multi-Strategy Lottery Predictor ================================ Implements… (+12 more)

### Community 2 - "webScraper.py"
Cohesion: 0.07
Nodes (25): re, selenium, selenium_common_exceptions, selenium_webdriver_chrome_options, selenium_webdriver_chrome_service, selenium_webdriver_common_by, selenium_webdriver_edge_options, selenium_webdriver_edge_service (+17 more)

### Community 3 - "LotteryPatternAnalyzer"
Cohesion: 0.08
Nodes (15): LotteryPatternAnalyzer, Analyze lottery patterns using statistical methods. This class provides…, Find the most common pairs of numbers drawn together., Analyze how often consecutive numbers appear in the same draw., Analyze the distribution of numbers (low vs high, odd vs even)., Analyze number frequency across ALL positions (N1-N6 combined). Since numbers…, Test if the lottery is truly random by comparing actual vs expected…, Analyze the frequency of each number drawn. Shows hot numbers (most frequent)… (+7 more)

### Community 4 - "PatternOptimizer"
Cohesion: 0.09
Nodes (18): hashlib, itertools, json, main(), PatternOptimizer, DataFrame, Pattern-Based Strategy Optimizer ================================ Recursively…, Save a result to the cache. (+10 more)

### Community 5 - "MultiStrategyPredictor"
Cohesion: 0.08
Nodes (14): main(), MultiStrategyPredictor, Combine top 3 hot numbers with top 3 cold numbers., Select numbers ensuring balanced range and odd/even distribution., Weight number appearances with exponential decay by recency., Tests multiple lottery prediction strategies and compares their performance., Pick numbers that are overdue compared to their average gap., Get actual lottery numbers for a specific date. (+6 more)

### Community 6 - "analyze_web.py"
Cohesion: 0.12
Nodes (21): barrido(), _cobertura(), _coger(), _euros(), _euros_de(), _indices(), _jugar(), _listas() (+13 more)

### Community 7 - "MaximumHitsPredictor"
Cohesion: 0.14
Nodes (12): main(), MaximumHitsPredictor, DataFrame, Generate predictions using pattern-based strategy with given parameters., Get actual lottery numbers for a specific date., Generate prediction by combining votes from top parameter configurations.…, Run backtest against historical data, showing 3+ hit results., Leverages cached optimizer results to generate predictions that maximize the… (+4 more)

### Community 8 - "OverdueBacktester"
Cohesion: 0.16
Nodes (9): main(), OverdueBacktester, Get the actual numbers drawn on a specific date. Args: date: The draw date…, Run backtesting for the last N days of actual results. Args: days_to_test:…, Backtest lottery predictions using the "longest since appeared" strategy.…, Generate and print a comprehensive summary of backtest results., Predict numbers for the next draw using the best performing strategy., Main entry point for the backtest. (+1 more)

### Community 10 - "db"
Cohesion: 0.31
Nodes (5): generate_best_prediction(), Generate the best possible prediction using comprehensive analysis., nextBonolotoDate(), Bonoloto draws every day, Sunday included., db

### Community 11 - "LotteryBacktester"
Cohesion: 0.24
Nodes (5): LotteryBacktester, Backtest lottery predictions using historical data and forecast future results.…, Run backtesting on the last N draws., Predict numbers for the next upcoming draw using ALL available data., Generate prediction scores for all numbers (1-49) based ONLY on data available…

### Community 12 - "Debug"
Cohesion: 0.29
Nodes (6): 1. Root cause, 2. Compare, 3. Hypothesis, 4. Fix, Debug, Red flags → back to step 1

### Community 14 - "Contexto del proyecto"
Cohesion: 0.29
Nodes (6): Comandos utiles, Contexto del proyecto, Estado, Notas para el agente, Produccion, Stack

### Community 15 - "Dashboard brief (for the Stitch prompt)"
Cohesion: 0.33
Nodes (5): Anatomy (always), Charts without libraries (inline SVG, no CDN), CSS, Dashboard brief (for the Stitch prompt), Data honesty (non-negotiable)

### Community 16 - "Verify (UI)"
Cohesion: 0.40
Nodes (4): 1. Screenshots, 2. Look, 3. Fix and repeat, Verify (UI)

### Community 17 - "Web design"
Cohesion: 0.29
Nodes (6): Before HECHO, Steps (only case 1 or case 2), Style = `DESIGN.md`, Web design, When Stitch does not run, When Stitch runs

### Community 18 - "Agent rules"
Cohesion: 0.50
Nodes (3): Agent rules, Flujo, Think → Simple → Surgical → Verify (Karpathy)

### Community 19 - "Laya"
Cohesion: 0.50
Nodes (3): Laya, Reply (decision-only requests), Steps

### Community 20 - "Review"
Cohesion: 0.50
Nodes (3): Check, Do, Review

### Community 26 - "comb"
Cohesion: 0.20
Nodes (12): comb(), consec_stats(), _count_z(), decade_rows(), ending_rows(), freq_table(), odd_even(), patrones() (+4 more)

### Community 27 - "load"
Cohesion: 0.20
Nodes (12): _comp(), dias_sin_datos(), fetch_dia(), fetch_live(), _html(), load(), parse_live(), 6 bolas del tablero en vivo, o None si el sorteo sigue en '?'. (+4 more)

### Community 28 - "_topk"
Cohesion: 0.21
Nodes (13): backtest(), _balance(), _huecos(), next_ticket(), noche(), picks_now(), _pool(), El criterio: un boleto con los 6 calientes de los últimos 120 sorteos. (+5 more)

### Community 29 - "cargar_premios"
Cohesion: 0.33
Nodes (6): cargar_premios(), uno(), _euro_es(), parse_premios(), Premio unitario publicado ese día. Si el 6 no tuvo acertantes, el 6 es el bote., Premios reales por fecha. Cache en data/premios.json; solo baja los que faltan.

### Community 30 - "shape_check"
Cohesion: 0.40
Nodes (4): La forma (cuántos impares, hueco, amplitud) se juzga con el pasado solo., Boletos armados solo con sorteos anteriores al último, contra ese último., _score_past(), shape_check()

### Community 32 - "build"
Cohesion: 0.16
Nodes (15): _armar(), _boletos(), build(), _buscar(), _claro(), _detalle_boleto(), _fila_clara(), _forma() (+7 more)

## Knowledge Gaps
- **33 isolated node(s):** `1. Root cause`, `2. Compare`, `3. Hypothesis`, `4. Fix`, `Red flags → back to step 1` (+28 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 212 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LotteryPatternAnalyzer` connect `LotteryPatternAnalyzer` to `commonFunctions.py`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `MultiStrategyPredictor` connect `MultiStrategyPredictor` to `commonFunctions.py`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **What connects `1. Root cause`, `2. Compare`, `3. Hypothesis` to the rest of the system?**
  _33 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `edgeHunt.py` be split into smaller, more focused modules?**
  _Cohesion score 0.053939714436805924 - nodes in this community are weakly interconnected._
- **Should `commonFunctions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1126984126984127 - nodes in this community are weakly interconnected._
- **Should `webScraper.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07058823529411765 - nodes in this community are weakly interconnected._
- **Should `LotteryPatternAnalyzer` be split into smaller, more focused modules?**
  _Cohesion score 0.08064516129032258 - nodes in this community are weakly interconnected._