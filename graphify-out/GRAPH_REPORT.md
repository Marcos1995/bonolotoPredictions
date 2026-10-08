# Graph Report - bonolotoPredictions  (2026-10-09)

## Corpus Check
- 33 files · ~85,053 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .mdc 2, (none) 2, .sqlite 1)

## Summary
- 390 nodes · 586 edges · 25 communities (19 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `42ca7c52`
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

## God Nodes (most connected - your core abstractions)
1. `build()` - 18 edges
2. `LotteryPatternAnalyzer` - 16 edges
3. `MultiStrategyPredictor` - 15 edges
4. `PatternOptimizer` - 14 edges
5. `MaximumHitsPredictor` - 12 edges
6. `BonolotoScraper` - 12 edges
7. `run_series()` - 11 edges
8. `hunt()` - 9 edges
9. `OverdueBacktester` - 9 edges
10. `State` - 8 edges

## Surprising Connections (you probably didn't know these)
- `load()` --calls--> `load_rows()`  [EXTRACTED]
  analyze_web.py → raffles.py
- `build()` --calls--> `nextBonolotoDate()`  [EXTRACTED]
  analyze_web.py → commonFunctions.py
- `run_series()` --calls--> `nextDrawDate()`  [EXTRACTED]
  edgeHunt.py → commonFunctions.py
- `generate_best_prediction()` --calls--> `nextBonolotoDate()`  [EXTRACTED]
  bestPrediction.py → commonFunctions.py
- `generate_best_prediction()` --calls--> `db`  [EXTRACTED]
  bestPrediction.py → sqliteClass.py

## Import Cycles
- None detected.

## Communities (25 total, 6 thin omitted)

### Community 0 - "edgeHunt.py"
Cohesion: 0.06
Nodes (46): Build judgments with Laya, Call, Design, borda(), _ensemble_members(), expected_hits(), _fill_from_groups(), _fz() (+38 more)

### Community 1 - "commonFunctions.py"
Cohesion: 0.10
Nodes (23): Best Prediction Generator ========================= Comprehensive analysis…, collections, colorama, Pick k unique ints from a value_counts-like Series, then fill low..high., topUniqueNumbers(), datetime, hashlib, itertools (+15 more)

### Community 2 - "webScraper.py"
Cohesion: 0.07
Nodes (25): re, selenium, selenium_common_exceptions, selenium_webdriver_chrome_options, selenium_webdriver_chrome_service, selenium_webdriver_common_by, selenium_webdriver_edge_options, selenium_webdriver_edge_service (+17 more)

### Community 3 - "LotteryPatternAnalyzer"
Cohesion: 0.08
Nodes (15): LotteryPatternAnalyzer, Analyze lottery patterns using statistical methods. This class provides…, Find the most common pairs of numbers drawn together., Analyze how often consecutive numbers appear in the same draw., Analyze the distribution of numbers (low vs high, odd vs even)., Analyze number frequency across ALL positions (N1-N6 combined). Since numbers…, Test if the lottery is truly random by comparing actual vs expected…, Analyze the frequency of each number drawn. Shows hot numbers (most frequent)… (+7 more)

### Community 4 - "PatternOptimizer"
Cohesion: 0.12
Nodes (14): main(), PatternOptimizer, DataFrame, Save a result to the cache., Get number frequency data for a given lookback window., Pattern-based strategy with configurable parameters. params: lookback: number…, Get actual lottery numbers for a specific date., Test a specific parameter combination. Uses caching to skip recalculation if… (+6 more)

### Community 5 - "MultiStrategyPredictor"
Cohesion: 0.08
Nodes (14): main(), MultiStrategyPredictor, Combine top 3 hot numbers with top 3 cold numbers., Select numbers ensuring balanced range and odd/even distribution., Weight number appearances with exponential decay by recency., Tests multiple lottery prediction strategies and compares their performance., Pick numbers that are overdue compared to their average gap., Get actual lottery numbers for a specific date. (+6 more)

### Community 6 - "analyze_web.py"
Cohesion: 0.07
Nodes (54): backtest(), _balance(), _boletos(), build(), calientes_ventanas(), cargar_premios(), uno(), chi2_uniform() (+46 more)

### Community 7 - "MaximumHitsPredictor"
Cohesion: 0.14
Nodes (12): main(), MaximumHitsPredictor, DataFrame, Generate predictions using pattern-based strategy with given parameters., Get actual lottery numbers for a specific date., Generate prediction by combining votes from top parameter configurations.…, Run backtest against historical data, showing 3+ hit results., Leverages cached optimizer results to generate predictions that maximize the… (+4 more)

### Community 8 - "OverdueBacktester"
Cohesion: 0.16
Nodes (9): main(), OverdueBacktester, Get the actual numbers drawn on a specific date. Args: date: The draw date…, Run backtesting for the last N days of actual results. Args: days_to_test:…, Backtest lottery predictions using the "longest since appeared" strategy.…, Generate and print a comprehensive summary of backtest results., Predict numbers for the next draw using the best performing strategy., Main entry point for the backtest. (+1 more)

### Community 10 - "db"
Cohesion: 0.24
Nodes (7): generate_best_prediction(), Generate the best possible prediction using comprehensive analysis., nextBonolotoDate(), nextDrawDate(), Next date after last_date whose weekday is in weekdays (Mon=0)., Bonoloto draws Mon-Sat. Skip Sunday., db

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
Cohesion: 0.40
Nodes (4): Before HECHO, Steps, Style = `DESIGN.md`, Web design

### Community 18 - "Agent rules"
Cohesion: 0.50
Nodes (3): Agent rules, Flujo, Think → Simple → Surgical → Verify (Karpathy)

### Community 19 - "Laya"
Cohesion: 0.50
Nodes (3): Laya, Reply (decision-only requests), Steps

### Community 20 - "Review"
Cohesion: 0.50
Nodes (3): Check, Do, Review

## Knowledge Gaps
- **30 isolated node(s):** `1. Root cause`, `2. Compare`, `3. Hypothesis`, `4. Fix`, `Red flags → back to step 1` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 196 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LotteryPatternAnalyzer` connect `LotteryPatternAnalyzer` to `commonFunctions.py`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `MultiStrategyPredictor` connect `MultiStrategyPredictor` to `commonFunctions.py`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `PatternOptimizer` connect `PatternOptimizer` to `commonFunctions.py`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **What connects `1. Root cause`, `2. Compare`, `3. Hypothesis` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `edgeHunt.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05786090005844535 - nodes in this community are weakly interconnected._
- **Should `commonFunctions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10256410256410256 - nodes in this community are weakly interconnected._
- **Should `webScraper.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07058823529411765 - nodes in this community are weakly interconnected._