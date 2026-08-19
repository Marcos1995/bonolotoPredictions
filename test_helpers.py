"""Assert helpers used by prediction ingest. Run: py -3.11 test_helpers.py"""
import datetime as dt
import pandas as pd
import commonFunctions as cf

# Sat 8 Aug 2026 -> skip Sunday -> Mon 10
assert cf.nextBonolotoDate(dt.date(2026, 8, 8)) == dt.date(2026, 8, 10)
assert cf.nextBonolotoDate(dt.date(2026, 8, 11)) == dt.date(2026, 8, 12)
assert cf.nextDrawDate(dt.date(2026, 8, 15), (3, 5)) == dt.date(2026, 8, 20)  # Sat -> Thu Primitiva

counts = pd.Series({10: 5, 25: 4, 39: 4, 40: 3, 43: 3, 46: 2, 1: 1})
ticket = cf.topUniqueNumbers(counts, k=6, low=1, high=49)
assert ticket == [10, 25, 39, 40, 43, 46]
assert len(set(ticket)) == 6
assert all(1 <= n <= 49 for n in ticket)

# Unpadded sheet dates must parse
for s, expected in [("9/08/2026", dt.date(2026, 8, 9)), ("11/08/2026", dt.date(2026, 8, 11))]:
    got = pd.to_datetime(s, dayfirst=True).date()
    assert got == expected, (s, got)

import edgeHunt as eh

assert abs(eh.expected_hits(6) - 6 * 6 / 49) < 1e-12
assert abs(eh.expected_hits(5, 50, 5) - 0.5) < 1e-12
assert abs(eh.var_hits(6) - 6 * (6 / 49) * (43 / 49) * (43 / 48)) < 1e-12
assert abs(eh.z_hits(eh.expected_hits(6), 1000)) < 1e-12
assert len({1, 2, 3, 4, 5, 6} & {4, 5, 6, 7, 8, 9}) == 3
st = eh.State()
st.i = 10
st.last[:] = 5
st.last[2] = 0  # num 2 unseen longest
got = eh._topk((st.i - st.last).astype(float), 3)
assert got[0] == 2, got

import raffles
assert {"Bonoloto", "Primitiva", "Euromillones", "ElGordo", "Eurodreams"} <= set(raffles.GAMES)
assert raffles.GAMES["Euromillones"]["w"] == 5
assert raffles.GAMES["Euromillones"]["n"] == 50

# --- tournament additions ---
import numpy as np

# Markov transition counts on a toy sequence
st2 = eh.State(n=5, k=2)
d0 = dt.date(2026, 1, 5)
for j, nums in enumerate([(1, 2), (2, 3), (3, 4)]):
    st2.update(nums, d0 + dt.timedelta(days=j))
assert st2.trans[1][2] == 1 and st2.trans[2][3] == 2 and st2.trans_n[2] == 2
preds = eh.strategies(st2, d0 + dt.timedelta(days=3))
assert preds["markov"] == [3, 4], preds["markov"]

# every strategy returns unique in-range numbers
for nm, picked in preds.items():
    assert len(picked) == len(set(picked)), nm
    assert all(1 <= x <= 5 for x in picked), (nm, picked)

# Borda rank-sum vote
assert eh.borda([[1, 2], [2, 3]], 2) == [2, 1]

# anti-lookahead: harness must predict draw t knowing only draws < t
A, B = (1, 2, 3, 4, 5, 6), (10, 20, 30, 40, 44, 48)
toy = [(dt.date(2026, 1, 1), A), (dt.date(2026, 1, 2), B)]
rows, _st, _ml, _top5, _wu, _ho = eh.hunt(toy, n=49, w=6, warmup=1, holdout=1, use_ml=False)
rl = {nm: alls for nm, k, sel, conf, alls in rows}
assert rl["repeat_last"]["mean"] == 0.0  # it predicted A; actual was B; no leak
assert rl["hot_all"]["mean"] == 0.0

# ML feature builder: finite matrix, one row per number
f = eh.ml_features(st2, d0 + dt.timedelta(days=3))
assert f.shape == (5, 12) and np.isfinite(f).all()
if eh.HAS_SKLEARN:
    mlt = eh.MLTier(5, 2)
    assert mlt.predict(st2, d0) == {}  # no models before first fit

print("ok")
