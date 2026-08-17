"""Assert helpers used by prediction ingest. Run: py -3.11 test_helpers.py"""
import datetime as dt
import pandas as pd
import commonFunctions as cf

# Sat 8 Aug 2026 -> skip Sunday -> Mon 10
assert cf.nextBonolotoDate(dt.date(2026, 8, 8)) == dt.date(2026, 8, 10)
assert cf.nextBonolotoDate(dt.date(2026, 8, 11)) == dt.date(2026, 8, 12)

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
assert abs(eh.var_hits(6) - 6 * (6 / 49) * (43 / 49) * (43 / 48)) < 1e-12
assert abs(eh.z_hits(eh.expected_hits(6), 1000)) < 1e-12
assert len({1, 2, 3, 4, 5, 6} & {4, 5, 6, 7, 8, 9}) == 3
st = eh.State()
st.i = 10
st.last[:] = 5
st.last[2] = 0  # num 2 unseen longest
got = eh._topk((st.i - st.last).astype(float), 3)
assert got[0] == 2, got

print("ok")
