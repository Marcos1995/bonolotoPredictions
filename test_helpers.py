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

print("ok")
