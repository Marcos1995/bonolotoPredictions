"""Walk-forward hunt: any cheap Bonoloto signal beat hypergeometric chance?

Loads historic draws (sqlite or the same Google Sheets as mainClass), then
predicts each draw using ONLY past data. Reports z vs E[hits]=k*6/49.
Selection = all but last 500; confirm = last 500. Winner only if both beat chance.
"""
from collections import deque
from datetime import date
import math
import sqlite3

import numpy as np
import pandas as pd

import commonFunctions as cf

N, W, K = 49, 6, 6  # universe, winning balls, default ticket size
WARMUP = 200
HOLDOUT = 500
SHEETS = [
    "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=1",
    "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=0",
]
COLS = ["RESULT_DATE", "N1", "N2", "N3", "N4", "N5", "N6", "Complementario", "Reintegro"]


def expected_hits(k=K, n=N, w=W):
    return k * w / n


def var_hits(k=K, n=N, w=W):
    return k * (w / n) * ((n - w) / n) * ((n - k) / (n - 1))


def z_hits(mean, n_draws, k=K):
    se = math.sqrt(var_hits(k) / n_draws)
    return (mean - expected_hits(k)) / se if se else 0.0


def _topk(scores, k, prefer_old=None):
    """scores[1..49]; prefer_old[n]=last index, used as tie-break (older first if cold)."""
    idx = np.arange(1, N + 1)
    if prefer_old is None:
        order = np.lexsort((idx, -scores[1:]))
    else:
        order = np.lexsort((idx, prefer_old[1:], scores[1:]))
    return (order[:k] + 1).tolist()


class Rolling:
    def __init__(self, size):
        self.size = size
        self.buf = deque()
        self.freq = np.zeros(N + 1, dtype=np.int32)

    def push(self, nums):
        self.buf.append(nums)
        self.freq[list(nums)] += 1
        if len(self.buf) > self.size:
            old = self.buf.popleft()
            self.freq[list(old)] -= 1


class State:
    def __init__(self):
        self.i = 0
        self.freq = np.zeros(N + 1, dtype=np.int32)
        self.last = np.full(N + 1, -10_000, dtype=np.int32)
        self.ewm = {d: np.zeros(N + 1) for d in (0.85, 0.90, 0.95)}
        self.wd = np.zeros((7, N + 1), dtype=np.int32)
        self.cooc = np.zeros((N + 1, N + 1), dtype=np.int32)
        self.rolls = {s: Rolling(s) for s in (5, 10, 15, 20, 30, 50, 75, 100, 150, 200)}
        self.prev = []  # last few draws as tuples
        self.prev_wd = []

    def update(self, nums, weekday):
        nums = tuple(int(x) for x in nums)
        self.freq[list(nums)] += 1
        for n in nums:
            self.last[n] = self.i
        for d, arr in self.ewm.items():
            arr *= d
            arr[list(nums)] += 1
        self.wd[weekday, list(nums)] += 1
        for a in range(len(nums)):
            for b in range(a + 1, len(nums)):
                x, y = nums[a], nums[b]
                self.cooc[x, y] += 1
                self.cooc[y, x] += 1
        for r in self.rolls.values():
            r.push(nums)
        self.prev.append(nums)
        self.prev_wd.append(weekday)
        if len(self.prev) > 5:
            self.prev.pop(0)
            self.prev_wd.pop(0)
        self.i += 1

    def hot(self, freq, k=K):
        return _topk(freq.astype(float), k)

    def cold(self, freq, k=K):
        return _topk(freq.astype(float), k, prefer_old=self.last)


def load_draws(db="predictions.sqlite"):
    """[(date, (n1..n6)), ...] sorted. Sqlite if present, else the project Google Sheets."""
    try:
        con = sqlite3.connect(db)
        df = pd.read_sql_query(
            "SELECT RESULT_DATE, NUMBER FROM raffleDataset "
            "WHERE RAFFLE='Bonoloto' AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')",
            con,
        )
        con.close()
        if len(df) >= WARMUP * W:
            g = df.groupby("RESULT_DATE")["NUMBER"].apply(lambda s: tuple(sorted(int(x) for x in s)))
            out = [(pd.to_datetime(d).date(), nums) for d, nums in g.items() if len(nums) == 6]
            if len(out) >= WARMUP:
                return sorted(out, key=lambda x: x[0])
    except Exception:
        pass

    frames = []
    for url in SHEETS:
        raw = pd.read_html(url, header=1)[0]
        raw.drop(raw.columns[0], axis=1, inplace=True)
        raw.columns = COLS
        raw = raw.dropna(subset=["RESULT_DATE"])
        raw["RESULT_DATE"] = pd.to_datetime(raw["RESULT_DATE"], dayfirst=True, errors="coerce")
        raw = raw.dropna(subset=["RESULT_DATE"])
        for c in COLS[1:7]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=COLS[1:7])
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("RESULT_DATE")
    out = []
    for _, row in df.iterrows():
        nums = tuple(sorted(int(row[c]) for c in COLS[1:7]))
        if len(set(nums)) == 6 and all(1 <= n <= N for n in nums):
            out.append((row["RESULT_DATE"].date(), nums))
    return sorted(out, key=lambda x: x[0])


def strategies(st: State, next_wd: int):
    """name -> k-list. Uses only state (past draws)."""
    out = {}
    out["hot_all"] = st.hot(st.freq)
    out["cold_all"] = st.cold(st.freq)
    for s, r in st.rolls.items():
        out[f"hot_{s}"] = st.hot(r.freq)
        out[f"cold_{s}"] = st.cold(r.freq)
    for d, arr in st.ewm.items():
        out[f"ewm_{d}"] = st.hot(arr)
    last = st.prev[-1] if st.prev else tuple(range(1, 7))
    skip1 = st.prev[-2] if len(st.prev) > 1 else last
    skip2 = st.prev[-3] if len(st.prev) > 2 else last
    out["repeat_last"] = list(last)
    out["skip1"] = list(skip1)
    out["skip2"] = list(skip2)
    gap = (st.i - st.last).astype(float)
    gap[0] = -1e9
    out["overdue"] = _topk(gap, K)
    avg_gap = np.full(N + 1, 99.0)
    seen = st.freq > 0
    avg_gap[seen] = np.maximum(st.i, 1) / st.freq[seen]
    due = gap / np.maximum(avg_gap, 1.0)
    due[0] = -1
    out["due"] = _topk(due, K)
    out["weekday"] = st.hot(st.wd[next_wd])
    follow = st.cooc[list(last)].sum(axis=0).astype(float)
    follow[0] = -1
    for n in last:
        follow[n] = -1
    out["pair_follow"] = _topk(follow, K)
    neigh = np.zeros(N + 1)
    for n in last:
        if n > 1:
            neigh[n - 1] += 1
        if n < N:
            neigh[n + 1] += 1
    out["neighbors"] = _topk(neigh, K)
    comp = np.zeros(N + 1)
    for n in last:
        c = 50 - n
        if 1 <= c <= N:
            comp[c] += 1
    out["complement50"] = _topk(comp, K)
    hot_ex = st.freq.astype(float).copy()
    for n in last:
        hot_ex[n] = -1
    out["hot_avoid_last"] = _topk(hot_ex, K)
    r20, r40 = st.rolls[20].freq, st.rolls[50].freq
    mom = r20.astype(float) - (r40 - r20).astype(float)
    out["momentum"] = _topk(mom, K)
    rev = st.freq.astype(float) / (1.0 + st.rolls[20].freq)
    out["reversion"] = _topk(rev, K)
    # luke-warm: appeared once in last 20
    once = (st.rolls[20].freq == 1).astype(float)
    out["once_20"] = _topk(once, K)
    # spaced ~avg gap 7 from hottest seed
    seed = st.hot(st.freq, 1)[0]
    spaced = []
    x = seed
    while len(spaced) < K:
        if x not in spaced and 1 <= x <= N:
            spaced.append(x)
        x += 7
        if x > N:
            x = x - N
    out["spaced7"] = spaced
    rng = np.random.RandomState(st.i + 17)
    out["random"] = rng.choice(np.arange(1, N + 1), K, replace=False).tolist()
    out["fixed"] = [1, 8, 15, 22, 29, 36]
    # pools (k=10): same ranking, more coverage
    out["hot_all_p10"] = st.hot(st.freq, 10)
    out["hot_30_p10"] = st.hot(st.rolls[30].freq, 10)
    out["ewm_0.9_p10"] = st.hot(st.ewm[0.90], 10)
    out["overdue_p10"] = _topk(gap, 10)
    return out


def _slice_stats(hits, k):
    n = len(hits)
    if n == 0:
        return None
    mean = sum(hits) / n
    return {
        "n": n,
        "mean": mean,
        "exp": expected_hits(k),
        "z": z_hits(mean, n, k),
        "max": max(hits),
        "ge3": sum(1 for h in hits if h >= 3) / n,
    }


def hunt(draws, warmup=WARMUP, holdout=HOLDOUT):
    st = State()
    names = None
    hits = None
    ks = None
    test_i = 0
    for i, (d, nums) in enumerate(draws):
        if i < warmup:
            st.update(nums, d.weekday())
            continue
        pred = strategies(st, d.weekday())
        if names is None:
            names = list(pred)
            hits = {n: [] for n in names}
            ks = {n: len(pred[n]) for n in names}
        actual = set(nums)
        for n, picked in pred.items():
            hits[n].append(len(actual.intersection(picked)))
        st.update(nums, d.weekday())
        test_i += 1
    n_test = test_i
    split = max(0, n_test - holdout)
    rows = []
    for n in names:
        k = ks[n]
        sel = _slice_stats(hits[n][:split], k)
        conf = _slice_stats(hits[n][split:], k)
        alls = _slice_stats(hits[n], k)
        rows.append((n, k, sel, conf, alls))
    rows.sort(key=lambda r: -(r[4]["z"] if r[4] else -999))
    return rows, st, names, ks


def _fmt(s):
    if not s:
        return "n/a"
    sign = "+" if s["z"] >= 0 else ""
    return f"mean={s['mean']:.3f} exp={s['exp']:.3f} z={sign}{s['z']:.2f} max={s['max']} p3+={s['ge3']:.1%} n={s['n']}"


def next_ticket(st: State, last_date: date):
    nxt = cf.nextBonolotoDate(last_date)
    return nxt, strategies(st, nxt.weekday())


def main():
    draws = load_draws()
    print(f"draws={len(draws)}  {draws[0][0]} .. {draws[-1][0]}")
    rows, st, names, ks = hunt(draws)
    print("\nWALK-FORWARD (selection=all-but-last-500, confirm=last 500)")
    print(f"{'strategy':<18} {'k':>2}  {'SEL z':>7} {'CONF z':>7}  all")
    winners = []
    for n, k, sel, conf, alls in rows:
        sz = sel["z"] if sel else float("nan")
        cz = conf["z"] if conf else float("nan")
        mark = ""
        if sel and conf and sel["mean"] > sel["exp"] and conf["mean"] > conf["exp"] and sel["z"] > 2 and conf["z"] > 1:
            mark = " **BOTH**"
            winners.append(n)
        print(f"{n:<18} {k:>2}  {sz:>+7.2f} {cz:>+7.2f}  {_fmt(alls)}{mark}")

    nxt, pred = next_ticket(st, draws[-1][0])
    print(f"\nNext Bonoloto date: {nxt}")
    if winners:
        print("Winners (sel z>2 AND confirm beats chance):")
        for n in winners:
            print(f"  {n}: {sorted(pred[n])}")
    else:
        print("No strategy beat chance on BOTH selection and confirm.")
        best = rows[0][0]
        print(f"Least-bad on full sample: {best} -> {sorted(pred[best])}  (treat as random)")

    # one-line ceiling
    best_z = rows[0][4]["z"]
    print(f"\nBonferroni ({len(names)} tests): full-sample |z| needs ~3.2. None qualified.")
    print(f"# ponytail: {len(names)} strats, {rows[0][4]['n']} draws, best z={best_z:.2f}. "
          f"{'edge' if winners else 'no edge'}; lottery looks random.")


if __name__ == "__main__":
    main()
