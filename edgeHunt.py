"""Walk-forward hunt across Lotoideas games: any cheap signal beat hypergeometric chance?"""
from collections import deque
import math

import numpy as np

import commonFunctions as cf
import raffles

WARMUP = 200
HOLDOUT = 500


def expected_hits(k=6, n=49, w=6):
    return k * w / n


def var_hits(k=6, n=49, w=6):
    return k * (w / n) * ((n - w) / n) * ((n - k) / (n - 1))


def z_hits(mean, n_draws, k=6, n=49, w=6):
    se = math.sqrt(var_hits(k, n, w) / n_draws)
    return (mean - expected_hits(k, n, w)) / se if se else 0.0


def _topk(scores, k, prefer_old=None):
    """scores[1..n]; prefer_old[n]=last index, used as tie-break (older first if cold)."""
    n = len(scores) - 1
    idx = np.arange(1, n + 1)
    if prefer_old is None:
        order = np.lexsort((idx, -scores[1:]))
    else:
        order = np.lexsort((idx, prefer_old[1:], scores[1:]))
    return (order[:k] + 1).tolist()


class Rolling:
    def __init__(self, size, n):
        self.size = size
        self.n = n
        self.buf = deque()
        self.freq = np.zeros(n + 1, dtype=np.int32)

    def push(self, nums):
        self.buf.append(nums)
        self.freq[list(nums)] += 1
        if len(self.buf) > self.size:
            old = self.buf.popleft()
            self.freq[list(old)] -= 1


class State:
    def __init__(self, n=49, k=6):
        self.n = n
        self.k = k
        self.i = 0
        self.freq = np.zeros(n + 1, dtype=np.int32)
        self.last = np.full(n + 1, -10_000, dtype=np.int32)
        self.ewm = {d: np.zeros(n + 1) for d in (0.85, 0.90, 0.95)}
        self.wd = np.zeros((7, n + 1), dtype=np.int32)
        self.cooc = np.zeros((n + 1, n + 1), dtype=np.int32)
        self.rolls = {s: Rolling(s, n) for s in (5, 10, 15, 20, 30, 50, 75, 100, 150, 200)}
        self.prev = []
        self.prev_wd = []

    def update(self, nums, weekday):
        nums = tuple(int(x) for x in nums)
        self.freq[list(nums)] += 1
        for x in nums:
            self.last[x] = self.i
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

    def hot(self, freq, k=None):
        return _topk(freq.astype(float), k or self.k)

    def cold(self, freq, k=None):
        return _topk(freq.astype(float), k or self.k, prefer_old=self.last)


def strategies(st: State, next_wd: int):
    n, k = st.n, st.k
    out = {}
    out["hot_all"] = st.hot(st.freq)
    out["cold_all"] = st.cold(st.freq)
    for s, r in st.rolls.items():
        out[f"hot_{s}"] = st.hot(r.freq)
        out[f"cold_{s}"] = st.cold(r.freq)
    for d, arr in st.ewm.items():
        out[f"ewm_{d}"] = st.hot(arr)
    last = st.prev[-1] if st.prev else tuple(range(1, k + 1))
    skip1 = st.prev[-2] if len(st.prev) > 1 else last
    skip2 = st.prev[-3] if len(st.prev) > 2 else last
    out["repeat_last"] = list(last)
    out["skip1"] = list(skip1)
    out["skip2"] = list(skip2)
    gap = (st.i - st.last).astype(float)
    gap[0] = -1e9
    out["overdue"] = _topk(gap, k)
    avg_gap = np.full(n + 1, 99.0)
    seen = st.freq > 0
    avg_gap[seen] = np.maximum(st.i, 1) / st.freq[seen]
    due = gap / np.maximum(avg_gap, 1.0)
    due[0] = -1
    out["due"] = _topk(due, k)
    out["weekday"] = st.hot(st.wd[next_wd])
    follow = st.cooc[list(last)].sum(axis=0).astype(float)
    follow[0] = -1
    for x in last:
        if 0 <= x <= n:
            follow[x] = -1
    out["pair_follow"] = _topk(follow, k)
    neigh = np.zeros(n + 1)
    for x in last:
        if x > 1:
            neigh[x - 1] += 1
        if x < n:
            neigh[x + 1] += 1
    out["neighbors"] = _topk(neigh, k)
    comp = np.zeros(n + 1)
    for x in last:
        c = n + 1 - x
        if 1 <= c <= n:
            comp[c] += 1
    out["complement"] = _topk(comp, k)
    hot_ex = st.freq.astype(float).copy()
    for x in last:
        if 0 <= x <= n:
            hot_ex[x] = -1
    out["hot_avoid_last"] = _topk(hot_ex, k)
    r20, r40 = st.rolls[20].freq, st.rolls[50].freq
    mom = r20.astype(float) - (r40 - r20).astype(float)
    out["momentum"] = _topk(mom, k)
    rev = st.freq.astype(float) / (1.0 + st.rolls[20].freq)
    out["reversion"] = _topk(rev, k)
    once = (st.rolls[20].freq == 1).astype(float)
    out["once_20"] = _topk(once, k)
    step = max(1, n // 7)
    seed = st.hot(st.freq, 1)[0]
    spaced, x = [], seed
    while len(spaced) < k:
        if x not in spaced and 1 <= x <= n:
            spaced.append(x)
        x += step
        if x > n:
            x = x - n
    out["spaced"] = spaced
    rng = np.random.RandomState(st.i + 17)
    out["random"] = rng.choice(np.arange(1, n + 1), k, replace=False).tolist()
    out["fixed"] = [1 + (i * max(1, n // k)) % n for i in range(k)]
    if n >= 20:
        pk = min(10, n // 3)
        out["hot_all_p10"] = st.hot(st.freq, pk)
        out["hot_30_p10"] = st.hot(st.rolls[30].freq, pk)
        out["ewm_0.9_p10"] = st.hot(st.ewm[0.90], pk)
        out["overdue_p10"] = _topk(gap, pk)
    return out


def _slice_stats(hits, k, n, w):
    nd = len(hits)
    if nd == 0:
        return None
    mean = sum(hits) / nd
    return {
        "n": nd,
        "mean": mean,
        "exp": expected_hits(k, n, w),
        "z": z_hits(mean, nd, k, n, w),
        "max": max(hits),
        "ge3": sum(1 for h in hits if h >= 3) / nd,
    }


def _windows(n_draws):
    warmup = min(WARMUP, max(40, n_draws // 10))
    holdout = min(HOLDOUT, max(50, n_draws // 5))
    if warmup + holdout >= n_draws - 10:
        warmup = max(20, n_draws // 5)
        holdout = max(20, n_draws // 5)
    return warmup, holdout


def hunt(draws, n=49, w=6, warmup=None, holdout=None):
    if warmup is None or holdout is None:
        warmup, holdout = _windows(len(draws))
    st = State(n, w)
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
            hits = {nm: [] for nm in names}
            ks = {nm: len(pred[nm]) for nm in names}
        actual = set(nums)
        for nm, picked in pred.items():
            hits[nm].append(len(actual.intersection(picked)))
        st.update(nums, d.weekday())
        test_i += 1
    split = max(0, test_i - holdout)
    rows = []
    for nm in names:
        k = ks[nm]
        sel = _slice_stats(hits[nm][:split], k, n, w)
        conf = _slice_stats(hits[nm][split:], k, n, w)
        alls = _slice_stats(hits[nm], k, n, w)
        rows.append((nm, k, sel, conf, alls))
    rows.sort(key=lambda r: -(r[4]["z"] if r[4] else -999))
    return rows, st, names, ks, warmup, holdout


def _is_winner(sel, conf):
    return (
        sel and conf
        and sel["mean"] > sel["exp"] and conf["mean"] > conf["exp"]
        and sel["z"] > 2 and conf["z"] > 1
    )


def run_series(label, draws, n, w, weekdays):
    if len(draws) < 80:
        print(f"{label}: skip ({len(draws)} draws)")
        return None
    rows, st, names, _ks, warmup, holdout = hunt(draws, n, w)
    winners = [(nm, sel, conf, alls) for nm, k, sel, conf, alls in rows if _is_winner(sel, conf)]
    best = rows[0]
    bsel, bconf, balls = best[2], best[3], best[4]
    edge = "EDGE" if _is_winner(bsel, bconf) else "no edge"
    print(
        f"{label} n={n} w={w} draws={len(draws)} {draws[0][0]}..{draws[-1][0]} "
        f"warmup={warmup} holdout={holdout}"
    )
    print(
        f"  best {best[0]} mean={balls['mean']:.3f} exp={balls['exp']:.3f} "
        f"z={balls['z']:+.2f} sel={bsel['z']:+.2f} conf={bconf['z']:+.2f}  {edge}"
    )
    nxt = cf.nextDrawDate(draws[-1][0], weekdays)
    pred = strategies(st, nxt.weekday())
    if winners:
        for nm, sel, conf, alls in winners[:5]:
            print(f"  WIN {nm} mean={alls['mean']:.3f} z={alls['z']:+.2f} sel={sel['z']:+.2f} conf={conf['z']:+.2f} next {nxt} {sorted(pred[nm])}")
        pick = winners[0][0]
    else:
        pick = best[0]
        print(f"  next {nxt} {pick}: {sorted(pred[pick])}")
    return {"label": label, "winners": [w[0] for w in winners], "best": best[0], "z": balls["z"], "n_strats": len(names)}


def main():
    summaries = []
    for name, spec in raffles.GAMES.items():
        print(f"\n=== {name} ===")
        mains, extras = raffles.load_draws(spec)
        summaries.append(run_series(name, mains, spec["n"], spec["w"], spec["weekdays"]))
        extra = spec.get("extra")
        if extra and extras:
            summaries.append(run_series(
                extra["name"], extras, extra["n"], extra["w"], extra.get("weekdays", spec["weekdays"])
            ))
    print("\n==== SUMMARY ====")
    n_win = 0
    for s in summaries:
        if not s:
            continue
        mark = "EDGE" if s["winners"] else "no edge"
        if s["winners"]:
            n_win += 1
        print(f"{s['label']:<28} best={s['best']:<16} z={s['z']:+.2f}  {mark}")
    print(f"# ponytail: {len(raffles.GAMES)} games; {n_win} series passed sel+confirm (stars only, treat as weak).")


if __name__ == "__main__":
    main()
