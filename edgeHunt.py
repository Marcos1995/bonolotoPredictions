"""Walk-forward tournament across all games: does ANY signal beat hypergeometric chance?

Zoo: hot/cold windows, EWM + slow "bayes" decays, overdue/due, group overdue
(decade/odd-even/sum/span), weekday/month,
Markov transitions, positional/delta/sum-band shapes, era-consistency bias,
a deliberately silly family (mirror, last digits, anti-birthday, year-ago...),
an sklearn ML tier (logistic + gradient boosting, walk-forward refits) and an
honest Borda ensemble (members chosen on the select window, judged on confirm).

Every strategy's metrics persist in sqlite (strategy_results) and its next-draw
pick is stored (next_draw_predictions) and re-scored automatically on the next
run, so real out-of-sample evidence accumulates draw after draw.

Run: py -3.11 main.py (ingest) then py -3.11 edgeHunt.py
"""
from collections import deque
import datetime as dt
import math
import sqlite3

import numpy as np

import commonFunctions as cf
import raffles

WARMUP = 200
HOLDOUT = 500
DB_FILE = "predictions.sqlite"
DATASET_TABLE = "raffleDataset"
RESULTS_TABLE = "strategy_results"
PRED_TABLE = "next_draw_predictions"
BLOCK = 150      # draws per era-consistency block
ML_MIN_FIT = 300  # observations before first model fit
ML_REFIT = 200    # refit cadence, in draws
ML_WINDOW = 2000  # sliding training window, in draws

try:
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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


def borda(pick_lists, k):
    """Rank-sum vote over ordered pick lists -> top-k numbers."""
    score = {}
    for pl in pick_lists:
        for pos, x in enumerate(pl):
            score[x] = score.get(x, 0.0) + (len(pl) - pos)
    return sorted(score, key=lambda x: (-score[x], x))[:k]


def _fill_from_groups(last, k, n, group_of):
    """k numbers from coldest groups first (group last-hit = max last of members)."""
    groups = {}
    for x in range(1, n + 1):
        groups.setdefault(group_of(x), []).append(x)
    order = sorted(groups, key=lambda g: (max(int(last[x]) for x in groups[g]), g))
    picked = []
    for g in order:
        for x in sorted(groups[g], key=lambda x: (int(last[x]), x)):
            picked.append(int(x))
            if len(picked) >= k:
                return picked
    return picked


def _stale_tercile_bounds(values):
    """(lo, hi) of the tercile whose last hit is oldest. None if too short."""
    if len(values) < 3:
        return None
    arr = np.asarray(values, dtype=float)
    a, b = np.percentile(arr, [100.0 / 3.0, 200.0 / 3.0])
    last_i = [-1, -1, -1]
    for t, v in enumerate(arr):
        last_i[0 if v <= a else 1 if v <= b else 2] = t
    which = min(range(3), key=lambda i: last_i[i])
    if which == 0:
        return -1e18, float(a)
    if which == 1:
        return float(a), float(b)
    return float(b), 1e18


def _nudge_into_band(pick, cand, lo, hi, fn):
    """Swap members until fn(pick) is in [lo, hi]. ponytail: 30 greedy swaps, not global opt."""
    pick = list(pick)
    rest = [c for c in cand if c not in pick]
    for _ in range(30):
        s = fn(pick)
        if lo <= s <= hi or not rest:
            return pick
        want_down = s > hi
        moved = False
        for i, old in enumerate(pick):
            for j, new in enumerate(rest):
                ns = fn(pick[:i] + [new] + pick[i + 1:])
                if (want_down and ns < s) or ((not want_down) and ns > s):
                    pick[i] = new
                    rest[j] = old
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return pick


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
        # slow decays (0.98/0.995) are the beta-binomial "bayes" family: same ranking
        self.ewm = {d: np.zeros(n + 1) for d in (0.85, 0.90, 0.95, 0.98, 0.995)}
        self.wd = np.zeros((7, n + 1), dtype=np.int32)
        self.mo = np.zeros((12, n + 1), dtype=np.int32)
        self.cooc = np.zeros((n + 1, n + 1), dtype=np.int32)
        self.trans = np.zeros((n + 1, n + 1), dtype=np.int32)   # draw t-1 -> draw t
        self.trans_n = np.zeros(n + 1, dtype=np.int32)
        self.pos = np.zeros((k, n + 1), dtype=np.int32)         # sorted-slot freq
        self.dpos = np.zeros((max(k - 1, 1), n + 1), dtype=np.int32)  # sorted-slot gaps
        self.sums = []
        self.hist = []
        self.block_freq = np.zeros(n + 1, dtype=np.int32)
        self.blocks = np.zeros(n + 1)  # era-consistency wins
        self.rolls = {s: Rolling(s, n) for s in (5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300)}
        self.prev = []

    def update(self, nums, date):
        nums = tuple(int(x) for x in nums)
        if self.prev:
            last = self.prev[-1]
            self.trans_n[list(last)] += 1
            for y in last:
                self.trans[y, list(nums)] += 1
        self.freq[list(nums)] += 1
        for x in nums:
            self.last[x] = self.i
        for d, arr in self.ewm.items():
            arr *= d
            arr[list(nums)] += 1
        self.wd[date.weekday(), list(nums)] += 1
        self.mo[date.month - 1, list(nums)] += 1
        srt = sorted(nums)
        for j, x in enumerate(srt):
            self.pos[j, x] += 1
        for j in range(len(srt) - 1):
            self.dpos[j, srt[j + 1] - srt[j]] += 1
        for a in range(len(nums)):
            for b in range(a + 1, len(nums)):
                x, y = nums[a], nums[b]
                self.cooc[x, y] += 1
                self.cooc[y, x] += 1
        for r in self.rolls.values():
            r.push(nums)
        self.sums.append(sum(nums))
        self.hist.append((date, nums))
        self.block_freq[list(nums)] += 1
        self.prev.append(nums)
        if len(self.prev) > 5:
            self.prev.pop(0)
        self.i += 1
        if self.i % BLOCK == 0:
            body = self.block_freq[1:]
            self.blocks[1:][body > np.median(body)] += 1
            self.block_freq[:] = 0

    def hot(self, freq, k=None):
        return _topk(freq.astype(float), k or self.k)

    def cold(self, freq, k=None):
        return _topk(freq.astype(float), k or self.k, prefer_old=self.last)


def strategies(st: State, next_date):
    n, k = st.n, st.k
    out = {}
    out["hot_all"] = st.hot(st.freq)
    out["cold_all"] = st.cold(st.freq)
    for s, r in st.rolls.items():
        out[f"hot_{s}"] = st.hot(r.freq)
        out[f"cold_{s}"] = st.cold(r.freq)
    for d, arr in st.ewm.items():
        nm = f"bayes_{d}" if d >= 0.98 else f"ewm_{d}"
        out[nm] = st.hot(arr)
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
    out["decade_overdue"] = _fill_from_groups(st.last, k, n, lambda x: (x - 1) // 10)
    nums = np.arange(n + 1)
    odd_sc, even_sc = gap.copy(), gap.copy()
    odd_sc[nums % 2 == 0] = -1e9
    even_sc[nums % 2 == 1] = -1e9
    out["odd_overdue"] = _topk(odd_sc, k)
    out["even_overdue"] = _topk(even_sc, k)
    pool = _topk(gap, min(n, 3 * k))
    sb = _stale_tercile_bounds(st.sums)
    out["sum_overdue"] = _nudge_into_band(out["overdue"], pool, *sb, sum) if sb else list(out["overdue"])
    spans = [max(ns) - min(ns) for _d, ns in st.hist]
    pb = _stale_tercile_bounds(spans)
    out["span_overdue"] = (
        _nudge_into_band(out["overdue"], pool, *pb, lambda p: max(p) - min(p)) if pb else list(out["overdue"])
    )
    out["weekday"] = st.hot(st.wd[next_date.weekday()])
    out["month"] = st.hot(st.mo[next_date.month - 1])
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

    # Markov: P(x in draw t | y in draw t-1), summed over last draw
    mk = np.zeros(n + 1)
    for y in last:
        if 0 <= y <= n and st.trans_n[y] > 0:
            mk += st.trans[y] / st.trans_n[y]
    mk[0] = -1
    out["markov"] = _topk(mk, k)

    # era-consistency bias: number of BLOCK-draw eras above median frequency
    out["consistent_hot"] = _topk(st.blocks + st.freq * 1e-9, k)

    # positional: greedy per sorted slot
    picked = []
    for j in range(k):
        order = np.argsort(-st.pos[j][1:], kind="stable") + 1
        for x in order:
            x = int(x)
            if x not in picked:
                picked.append(x)
                break
    out["positional"] = picked

    # positional_mean: rounded mean of each sorted slot
    pm, used = [], set()
    for j in range(k):
        tot = st.pos[j][1:].sum()
        x = int(round((np.arange(1, n + 1) * st.pos[j][1:]).sum() / tot)) if tot else j + 1
        x = min(max(x, 1), n)
        guard = 0
        while x in used and guard <= n:
            x = x + 1 if x < n else 1
            guard += 1
        used.add(x)
        pm.append(x)
    out["positional_mean"] = pm

    # deltas: most common start + most common slot gaps
    d0 = int(np.argmax(st.pos[0][1:])) + 1 if st.pos[0].sum() else 1
    dl, dused = [d0], {d0}
    x = d0
    for j in range(k - 1):
        g = int(np.argmax(st.dpos[j][1:])) + 1 if st.dpos[j].sum() else 2
        x = x + g
        if x > n:
            break
        if x not in dused:
            dl.append(x)
            dused.add(x)
    for x in st.hot(st.freq, n):
        if len(dl) >= k:
            break
        if x not in dused:
            dl.append(x)
            dused.add(x)
    out["deltas"] = dl[:k]

    # sum_band: hottest EWM set nudged into the modal sum band
    cand = _topk(st.ewm[0.90], min(n, 3 * k))
    pick = cand[:k]
    if len(st.sums) >= 30:
        lo, hi = np.percentile(st.sums, [25, 75])
        rest = [c for c in cand if c not in pick]
        for _ in range(30):
            s = sum(pick)
            if lo <= s <= hi or not rest:
                break
            if s > hi:
                worst = max(pick)
                repl = next((c for c in rest if c < worst), None)
            else:
                worst = min(pick)
                repl = next((c for c in rest if c > worst), None)
            if repl is None:
                break
            pick[pick.index(worst)] = repl
            rest.remove(repl)
    out["sum_band"] = list(pick)

    # the deliberately silly family
    out["mirror"] = [n + 1 - x for x in out["hot_all"]]
    digit = np.zeros(10)
    idx = np.arange(1, n + 1)
    np.add.at(digit, idx % 10, st.freq[1:].astype(float))
    ld = np.zeros(n + 1)
    ld[1:] = digit[idx % 10] * 10000 + st.freq[1:]
    out["last_digit_hot"] = _topk(ld, k)
    used_dec = {x // 10 for x in last}
    dec = np.zeros(n + 1)
    dec[1:] = st.freq[1:] + np.where(np.isin(idx // 10, list(used_dec)), 0, 10000)
    out["decade_rotation"] = _topk(dec, k)
    if n > 31 + k:
        ab = st.freq.astype(float).copy()
        ab[:32] = -1
        out["anti_birthday"] = _topk(ab, k)
    odd = st.freq.astype(float).copy()
    odd[idx[idx % 2 == 0]] = -1
    out["odd_hot"] = _topk(odd, k)
    even = st.freq.astype(float).copy()
    even[idx[idx % 2 == 1]] = -1
    out["even_hot"] = _topk(even, k)
    if len(st.hist) >= 60:
        span = max((st.hist[-1][0] - st.hist[0][0]).days, 1)
        back = int(round(365 * len(st.hist) / span))
        j = len(st.hist) - back
        if 0 <= j < len(st.hist):
            out["year_ago"] = list(st.hist[j][1])[:k]

    if n >= 20:
        pk = min(10, n // 3)
        out["hot_all_p10"] = st.hot(st.freq, pk)
        out["hot_30_p10"] = st.hot(st.rolls[30].freq, pk)
        out["ewm_0.9_p10"] = st.hot(st.ewm[0.90], pk)
        out["overdue_p10"] = _topk(gap, pk)
        out["markov_p10"] = _topk(mk, pk)
        out["bayes_0.995_p10"] = st.hot(st.ewm[0.995], pk)
    return out


def ml_features(st: State, next_date):
    """Per-number feature matrix (n, 12) built only from past draws."""
    n = st.n
    i = max(st.i, 1)
    freq = st.freq[1:].astype(float)
    gap = np.clip((st.i - st.last[1:]).astype(float), 0, 2000)
    avg_gap = i / np.maximum(freq, 1.0)
    feats = [gap / 100.0, np.clip(gap / avg_gap, 0.0, 10.0), freq / i]
    for s in (5, 20, 50, 100):
        feats.append(st.rolls[s].freq[1:] / s)
    for d in (0.90, 0.95, 0.995):
        arr = st.ewm[d][1:]
        feats.append(arr / (arr.max() + 1e-9))
    feats.append(np.full(n, next_date.weekday() / 6.0))
    feats.append(np.full(n, (next_date.month - 1) / 11.0))
    return np.column_stack(feats)


class MLTier:
    """Per-number binary classifiers, walk-forward: predict, then observe, refit periodically.
    ponytail: fixed hyperparams + sliding ML_WINDOW; upgrade path is tuning/calibration."""

    def __init__(self, n, k, enabled=True):
        self.n = n
        self.k = k
        self.enabled = enabled and HAS_SKLEARN
        self.X = deque(maxlen=ML_WINDOW)
        self.Y = deque(maxlen=ML_WINDOW)
        self.models = {}
        self.since = 0
        self._f = None

    def prep(self, st, next_date):
        if self.enabled:
            self._f = ml_features(st, next_date)

    def predict(self, st, next_date):
        if not self.enabled:
            return {}
        self._f = ml_features(st, next_date)
        out = {}
        for name, m in self.models.items():
            p = m.predict_proba(self._f)[:, 1]
            sc = np.full(self.n + 1, -1.0)
            sc[1:] = p
            out[f"ml_{name}"] = _topk(sc, self.k)
            if self.n >= 20:
                out[f"ml_{name}_p10"] = _topk(sc, min(10, self.n // 3))
        return out

    def observe(self, nums):
        if not self.enabled or self._f is None:
            return
        y = np.zeros(self.n)
        y[[x - 1 for x in nums]] = 1.0
        self.X.append(self._f)
        self.Y.append(y)
        self._f = None
        self.since += 1
        if (not self.models and len(self.X) >= ML_MIN_FIT) or (self.models and self.since >= ML_REFIT):
            self._fit()

    def _fit(self):
        X = np.vstack(self.X)
        y = np.concatenate(self.Y)
        self.models = {
            "logit": LogisticRegression(max_iter=300).fit(X, y),
            "gbdt": HistGradientBoostingClassifier(
                max_iter=60, max_depth=3, early_stopping=False
            ).fit(X, y),
        }
        self.since = 0


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


def _ensemble_members(rows, w, m=5):
    cands = []
    for nm, k, sel, conf, alls in rows:
        if k != w or nm == "random" or sel is None or sel["n"] < 50:
            continue
        cands.append((sel["z"], nm))
    cands.sort(reverse=True)
    return [nm for _, nm in cands[:m]]


def hunt(draws, n=49, w=6, warmup=None, holdout=None, use_ml=True):
    if warmup is None or holdout is None:
        warmup, holdout = _windows(len(draws))
    st = State(n, w)
    ml = MLTier(n, w, enabled=use_ml)
    test_total = len(draws) - warmup
    split = max(0, test_total - holdout)
    recs = {}   # name -> [(test_idx, hits)]
    picks = {}  # name -> {test_idx: ordered picks}, confirm window only (for ensemble)
    ks = {}
    test_i = 0
    for i, (d, nums) in enumerate(draws):
        if i < warmup:
            ml.prep(st, d)
            st.update(nums, d)
            ml.observe(nums)
            continue
        pred = strategies(st, d)
        pred.update(ml.predict(st, d))
        actual = set(nums)
        for nm, picked in pred.items():
            ks.setdefault(nm, len(picked))
            recs.setdefault(nm, []).append((test_i, len(actual.intersection(picked))))
            if test_i >= split and len(picked) == w:
                picks.setdefault(nm, {})[test_i] = picked
        st.update(nums, d)
        ml.observe(nums)
        test_i += 1
    rows = []
    for nm, rec in recs.items():
        k = ks[nm]
        sel = _slice_stats([h for t, h in rec if t < split], k, n, w)
        conf = _slice_stats([h for t, h in rec if t >= split], k, n, w)
        alls = _slice_stats([h for _, h in rec], k, n, w)
        rows.append((nm, k, sel, conf, alls))
    # honest ensemble: members chosen on select window only, judged on confirm only
    top5 = _ensemble_members(rows, w)
    if top5:
        ens = []
        for t in range(split, test_total):
            member = [picks[nm][t] for nm in top5 if nm in picks and t in picks[nm]]
            if member:
                actual = set(draws[warmup + t][1])
                ens.append(len(actual & set(borda(member, w))))
        if ens:
            conf = _slice_stats(ens, w, n, w)
            rows.append(("ensemble_top5", w, None, conf, conf))
    rows.sort(key=lambda r: -(r[4]["z"] if r[4] else -999))
    return rows, st, ml, top5, warmup, holdout


def _is_winner(sel, conf):
    if sel is None:  # ensemble: built from select-window info, judged on confirm only
        return bool(conf) and conf["mean"] > conf["exp"] and conf["z"] > 2
    return (
        sel and conf
        and sel["mean"] > sel["exp"] and conf["mean"] > conf["exp"]
        and sel["z"] > 2 and conf["z"] > 1
    )


# ---------- sqlite persistence: accumulate evidence run after run ----------

def _series_meta(label):
    """label -> (sqlite raffle, number types, n, w, start date or None)."""
    for name, spec in raffles.GAMES.items():
        if label == name:
            return name, tuple(f"N{j}" for j in range(1, spec["w"] + 1)), spec["n"], spec["w"], None
        extra = spec.get("extra")
        if extra and label == extra["name"]:
            return name, spec["extra_types"], extra["n"], extra["w"], extra.get("from")
    raise KeyError(label)


def _pcon():
    con = sqlite3.connect(DB_FILE)
    con.execute(f"""CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
        RUN_DATE TEXT, SERIES TEXT, STRATEGY TEXT, K INTEGER, N_TEST INTEGER,
        MEAN_HITS REAL, EXP_HITS REAL, Z_ALL REAL, Z_SELECT REAL, Z_CONFIRM REAL, GE3 REAL,
        PRIMARY KEY (RUN_DATE, SERIES, STRATEGY))""")
    con.execute(f"""CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
        SERIES TEXT, DRAW_DATE TEXT, STRATEGY TEXT, K INTEGER, NUMBERS TEXT,
        HITS INTEGER, CREATED TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (SERIES, DRAW_DATE, STRATEGY))""")
    return con


def score_past_predictions():
    """Score stored predictions against results now in the dataset. The real test."""
    con = _pcon()
    rows = con.execute(
        f"SELECT rowid, SERIES, DRAW_DATE, STRATEGY, K, NUMBERS FROM {PRED_TABLE} WHERE HITS IS NULL"
    ).fetchall()
    scored = {}
    for rowid, series, ds, strat, k, numbers in rows:
        raffle, types, n, w, _start = _series_meta(series)
        marks = ",".join("?" * len(types))
        actual = [r[0] for r in con.execute(
            f"SELECT NUMBER FROM {DATASET_TABLE} WHERE RAFFLE=? AND RESULT_DATE=? AND NUMBER_TYPE IN ({marks})",
            (raffle, ds, *types))]
        if len(actual) != w:
            continue  # that draw is not ingested yet
        hits = len(set(int(x) for x in numbers.split(",")) & set(actual))
        con.execute(f"UPDATE {PRED_TABLE} SET HITS=? WHERE rowid=?", (hits, rowid))
        scored.setdefault((series, ds, n, w), []).append((strat, hits, k))
    con.commit()
    con.close()
    if not scored:
        print("no stored predictions to score yet")
    for (series, ds, n, w), lst in sorted(scored.items()):
        main_k = [r for r in lst if r[2] == w]
        best = max(lst, key=lambda r: r[1])
        mean = sum(r[1] for r in main_k) / len(main_k) if main_k else 0.0
        print(f"scored {series} {ds}: {len(lst)} preds, mean hits {mean:.2f} vs chance {expected_hits(w, n, w):.2f}, best {best[0]}={best[1]}")


def save_run(label, rows):
    con = _pcon()
    run = dt.date.today().isoformat()
    for nm, k, sel, conf, alls in rows:
        con.execute(
            f"INSERT OR REPLACE INTO {RESULTS_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run, label, nm, k,
             alls["n"] if alls else 0,
             alls["mean"] if alls else None,
             alls["exp"] if alls else None,
             alls["z"] if alls else None,
             sel["z"] if sel else None,
             conf["z"] if conf else None,
             alls["ge3"] if alls else None))
    con.commit()
    con.close()


def save_predictions(label, draw_date, preds):
    con = _pcon()
    for nm, picked in preds.items():
        con.execute(
            f"INSERT OR IGNORE INTO {PRED_TABLE} (SERIES, DRAW_DATE, STRATEGY, K, NUMBERS, HITS) VALUES (?,?,?,?,?,NULL)",
            (label, draw_date.isoformat(), nm, len(picked), ",".join(map(str, picked))))
    con.commit()
    con.close()


def load_sqlite_draws(label):
    """[(date, tuple)] for a series from the ingested sqlite dataset."""
    raffle, types, n, w, start = _series_meta(label)
    con = sqlite3.connect(DB_FILE)
    marks = ",".join("?" * len(types))
    bydate = {}
    try:
        for ds, num in con.execute(
            f"SELECT RESULT_DATE, NUMBER FROM {DATASET_TABLE} WHERE RAFFLE=? AND NUMBER_TYPE IN ({marks})",
            (raffle, *types)):
            bydate.setdefault(ds, []).append(int(num))
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
    out = []
    for ds, vals in sorted(bydate.items()):
        d = dt.date.fromisoformat(ds[:10])
        if start and d < start:
            continue
        if len(vals) == w and len(set(vals)) == w and all(1 <= x <= n for x in vals):
            out.append((d, tuple(vals)))
    return out


# ---------- runner ----------

def _fz(s):
    return f"{s['z']:+.2f}" if s else "  n/a"


def run_series(label, draws, n, w, weekdays, use_ml=True):
    rows, st, ml, top5, warmup, holdout = hunt(draws, n, w, use_ml=use_ml)
    winners = [(nm, sel, conf, alls) for nm, k, sel, conf, alls in rows if _is_winner(sel, conf)]
    best = rows[0]
    bsel, bconf, balls = best[2], best[3], best[4]
    edge = "EDGE" if winners else "no edge"
    print(
        f"{label} n={n} w={w} draws={len(draws)} {draws[0][0]}..{draws[-1][0]} "
        f"warmup={warmup} holdout={holdout} strategies={len(rows)}"
    )
    for nm, k, sel, conf, alls in rows[:8]:
        print(f"    {nm:<20} k={k:<2} mean={alls['mean']:.3f} exp={alls['exp']:.3f} "
              f"z={alls['z']:+.2f} sel={_fz(sel)} conf={_fz(conf)}")
    print(f"  best {best[0]} mean={balls['mean']:.3f} exp={balls['exp']:.3f} "
          f"z={balls['z']:+.2f} sel={_fz(bsel)} conf={_fz(bconf)}  {edge}")
    for nm, sel, conf, alls in winners[:5]:
        print(f"  WIN {nm} mean={alls['mean']:.3f} z={alls['z']:+.2f} sel={_fz(sel)} conf={_fz(conf)}")
    nxt = cf.nextDrawDate(draws[-1][0], weekdays)
    pred = strategies(st, nxt)
    pred.update(ml.predict(st, nxt))
    if top5:
        member = [pred[nm] for nm in top5 if nm in pred]
        if member:
            pred["ensemble_top5"] = borda(member, w)
            pred["pool10_borda"] = borda(member, min(10, max(n // 3, w)))
    by = {nm: alls for nm, k, sel, conf, alls in rows}
    pick = winners[0][0] if winners else best[0]
    print(f"  next draw {nxt}:")
    print(f"    pick {pick}: {sorted(pred[pick])}")
    if "ensemble_top5" in pred:
        print(f"    ensemble_top5 {top5}: {sorted(pred['ensemble_top5'])}")
        print(f"    pool10_borda: {sorted(pred['pool10_borda'])}")
    print(f"    honest: chance={expected_hits(w, n, w):.2f} hits/draw, '{pick}' backtest={by[pick]['mean']:.2f}")
    save_run(label, rows)
    save_predictions(label, nxt, pred)
    return {"label": label, "winners": [x[0] for x in winners], "best": best[0],
            "z": balls["z"], "n_strats": len(rows)}


def main(use_ml=True):
    print("== scoring stored predictions against ingested results ==")
    score_past_predictions()
    summaries = []
    for name, spec in raffles.GAMES.items():
        print(f"\n=== {name} ===")
        series = [(name, spec["n"], spec["w"], spec["weekdays"])]
        extra = spec.get("extra")
        if extra:
            series.append((extra["name"], extra["n"], extra["w"], extra.get("weekdays", spec["weekdays"])))
        for label, n, w, wds in series:
            draws = load_sqlite_draws(label)
            if len(draws) < 80:
                print(f"{label}: skip ({len(draws)} draws in sqlite; run main.py to ingest)")
                continue
            summaries.append(run_series(label, draws, n, w, wds, use_ml=use_ml))
    print("\n==== SUMMARY ====")
    n_win, n_tests = 0, 0
    for s in summaries:
        if not s:
            continue
        n_tests += s["n_strats"]
        mark = "EDGE" if s["winners"] else "no edge"
        if s["winners"]:
            n_win += 1
        print(f"{s['label']:<28} best={s['best']:<18} z={s['z']:+.2f}  {mark}")
    # multiple-testing honesty: P(z_sel>2)*P(z_conf>1) under the null
    exp_false = n_tests * 0.0228 * 0.1587
    print(f"\n{n_tests} strategy-tests across {len(summaries)} series; "
          f"{sum(len(s['winners']) for s in summaries if s)} sel+confirm winners vs ~{exp_false:.1f} "
          f"expected from pure chance -> only a persistent winner in {PRED_TABLE} scoring counts as real.")


if __name__ == "__main__":
    main()
