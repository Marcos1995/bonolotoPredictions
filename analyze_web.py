"""Bonoloto: patrones históricos y backtest walk-forward contra el azar.

Escribe data/bonoloto.json. Fuente: hojas públicas de raffles.py (mismo origen
que el sqlite local). Sin lookahead: cada regla usa solo sorteos anteriores.
"""
import json
import math
from collections import Counter
from pathlib import Path

import commonFunctions as cf
import raffles

N, W = 49, 6
WARMUP = 300
WIN = 100
ODDS = 25
EVENS = 24


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


SPACE = comb(N, W)
EXP_HITS = W * W / N  # 6 aciertos esperados al jugar 6 números


def z_mean(mean, n_test):
    # Var(|A ∩ B|) con |A|=|B|=6 sobre {1..49}
    var = W * (W / N) * ((N - W) / N) * ((N - W) / (N - 1))
    se = math.sqrt(var / n_test) if n_test else 0
    return (mean - EXP_HITS) / se if se else 0.0


def load():
    spec = raffles.GAMES["Bonoloto"]
    rows = raffles.load_rows(spec)
    draws, reins = [], []
    for d, main, extra in rows:
        draws.append((d, tuple(sorted(main))))
        r = None
        if len(extra) >= 2:
            try:
                r = int(extra[1])
            except (TypeError, ValueError):
                r = None
        if r is not None and 0 <= r <= 9:
            reins.append((d, r))
    return draws, reins


def freq_table(draws):
    c = Counter()
    for _d, nums in draws:
        c.update(nums)
    exp = len(draws) * W / N
    out = []
    for n in range(1, N + 1):
        k = c[n]
        # aprox binomial; dentro del sorteo no hay reemplazo, el sesgo es pequeño
        p = W / N
        sd = math.sqrt(len(draws) * p * (1 - p)) if draws else 1
        out.append({"n": n, "count": k, "z": round((k - exp) / sd, 2) if sd else 0})
    return out, exp


def chi2_uniform(counts, exp):
    if exp <= 0:
        return None
    stat = sum((c - exp) ** 2 / exp for c in counts)
    # p-value via regularized gamma, df = 48. Wilson-Hilferty approx.
    df = N - 1
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    # one-sided upper
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return round(stat, 1), round(p, 4)


def odd_even(draws):
    hist = Counter(sum(1 for x in nums if x % 2) for _d, nums in draws)
    rows = []
    for k in range(W + 1):
        p = comb(ODDS, k) * comb(EVENS, W - k) / SPACE
        obs = hist[k] / len(draws)
        rows.append({
            "impares": k,
            "pares": W - k,
            "sorteos": hist[k],
            "obs": round(obs, 4),
            "azar": round(p, 4),
        })
    return rows


def consec_stats(draws):
    def pairs(nums):
        s = sorted(nums)
        return sum(1 for a, b in zip(s, s[1:]) if b - a == 1)

    pc = [pairs(nums) for _d, nums in draws]
    no_consec = comb(N - W + 1, W) / SPACE
    return {
        "con_al_menos_uno": round(sum(1 for x in pc if x >= 1) / len(draws), 4),
        "azar_al_menos_uno": round(1 - no_consec, 4),
        "media_parejas": round(sum(pc) / len(draws), 3),
        "hist": [{"parejas": k, "sorteos": pc.count(k)} for k in range(0, 5)],
    }


def sum_stats(draws):
    sums = [sum(nums) for _d, nums in draws]
    # buckets of 15 around the mean 150
    edges = list(range(60, 250, 15))
    buckets = []
    for a, b in zip(edges, edges[1:]):
        buckets.append({"desde": a, "hasta": b, "sorteos": sum(1 for s in sums if a <= s < b)})
    return {
        "media": round(sum(sums) / len(sums), 1),
        "azar": 150.0,
        "buckets": buckets,
    }


def reintegro_stats(reins):
    c = Counter(r for _d, r in reins)
    exp = len(reins) / 10
    rows = [{"n": i, "count": c[i], "z": round((c[i] - exp) / math.sqrt(exp), 2) if exp else 0} for i in range(10)]
    return {"n": len(reins), "esperado": round(exp, 1), "filas": rows}


def _topk(score, k=W, reverse=True):
    order = sorted(range(1, N + 1), key=lambda n: (score[n], -n), reverse=reverse)
    return order[:k]


def backtest(draws):
    """Reglas que miran solo el pasado. Azar = media teórica, no una simulación."""
    last_seen = [0] * (N + 1)
    total = [0] * (N + 1)
    window = []
    win_count = [0] * (N + 1)
    acc = {name: [] for name in ("calientes", "frios", "historico", "retrasados", "repetir", "equilibrado")}

    for i, (_d, nums) in enumerate(draws):
        if i >= WARMUP:
            hot = _topk(win_count, reverse=True)
            cold = _topk(win_count, reverse=False)
            hist = _topk(total, reverse=True)
            gap = [0] + [i - last_seen[n] for n in range(1, N + 1)]
            late = _topk(gap, reverse=True)
            prev = list(draws[i - 1][1])
            bal = _balance(hist)
            actual = set(nums)
            for name, pick in (
                ("calientes", hot),
                ("frios", cold),
                ("historico", hist),
                ("retrasados", late),
                ("repetir", prev),
                ("equilibrado", bal),
            ):
                acc[name].append(len(actual & set(pick)))
        for n in nums:
            total[n] += 1
            last_seen[n] = i
        window.append(nums)
        for n in nums:
            win_count[n] += 1
        if len(window) > WIN:
            old = window.pop(0)
            for n in old:
                win_count[n] -= 1

    n_test = len(next(iter(acc.values())))
    meta = {
        "calientes": ("Calientes", f"Los 6 que más salieron en los {WIN} sorteos previos."),
        "frios": ("Fríos", f"Los 6 que menos salieron en los {WIN} sorteos previos."),
        "historico": ("Frecuencia total", "Los 6 más repetidos en todo el pasado."),
        "retrasados": ("Retrasados", "Los 6 que más sorteos llevan sin salir."),
        "repetir": ("Repetir ayer", "Jugar otra vez la combinación del sorteo anterior."),
        "equilibrado": ("3 y 3", "De los más frecuentes, forzar 3 pares y 3 impares."),
    }
    rows = []
    for key, hits in acc.items():
        mean = sum(hits) / n_test
        name, desc = meta[key]
        rows.append({
            "id": key,
            "nombre": name,
            "desc": desc,
            "media": round(mean, 4),
            "azar": round(EXP_HITS, 4),
            "z": round(z_mean(mean, n_test), 2),
            "pruebas": n_test,
        })
    rows.sort(key=lambda r: r["z"], reverse=True)
    return rows


def _balance(ranked):
    odd, even = [], []
    for n in ranked:
        (odd if n % 2 else even).append(n)
        if len(odd) >= 3 and len(even) >= 3:
            break
    # complete from the rest of 1..49 if the top list is lopsided
    if len(odd) < 3 or len(even) < 3:
        for n in range(1, N + 1):
            if n in odd or n in even:
                continue
            (odd if n % 2 else even).append(n)
            if len(odd) >= 3 and len(even) >= 3:
                break
    return sorted(odd[:3] + even[:3])


def _scores(draws):
    total = [0] * (N + 1)
    for _d, nums in draws:
        for n in nums:
            total[n] += 1
    return total


def next_ticket(draws):
    """3 pares y 3 impares de los más frecuentes. El backtest no le gana al azar."""
    return sorted(_balance(_topk(_scores(draws), k=N, reverse=True)))


def picks_now(draws):
    total = _scores(draws)
    recent = _scores(draws[-WIN:])
    last_seen = {}
    for i, (_d, nums) in enumerate(draws):
        for n in nums:
            last_seen[n] = i
    gap = [0] + [len(draws) - last_seen.get(n, 0) for n in range(1, N + 1)]
    return {
        "calientes": sorted(_topk(recent, reverse=True)),
        "frios": sorted(_topk(recent, reverse=False)),
        "historico": sorted(_topk(total, reverse=True)),
        "retrasados": sorted(_topk(gap, reverse=True)),
        "repetir": sorted(draws[-1][1]),
        "equilibrado": next_ticket(draws),
    }


def slice_pack(draws):
    freq, exp = freq_table(draws)
    oe = odd_even(draws)
    cons = consec_stats(draws)
    return {
        "sorteos": len(draws),
        "frecuencia": freq,
        "pares_impares": oe,
        "seguidos": cons["con_al_menos_uno"],
        "esperado_por_numero": round(exp, 1),
    }


def shape_check(draws):
    """La forma (cuántos impares, hueco, amplitud) se juzga con el pasado solo."""
    odds = [sum(1 for x in nums if x % 2) for _d, nums in draws]
    hist = [0] * 7
    hit3 = hit_mode = 0
    for i, k in enumerate(odds):
        if i >= WARMUP:
            mode = max(range(7), key=lambda j: (hist[j], -abs(j - 3)))
            hit_mode += k == mode
            hit3 += k == 3
        hist[k] += 1
    n_test = len(draws) - WARMUP
    prior = [0] * 7
    for k in odds[:-1]:
        prior[k] += 1
    last_nums = draws[-1][1]
    last_k = odds[-1]

    def span(nums):
        s = sorted(nums)
        return s[-1] - s[0]

    spans = [span(nums) for _d, nums in draws]
    mass = 0
    exp_span = 0.0
    for s in range(W - 1, N):
        c = (N - s) * comb(s - 1, W - 2)
        mass += c
        exp_span += s * c
    exp_span /= mass

    high_n, low_n = 25, 24
    highs = Counter(sum(1 for x in nums if x >= 25) for _d, nums in draws)
    altos = []
    for h in range(W + 1):
        p = comb(high_n, h) * comb(low_n, W - h) / SPACE
        altos.append({
            "altos": h,
            "sorteos": highs[h],
            "obs": round(highs[h] / len(draws), 4),
            "azar": round(p, 4),
        })
    return {
        "pruebas": n_test,
        "acierto_3_y_3": round(hit3 / n_test, 4),
        "acierto_moda_del_pasado": round(hit_mode / n_test, 4),
        "ultimo": {
            "fecha": draws[-1][0].isoformat(),
            "numeros": list(last_nums),
            "impares": last_k,
            "moda_con_el_pasado": max(range(7), key=lambda j: prior[j]),
            "era_3_y_3": last_k == 3,
        },
        "amplitud_media": round(sum(spans) / len(spans), 1),
        "amplitud_azar": round(exp_span, 1),
        "hueco_medio": round(sum(spans) / len(spans) / (W - 1), 2),
        "hueco_azar": round(exp_span / (W - 1), 2),
        "altos": altos,
        "con_el_pasado": _score_past(draws),
    }


def _score_past(draws):
    """Boletos armados solo con sorteos anteriores al último, contra ese último."""
    actual = set(draws[-1][1])
    picks = picks_now(draws[:-1])
    return {
        "fecha": draws[-1][0].isoformat(),
        "salio": list(draws[-1][1]),
        "reglas": {k: {"numeros": v, "aciertos": len(actual & set(v))} for k, v in picks.items()},
    }


def patrones(draws):
    """Decenas, terminaciones y distancias, frente a la combinatoria."""
    n = len(draws)
    groups = [
        ("1-9", list(range(1, 10))),
        ("10-19", list(range(10, 20))),
        ("20-29", list(range(20, 30))),
        ("30-39", list(range(30, 40))),
        ("40-49", list(range(40, 50))),
    ]
    decenas = []
    for name, balls in groups:
        bset = set(balls)
        k = len(balls)
        counts = [len(bset & set(nums)) for _d, nums in draws]
        p0 = comb(N - k, W) / SPACE
        p1 = comb(k, 1) * comb(N - k, W - 1) / SPACE
        decenas.append({
            "decena": name,
            "bolas": k,
            "por_sorteo": round(sum(counts) / n, 3),
            "azar": round(W * k / N, 3),
            "con_uno": round(sum(c >= 1 for c in counts) / n, 4),
            "azar_uno": round(1 - p0, 4),
            "con_dos": round(sum(c >= 2 for c in counts) / n, 4),
            "azar_dos": round(1 - p0 - p1, 4),
        })

    def min_gap(nums):
        s = sorted(nums)
        return min(b - a for a, b in zip(s, s[1:]))

    mg = [min_gap(nums) for _d, nums in draws]
    p_cerca = 1 - comb(N - (W - 1) * 2, W) / SPACE
    distinct_ends = comb(9, 5) * 4 * (5 ** 5) + comb(9, 6) * (5 ** 6)
    same_end = sum(
        1 for _d, nums in draws if len({x % 10 for x in nums}) < W
    ) / n
    return {
        "decenas": decenas,
        "seguidos": round(sum(g == 1 for g in mg) / n, 4),
        "a_lo_sumo_2": round(sum(g <= 2 for g in mg) / n, 4),
        "azar_a_lo_sumo_2": round(p_cerca, 4),
        "misma_terminacion": round(same_end, 4),
        "azar_misma_terminacion": round(1 - distinct_ends / SPACE, 4),
        "bombo": "Las normas meten 49 bolas del mismo material y peso en un bombo físico, y el reintegro en otro de 10.",
    }


def build():
    draws, reins = load()
    freq, exp = freq_table(draws)
    counts = [row["count"] for row in freq]
    chi = chi2_uniform(counts, exp)
    oe = odd_even(draws)
    cons = consec_stats(draws)
    last_date = draws[-1][0]
    year_cut = last_date.replace(year=last_date.year - 1)
    windows = {
        "todo": slice_pack(draws),
        "d200": slice_pack(draws[-200:]),
        "anyo": slice_pack([d for d in draws if d[0] >= year_cut]),
    }
    sums = sum_stats(draws)
    formas = shape_check(draws)
    rein = reintegro_stats(reins)
    strategies = backtest(draws)
    last = draws[-1][0]
    nxt = cf.nextBonolotoDate(last)
    best = strategies[0]
    modal = max(oe, key=lambda r: r["sorteos"])
    payload = {
        "juego": "Bonoloto",
        "fuente": "Hojas públicas Lotoideas (mismo origen que raffles.py)",
        "desde": draws[0][0].isoformat(),
        "hasta": last.isoformat(),
        "proximo": nxt.isoformat(),
        "sorteos": len(draws),
        "universo": SPACE,
        "aciertos_azar": round(EXP_HITS, 4),
        "chi2": {"stat": chi[0], "p": chi[1], "esperado_por_numero": round(exp, 1)} if chi else None,
        "frecuencia": freq,
        "pares_impares": oe,
        "seguidos": cons,
        "sumas": sums,
        "formas": formas,
        "patrones": patrones(draws),
        "reintegro": rein,
        "estrategias": strategies,
        "mejor_z": best["z"],
        "forma_modal": {"impares": modal["impares"], "pares": modal["pares"], "obs": modal["obs"], "azar": modal["azar"]},
        "ventanas": windows,
        "boletos": picks_now(draws),
        "intento": {
            "regla": best["id"],
            "nombre": best["nombre"],
            "numeros": picks_now(draws)[best["id"]],
            "media": best["media"],
            "z": best["z"],
            "extra": round(best["media"] - EXP_HITS, 4),
        },
        "propuesta": {
            "numeros": next_ticket(draws),
            "nota": "3 pares y 3 impares de los más frecuentes. Esa regla queda por debajo del azar en el backtest.",
        },
        "repeticiones_exactas": _repeats(draws),
        "warmup": WARMUP,
        "ventana": WIN,
    }
    out = Path("data")
    out.mkdir(exist_ok=True)
    (out / "bonoloto.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        f"sorteos={payload['sorteos']} {payload['desde']}..{payload['hasta']} proximo={payload['proximo']} "
        f"mejor={best['nombre']} z={best['z']} forma3={formas['acierto_3_y_3']} "
        f"ultimo={formas['ultimo']} hueco={formas['hueco_medio']}/{formas['hueco_azar']}"
    )
    p = payload["patrones"]
    print("decenas", [(d["decena"], d["por_sorteo"], d["azar"], d["con_uno"]) for d in p["decenas"]])
    print(
        f"seguidos={p['seguidos']} cerca2={p['a_lo_sumo_2']}/{p['azar_a_lo_sumo_2']} "
        f"terminacion={p['misma_terminacion']}/{p['azar_misma_terminacion']}"
    )


def _repeats(draws):
    c = Counter(nums for _d, nums in draws)
    return sum(1 for v in c.values() if v > 1)


if __name__ == "__main__":
    build()
