"""Bonoloto: patrones históricos y backtest walk-forward contra el azar.

Escribe data/bonoloto.json. Fuente: hojas públicas de raffles.py (mismo origen
que el sqlite local). Sin lookahead: cada regla usa solo sorteos anteriores.
"""
import bisect
import datetime as dt
import json
import math
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import commonFunctions as cf
import raffles

LIVE_URL = "https://www.combinacionganadora.com/bonoloto/"
_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

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


def parse_live(html):
    """6 bolas del tablero en vivo, o None si el sorteo sigue en '?'."""
    sorteo = re.search(r'id="sorteo"(.*)$', html, re.S)
    chunk = sorteo.group(1) if sorteo else html
    block = re.search(r'data-gameNumbers(.*?)</ul>', chunk, re.S)
    if not block:
        return None
    body = block.group(1)
    mains = [int(x) for x in re.findall(r'<li class="bBonoloto[^"]*">(\d{1,2})</li>', body)]
    if len(mains) != 6 or len(set(mains)) != 6 or not all(1 <= n <= 49 for n in mains):
        return None
    extras = tuple(int(x) for x in re.findall(r'data-extra.*?</span>(\d+)</li>', body))
    dm = re.search(r"(\d{1,2}) (\w+) (\d{4})</span>", chunk)
    if not dm:
        return None
    month = _MESES.get(dm.group(2).lower())
    if not month:
        return None
    return dt.date(int(dm.group(3)), month, int(dm.group(1))), tuple(mains), extras


def fetch_live():
    """SELAE responde 403 desde aquí. Este tablero publica el sorteo a los pocos minutos."""
    req = urllib.request.Request(LIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    except Exception:
        return None
    return parse_live(html)


def load():
    spec = raffles.GAMES["Bonoloto"]
    rows = raffles.load_rows(spec)
    draws, reins = [], []
    comps = {}
    ya_ordenadas = 0
    for d, main, extra in rows:
        raw = tuple(main)
        ordered = tuple(sorted(main))
        if raw == ordered:
            ya_ordenadas += 1
        draws.append((d, ordered))
        r = None
        if len(extra) >= 2:
            try:
                r = int(extra[1])
            except (TypeError, ValueError):
                r = None
        if r is not None and 0 <= r <= 9:
            reins.append((d, r))
        c = _comp(extra)
        if c is not None:
            comps[d] = c
    en_vivo = None
    live = fetch_live()
    if live and live[0] not in {d for d, _n in draws}:
        d, main, extra = live
        ordered = tuple(sorted(main))
        if tuple(main) == ordered:
            ya_ordenadas += 1
        draws.append((d, ordered))
        draws.sort(key=lambda x: x[0])
        if len(extra) >= 2 and 0 <= extra[1] <= 9:
            reins.append((d, extra[1]))
        c = _comp(extra)
        if c is not None:
            comps[d] = c
        en_vivo = d.isoformat()
    orden = {
        "filas": len(draws),
        "ya_ordenadas": ya_ordenadas,
        "nota": "La hoja pública trae las seis bolas de menor a mayor. El bombo no sale así: puede salir 45, luego 8, luego 17. Ese orden de extracción no está en los datos.",
        "en_vivo": en_vivo,
    }
    return draws, reins, orden, comps


def _comp(extra):
    if not extra:
        return None
    try:
        c = int(extra[0])
    except (TypeError, ValueError):
        return None
    return c if 1 <= c <= 49 else None


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


DECADE_GROUPS = (
    ("1-9", list(range(1, 10))),
    ("10-19", list(range(10, 20))),
    ("20-29", list(range(20, 30))),
    ("30-39", list(range(30, 40))),
    ("40-49", list(range(40, 50))),
)


def _count_z(mean, exp, var, n):
    se = math.sqrt(var / n) if n and var > 0 else 0
    return (mean - exp) / se if se else 0.0


def decade_rows(draws):
    """Cuántas bolas de cada decena por sorteo, frente a la hipergeométrica."""
    n = len(draws)
    rows = []
    for name, balls in DECADE_GROUPS:
        bset = set(balls)
        k = len(balls)
        counts = [len(bset & set(nums)) for _d, nums in draws]
        mean = sum(counts) / n
        exp = W * k / N
        var = W * (k / N) * ((N - k) / N) * ((N - W) / (N - 1))
        p0 = comb(N - k, W) / SPACE
        p1 = comb(k, 1) * comb(N - k, W - 1) / SPACE
        rows.append({
            "decena": name,
            "bolas": k,
            "por_sorteo": round(mean, 3),
            "azar": round(exp, 3),
            "z": round(_count_z(mean, exp, var, n), 2),
            "con_uno": round(sum(c >= 1 for c in counts) / n, 4),
            "azar_uno": round(1 - p0, 4),
            "con_dos": round(sum(c >= 2 for c in counts) / n, 4),
            "azar_dos": round(1 - p0 - p1, 4),
        })
    return rows


def ending_rows(draws):
    """Terminación 0-9. El 0 solo tiene 4 bolas (10, 20, 30, 40); el resto, 5."""
    n = len(draws)
    c = Counter()
    for _d, nums in draws:
        c.update(x % 10 for x in nums)
    rows = []
    for digit in range(10):
        k = 4 if digit == 0 else 5
        exp = W * k / N
        var = W * (k / N) * ((N - k) / N) * ((N - W) / (N - 1))
        mean = c[digit] / n
        rows.append({
            "digito": digit,
            "bolas": k,
            "veces": c[digit],
            "por_sorteo": round(mean, 3),
            "azar": round(exp, 3),
            "z": round(_count_z(mean, exp, var, n), 2),
        })
    return rows


def slice_pack(draws):
    freq, exp = freq_table(draws)
    oe = odd_even(draws)
    cons = consec_stats(draws)
    return {
        "sorteos": len(draws),
        "frecuencia": freq,
        "pares_impares": oe,
        "decenas": decade_rows(draws),
        "terminaciones": ending_rows(draws),
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
    decenas = decade_rows(draws)

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
        "terminaciones": ending_rows(draws),
        "seguidos": round(sum(g == 1 for g in mg) / n, 4),
        "a_lo_sumo_2": round(sum(g <= 2 for g in mg) / n, 4),
        "azar_a_lo_sumo_2": round(p_cerca, 4),
        "misma_terminacion": round(same_end, 4),
        "azar_misma_terminacion": round(1 - distinct_ends / SPACE, 4),
        "bombo": "Las normas meten 49 bolas del mismo material y peso en un bombo físico, y el reintegro en otro de 10.",
    }


def _pool(past, window, k):
    return _topk(_scores(past[-window:]), k=k, reverse=True)


def _euro_es(text):
    raw = text.replace(".", "").replace(",", ".").strip()
    return float(raw) if raw else 0.0


def parse_premios(html):
    """Premio unitario publicado ese día. Si el 6 no tuvo acertantes, el 6 es el bote."""
    rows = re.findall(
        r'data-cat>([^<]+)</td>\s*<td>([^<]*)</td>\s*<td data-prize>([^<]*)',
        html,
    )
    if not rows:
        return None
    out = {}
    for cat, n, prize in rows:
        cat = cat.strip()
        try:
            acertantes = int(n.replace(".", "").replace(",", ""))
        except ValueError:
            acertantes = 0
        dinero = _euro_es(re.sub(r"[^\d,.\-]", "", prize))
        if cat.startswith("1"):
            out["6"], out["6n"] = dinero, acertantes
        elif cat.startswith("2"):
            out["5c"], out["5cn"] = dinero, acertantes
        elif cat.startswith("3"):
            out["5"], out["5n"] = dinero, acertantes
        elif cat.startswith("4"):
            out["4"], out["4n"] = dinero, acertantes
        elif cat.startswith("5"):
            out["3"], out["3n"] = dinero, acertantes
    bote = re.search(r"<dt>\s*Bote\s*</dt>\s*<dd>([^<]+)", html)
    out["bote"] = _euro_es(re.sub(r"[^\d,.\-]", "", bote.group(1))) if bote else 0.0
    if out.get("6n", 0) == 0:
        out["6"] = out["bote"]
    return out if "3" in out else None


def cargar_premios(fechas):
    """Premios reales por fecha. Cache en data/premios.json; solo baja los que faltan."""
    path = Path("data/premios.json")
    cache = {}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
    faltan = [d for d in fechas if "3n" not in cache.get(d.isoformat(), {})]

    def uno(d):
        url = f"https://www.combinacionganadora.com/bonoloto/resultados/{d.isoformat()}/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
            return d.isoformat(), parse_premios(html)
        except Exception:
            return d.isoformat(), None

    if faltan:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (key, parsed) in enumerate(pool.map(uno, faltan), 1):
                if parsed:
                    cache[key] = parsed
                if i % 40 == 0:
                    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    print(f"premios {i}/{len(faltan)}")
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"premios {sum(1 for d in fechas if d.isoformat() in cache)}/{len(fechas)}")
    return cache


def _boletos(tickets, actual, comp):
    """Aciertos reales de cada boleto, no de todas las combinaciones del grupo."""
    premio = set(actual)
    b = {"3": 0, "4": 0, "5": 0, "5c": 0, "6": 0}
    tuvo5 = tuvo6 = False
    for jugados in tickets:
        hits = len(premio & set(jugados))
        resto = [n for n in jugados if n not in premio]
        if hits == 6:
            b["6"] += 1
            tuvo6 = True
        elif hits == 5 and comp is not None and resto == [comp]:
            b["5c"] += 1
            tuvo5 = True
        elif hits == 5:
            b["5"] += 1
            tuvo5 = True
        elif hits >= 3:
            b[str(hits)] += 1
    return tuvo5, tuvo6, b


def _euros(tabla, boletos):
    """Cada boleto cobra el premio de ese día. Si ya había acertantes, el premio se reparte con ellos."""
    if not any(boletos.values()):
        return 0.0
    if not tabla:
        return None
    total = 0.0
    for key, n in boletos.items():
        if not n:
            continue
        unit = tabla[key]
        ya = tabla.get(key + "n", 0)
        total += n * unit if ya == 0 else n * ya * unit / (ya + n)
    return total


def _mascaras(v):
    out = []
    for comb in combinations(range(v), 6):
        m = 0
        for i in comb:
            m |= 1 << i
        out.append(m)
    return out


def _listas(uni, goal):
    return [[i for i, w in enumerate(uni) if (t & w).bit_count() >= goal] for t in uni]


def _coger(uni, listas, tope, usados):
    """Añade hasta `tope` boletos que cubran combinaciones aún descubiertas."""
    falta = set(range(len(uni)))
    for j in usados:
        for i in listas[j]:
            falta.discard(i)
    elegidos = []
    while falta and len(elegidos) < tope:
        mejor = None
        mejor_n = 0
        for j, lista in enumerate(listas):
            if j in usados or uni[j] in elegidos:
                continue
            n_hit = sum(1 for i in lista if i in falta)
            if n_hit > mejor_n:
                mejor_n = n_hit
                mejor = j
        if not mejor_n:
            break
        elegidos.append(uni[mejor])
        usados.add(mejor)
        for i in listas[mejor]:
            falta.discard(i)
    return elegidos


def _rueda(v, tope):
    uni = _mascaras(v)
    por_meta = {meta: _listas(uni, meta) for meta in (5, 4, 3)}
    usados = set()
    masks = []
    for meta in (5, 4, 3):
        if len(masks) >= tope:
            break
        masks.extend(_coger(uni, por_meta[meta], tope - len(masks), usados))
    for j, t in enumerate(uni):
        if len(masks) >= tope:
            break
        if j not in usados:
            masks.append(t)
            usados.add(j)
    total = len(uni)
    cobertura = {}
    for meta in (3, 4, 5, 6):
        cubiertas = sum(1 for w in uni if any((t & w).bit_count() >= meta for t in masks))
        cobertura[str(meta)] = round(cubiertas / total, 4)
    return masks, cobertura, total


def _cobertura(masks, uni):
    total = len(uni)
    cobertura = {}
    for meta in (3, 4, 5, 6):
        cubiertas = sum(1 for w in uni if any((t & w).bit_count() >= meta for t in masks))
        cobertura[str(meta)] = round(cubiertas / total, 4) if total else 0
    return cobertura


def _indices(masks, k):
    return [[i for i in range(k) if m & (1 << i)] for m in masks]


def _jugar(pool, indices, esta, comp):
    b = {"3": 0, "4": 0, "5": 0, "5c": 0, "6": 0}
    for idxs in indices:
        hits = 0
        resto = None
        for i in idxs:
            n = pool[i]
            if esta[n]:
                hits += 1
            else:
                resto = n
        if hits == 6:
            b["6"] += 1
        elif hits == 5 and comp is not None and resto == comp:
            b["5c"] += 1
        elif hits == 5:
            b["5"] += 1
        elif hits >= 3:
            b[str(hits)] += 1
    return b


def _huecos(pos, i, window):
    lo = i - window
    gaps = [0] * (N + 1)
    for n in range(1, N + 1):
        arr = pos[n]
        j = bisect.bisect_left(arr, i) - 1
        if j < 0 or arr[j] < lo:
            gaps[n] = window
        else:
            gaps[n] = (i - 1) - arr[j]
    return gaps


def barrido(draws, comps, premios):
    """Último año y premios reales. Sin el sorteo de hoy: cada boleto usa solo el pasado."""
    windows = list(range(20, 401, 20))
    modos = (
        ("calientes", "Calientes", "count", True),
        ("frios", "Fríos", "count", False),
        ("retrasados", "Retrasados", "gap", True),
    )
    last = draws[-1][0]
    idx = next(i for i, (d, _n) in enumerate(draws) if d >= last - dt.timedelta(days=364))
    start = max(idx, 1)
    n = len(draws) - start
    cfgs = []
    for k in (6, 8, 9, 10, 12):
        tope = 1 if k == 6 else 20
        masks, _cob, total = _rueda(k, tope)
        uni = _mascaras(k)
        topes = (1,) if k == 6 else (1, 5, 10, 20)
        for t in topes:
            uso = masks[:t]
            cfgs.append((k, t, _indices(uso, k), _cobertura(uso, uni), total))
    prefix = [[0] * (N + 1) for _ in range(len(draws) + 1)]
    pos = [[] for _ in range(N + 1)]
    for i, (_d, nums) in enumerate(draws):
        nxt = prefix[i + 1]
        nxt[:] = prefix[i]
        for num in nums:
            nxt[num] += 1
            pos[num].append(i)
    vacio = {"3": 0, "4": 0, "5": 0, "5c": 0, "6": 0}
    series = {}
    for modo, _nombre, _kind, _rev in modos:
        for k, t, _idx, _cob, _total in cfgs:
            for window in windows:
                series[(modo, k, t, window)] = [0.0, dict(vacio), [], 0.0]
    for paso, i in enumerate(range(start, len(draws)), 1):
        fecha_d, actual = draws[i]
        fecha = fecha_d.isoformat()
        esta = [False] * (N + 1)
        for num in actual:
            esta[num] = True
        comp = comps.get(fecha_d)
        tabla = premios.get(fecha)
        for window in windows:
            counts = [prefix[i][num] - prefix[i - window][num] for num in range(N + 1)]
            gaps = _huecos(pos, i, window)
            listas = {
                "calientes": _topk(counts, k=12, reverse=True),
                "frios": _topk(counts, k=12, reverse=False),
                "retrasados": _topk(gaps, k=12, reverse=True),
            }
            for modo, ranked in listas.items():
                for k, t, indices, _cob, _total in cfgs:
                    boletos = _jugar(ranked[:k], indices, esta, comp)
                    pago = _euros(tabla, boletos)
                    if pago is None:
                        continue
                    celda = series[(modo, k, t, window)]
                    celda[0] += pago
                    if pago > celda[3]:
                        celda[3] = pago
                    for key in vacio:
                        celda[1][key] += boletos[key]
                    if boletos["4"] or boletos["5"] or boletos["5c"] or boletos["6"]:
                        celda[2].append({
                            "fecha": fecha,
                            "salio": list(actual),
                            "complementario": comp,
                            "boletos": boletos,
                            "euros": round(pago, 2),
                        })
        if paso % 80 == 0:
            print(f"barrido {paso}/{n}", flush=True)
    umbral = (7 * len(windows)) // 10
    corte = len(windows) // 5
    pools = []
    for modo, nombre, _kind, _rev in modos:
        for k, t, _idx, cobertura, total in cfgs:
            filas = []
            for window in windows:
                cobrado, suma, dias, mayor = series[(modo, k, t, window)]
                saldo = round(cobrado - n * t * 0.50, 2)
                filas.append({
                    "ventana": window,
                    "saldo": saldo,
                    "sin_mayor": round(saldo - mayor, 2),
                    "boletos": suma,
                    "premios": dias,
                    "mayor": mayor,
                })
            ordenadas = sorted(f["saldo"] for f in filas)
            centro = ordenadas[corte:len(ordenadas) - corte] if corte else ordenadas
            mitad = len(ordenadas) // 2
            mediana = ordenadas[mitad] if len(ordenadas) % 2 else round((ordenadas[mitad - 1] + ordenadas[mitad]) / 2, 2)
            cercana = min(filas, key=lambda f: (abs(f["saldo"] - mediana), f["ventana"]))
            exitos = []
            grandes = {}
            for f in filas:
                if f["saldo"] > 0:
                    exitos.append({
                        "ventana": f["ventana"],
                        "saldo": f["saldo"],
                        "sin_mayor": f["sin_mayor"],
                        "boletos": f["boletos"],
                        "premios": f["premios"],
                    })
                for dia in f["premios"]:
                    if not (dia["boletos"]["5"] or dia["boletos"]["5c"] or dia["boletos"]["6"]):
                        continue
                    rec = grandes.setdefault(dia["fecha"], {
                        "fecha": dia["fecha"],
                        "salio": dia["salio"],
                        "complementario": dia["complementario"],
                        "euros": dia["euros"],
                        "boletos": dia["boletos"],
                        "ventanas": [],
                    })
                    rec["ventanas"].append(f["ventana"])
                    if dia["euros"] > rec["euros"]:
                        rec["euros"] = dia["euros"]
                        rec["boletos"] = dia["boletos"]
            lista_grandes = sorted(grandes.values(), key=lambda r: r["fecha"])
            pools.append({
                "modo": modo,
                "nombre": nombre,
                "cuantos": k,
                "combinaciones": total,
                "apuestas": t,
                "coste_dia": round(t * 0.50, 2),
                "cobertura": cobertura,
                "en_positivo": len(exitos),
                "ventanas": len(filas),
                "mediana": mediana,
                "centro": round(max(centro), 2),
                "centro_n": len(centro),
                "centro_positivas": sum(s > 0 for s in centro),
                "boletos": cercana["boletos"],
                "sin_mayor": cercana["sin_mayor"],
                "suma": round(sum(f["saldo"] for f in filas), 2),
                "maximo": max(f["saldo"] for f in filas),
                "exitos": exitos,
                "grandes": lista_grandes,
                "rentable": mediana > 0 and len(exitos) >= umbral,
            })
            print(
                f"{modo} {k} apuestas={t} mediana={mediana} "
                f"positivas={len(exitos)}/{len(filas)} grandes={len(lista_grandes)}",
                flush=True,
            )
    pools.sort(key=lambda p: p["mediana"], reverse=True)
    return {
        "sorteos": n,
        "desde": draws[start][0].isoformat(),
        "ventanas": len(windows),
        "min_ventana": windows[0],
        "max_ventana": windows[-1],
        "paso": windows[1] - windows[0],
        "corte": corte,
        "centro_n": len(windows) - 2 * corte,
        "umbral": umbral,
        "pools": pools,
        "rentable": any(p["rentable"] for p in pools),
    }


def build():
    draws, reins, orden, comps = load()
    last = draws[-1][0]
    idx = next(i for i, (d, _n) in enumerate(draws) if d >= last - dt.timedelta(days=364))
    premios = cargar_premios([d for d, _n in draws[max(idx, 1):]])
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
        "fuente": "Hojas públicas Lotoideas" + (" y el tablero en vivo del último hueco" if orden.get("en_vivo") else ""),
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
        "orden": orden,
        "patrones": patrones(draws),
        "reintegro": rein,
        "estrategias": strategies,
        "mejor_z": best["z"],
        "forma_modal": {"impares": modal["impares"], "pares": modal["pares"], "obs": modal["obs"], "azar": modal["azar"]},
        "ventanas": windows,
        "calientes_ventanas": barrido(draws, comps, premios),
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
    v = payload["calientes_ventanas"]
    p = v["pools"][0]
    print(
        f"rentable={v['rentable']} mejor={p['nombre']} {p['cuantos']} "
        f"apuestas={p['apuestas']} mediana={p['mediana']} "
        f"{p['en_positivo']}/{p['ventanas']} sin_mayor={p['sin_mayor']} "
        f"suma={p['suma']} maximo={p['maximo']} grandes={len(p['grandes'])}"
    )
    rico = max(v["pools"], key=lambda row: row["suma"])
    print(
        f"suma_mayor={rico['nombre']} {rico['cuantos']} apuestas={rico['apuestas']} "
        f"suma={rico['suma']} maximo={rico['maximo']} mediana={rico['mediana']}"
    )
    p = payload["patrones"]
    print("orden", orden["ya_ordenadas"], "/", orden["filas"])
    print("decenas", [(d["decena"], d["bolas"], d["por_sorteo"], d["azar"], d["z"]) for d in p["decenas"]])
    print(
        f"seguidos={p['seguidos']} cerca2={p['a_lo_sumo_2']}/{p['azar_a_lo_sumo_2']} "
        f"terminacion={p['misma_terminacion']}/{p['azar_misma_terminacion']}"
    )


def _repeats(draws):
    c = Counter(nums for _d, nums in draws)
    return sum(1 for v in c.values() if v > 1)


if __name__ == "__main__":
    build()
