"""Bonoloto: patrones históricos y backtest walk-forward contra el azar.

Escribe data/bonoloto.json. Fuente: hojas públicas de raffles.py (mismo origen
que el sqlite local). Si a la hoja le falta un sorteo reciente, se rellena con
la ficha de ese día. Sin lookahead: cada regla usa solo sorteos anteriores.
"""
import bisect
import datetime as dt
import json
import math
import re
import urllib.request
from collections import Counter
from zoneinfo import ZoneInfo
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


def _html(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    except Exception:
        return None


def fetch_live():
    """SELAE responde 403 desde aquí. Este tablero publica el sorteo a los pocos minutos."""
    html = _html(LIVE_URL)
    return parse_live(html) if html else None


def fetch_dia(day):
    """Ficha de un día ya celebrado. None si esa fecha aún no está publicada."""
    html = _html(f"https://www.combinacionganadora.com/bonoloto/resultados/{day.isoformat()}/")
    got = parse_live(html) if html else None
    if not got or got[0] != day:
        return None
    return got


_MADRID = ZoneInfo("Europe/Madrid")


def _tope_publicado(today=None, now=None):
    """Último sorteo que ya debería estar publicado. El bombo es a las 21:30 en Madrid.

    Una hora sin zona se lee como hora de Madrid. El runner de GitHub va en UTC.
    """
    if now is None:
        now = dt.datetime.now(_MADRID)
    elif now.tzinfo is not None:
        now = now.astimezone(_MADRID)
    today = today or now.date()
    end = today
    if end.weekday() < 6 and (now.hour, now.minute) < (21, 40):
        end -= dt.timedelta(days=1)
    while end.weekday() == 6:
        end -= dt.timedelta(days=1)
    return end


def dias_sin_datos(have, today=None, now=None, days=21):
    tope = _tope_publicado(today, now)
    start = tope - dt.timedelta(days=days)
    missing = []
    d = start
    while d <= tope:
        if d.weekday() < 6 and d not in have:
            missing.append(d)
        d += dt.timedelta(days=1)
    return missing


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
    rellenos = []
    have = {d for d, _n in draws}
    for day in dias_sin_datos(have):
        got = fetch_dia(day)
        if not got:
            continue
        d, main, extra = got
        ordered = tuple(sorted(main))
        if tuple(main) == ordered:
            ya_ordenadas += 1
        draws.append((d, ordered))
        have.add(d)
        rellenos.append(d.isoformat())
        if len(extra) >= 2 and 0 <= extra[1] <= 9:
            reins.append((d, extra[1]))
        c = _comp(extra)
        if c is not None:
            comps[d] = c
    if rellenos:
        draws.sort(key=lambda x: x[0])
    siguen = [d.isoformat() for d in dias_sin_datos(have)]
    if siguen:
        print("sin datos", " ".join(siguen))
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
        "rellenos": rellenos,
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


HISTORIA = 1825
_MES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
PLAZO = "1 mes"


def _mediana(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def _meses(draws, start, n):
    """Meses naturales en la ventana. (etiqueta, lo, hi): el saldo es pref[hi] - pref[lo]."""
    out = []
    cur = None
    lo = 0
    for paso, i in enumerate(range(start, start + n), 1):
        d = draws[i][0]
        key = (d.year, d.month)
        if cur is None:
            cur = key
            lo = paso - 1
        elif key != cur:
            y, m = cur
            out.append((f"{_MES[m - 1]} {y}", lo, paso - 1))
            cur = key
            lo = paso - 1
    if cur is not None:
        y, m = cur
        out.append((f"{_MES[m - 1]} {y}", lo, n))
    if out:
        i0 = start + out[0][1]
        d0 = draws[i0][0]
        prev = draws[i0 - 1][0] if i0 else None
        if prev and (prev.year, prev.month) == (d0.year, d0.month):
            out = out[1:]
    return out


def barrido(draws, comps, premios):
    """La mirada elige los números (20 a 400). El plazo es un mes natural."""
    modos = (
        ("calientes", "Calientes", "count", True),
        ("frios", "Fríos", "count", False),
        ("retrasados", "Retrasados", "gap", True),
    )
    max_w = HISTORIA
    start = len(draws) - max_w
    n = max_w
    cfgs = []
    for k in range(6, 13):
        masks, _cob, total = _rueda(k, min(20, comb(k, 6)))
        uni = _mascaras(k)
        vistos = set()
        for t in ((1,) if k == 6 else (1, 5, 10, 20)):
            n_ap = min(t, len(masks))
            if n_ap in vistos:
                continue
            vistos.add(n_ap)
            uso = masks[:n_ap]
            cfgs.append((k, n_ap, _indices(uso, k), _cobertura(uso, uni), total))
    prefix = [[0] * (N + 1) for _ in range(len(draws) + 1)]
    pos = [[] for _ in range(N + 1)]
    for i, (_d, nums) in enumerate(draws):
        nxt = prefix[i + 1]
        nxt[:] = prefix[i]
        for num in nums:
            nxt[num] += 1
            pos[num].append(i)
    miradas = list(range(20, 401, 20))
    meses = _meses(draws, start, n)
    lo_ult = meses[-1][1]
    largo_ult = n - lo_ult
    series = {}
    grandes = {}
    acum = {}
    snaps = {}
    for modo, _nombre, _kind, _rev in modos:
        for k, t, _idx, _cob, _total in cfgs:
            for window in miradas:
                key = (modo, k, t, window)
                series[key] = [0.0]
                grandes[key] = []
                acum[key] = [0, 0, 0, 0, 0, 0, 0]
                snaps[key] = {}
    cortes = {lo_ult} if lo_ult else set()
    for paso, i in enumerate(range(start, len(draws)), 1):
        fecha_d, actual = draws[i]
        fecha = fecha_d.isoformat()
        esta = [False] * (N + 1)
        for num in actual:
            esta[num] = True
        comp = comps.get(fecha_d)
        tabla = premios.get(fecha)
        for window in miradas:
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
                    key = (modo, k, t, window)
                    prev = series[key][-1]
                    if pago is None:
                        series[key].append(prev)
                        continue
                    series[key].append(prev + pago)
                    box = acum[key]
                    box[0] += boletos["3"]
                    box[1] += boletos["4"]
                    box[2] += boletos["5"]
                    box[3] += boletos["5c"]
                    box[4] += boletos["6"]
                    if boletos["3"] or boletos["4"] or boletos["5"] or boletos["5c"] or boletos["6"]:
                        box[5] += 1
                        if not (boletos["5"] or boletos["5c"] or boletos["6"]):
                            box[6] += 1
                    if boletos["5"] or boletos["5c"] or boletos["6"]:
                        grandes[key].append({
                            "i": paso,
                            "fecha": fecha,
                            "salio": list(actual),
                            "complementario": comp,
                            "boletos": boletos,
                            "euros": round(pago, 2),
                        })
        if paso in cortes or paso == n:
            for key, box in acum.items():
                snaps[key][paso] = tuple(box)
        if paso % 200 == 0:
            print(f"barrido {paso}/{n}", flush=True)
    plazos = [{
        "nombre": PLAZO,
        "sorteos": largo_ult,
        "desde": draws[start + lo_ult][0].isoformat(),
        "hasta": draws[-1][0].isoformat(),
    }]
    def delta(key, largo):
        end = snaps[key][n]
        base = snaps[key].get(n - largo, (0, 0, 0, 0, 0, 0, 0))
        return [end[i] - base[i] for i in range(7)]

    pools = []
    mejores = {PLAZO: None}
    uno_saldo = None
    uno_sin = None
    for modo, nombre, _kind, _rev in modos:
        for k, t, _idx, _cob, _total in cfgs:
            for window in miradas:
                key = (modo, k, t, window)
                pref = series[key]
                hits = grandes[key]
                for nombre_p, largo in ((PLAZO, largo_ult),):
                    cobrado = pref[-1] - pref[-1 - largo]
                    saldo = round(cobrado - largo * t * 0.50, 2)
                    tres, cuatros, cincos, cincoc, seises, dias, dias_menor = delta(key, largo)
                    vistos = [dia for dia in hits if dia["i"] > n - largo]
                    euros6 = sum(dia["euros"] for dia in vistos if dia["boletos"]["6"])
                    euros_grande = sum(dia["euros"] for dia in vistos)
                    row = {
                        "modo": modo,
                        "nombre": nombre,
                        "cuantos": k,
                        "apuestas": t,
                        "coste_dia": round(t * 0.50, 2),
                        "plazo": nombre_p,
                        "mirados": window,
                        "sorteos": largo,
                        "saldo": saldo,
                        "sin6": round(saldo - euros6, 2),
                        "sin5": round(saldo - euros_grande, 2),
                        "tres": tres,
                        "cuatros": cuatros,
                        "cincos": cincos,
                        "cincoc": cincoc,
                        "seises": seises,
                        "dias": dias,
                        "dias_menor": dias_menor,
                        "grandes": [
                            {k2: v for k2, v in dia.items() if k2 != "i"}
                            for dia in vistos
                        ],
                    }
                    if saldo > 0:
                        pools.append(row)
                    cur = mejores[nombre_p]
                    if cur is None or saldo > cur["saldo"]:
                        mejores[nombre_p] = row
                    if t == 1 and (uno_saldo is None or saldo > uno_saldo["saldo"]):
                        uno_saldo = row
                    if t == 1 and (uno_sin is None or row["sin5"] > uno_sin["sin5"]):
                        uno_sin = row
    pools.sort(key=lambda p: p["saldo"], reverse=True)
    for row in pools[:8]:
        print(
            f"+ {row['nombre']} {row['cuantos']} mirados={row['mirados']} {row['plazo']} "
            f"apuestas={row['apuestas']} saldo={row['saldo']} sin5={row['sin5']} "
            f"3={row['tres']} 4={row['cuatros']} dias={row['dias']}/{row['dias_menor']}",
            flush=True,
        )
    if uno_saldo:
        print(
            f"1apuesta saldo={uno_saldo['nombre']} {uno_saldo['cuantos']} mirados={uno_saldo['mirados']} "
            f"{uno_saldo['plazo']} saldo={uno_saldo['saldo']} sin5={uno_saldo['sin5']} "
            f"3={uno_saldo['tres']} 4={uno_saldo['cuatros']}",
            flush=True,
        )
    if uno_sin:
        print(
            f"1apuesta sin5={uno_sin['nombre']} {uno_sin['cuantos']} mirados={uno_sin['mirados']} "
            f"{uno_sin['plazo']} saldo={uno_sin['saldo']} sin5={uno_sin['sin5']}",
            flush=True,
        )
    claves = [
        (modo, nombre, k, t, window)
        for modo, nombre, _kind, _rev in modos
        for k, t, _idx, _cob, _total in cfgs
        for window in miradas
    ]
    reglas = [
        {"modo": modo, "nombre": nombre, "cuantos": k, "apuestas": t, "mirados": window}
        for modo, nombre, k, t, window in claves
    ]
    n_reg = len(claves)
    premios_acc = [0.0] * n_reg
    gasto_acc = [0.0] * n_reg
    euros6_acc = [0.0] * n_reg
    euros5_acc = [0.0] * n_reg
    pos_acc = [0] * n_reg
    rindes_mes = [[] for _ in range(n_reg)]
    tramos = []
    for etiqueta, lo, hi_t in meses:
        saldo_t = []
        sin5_t = []
        seises_t = []
        sorteos_m = hi_t - lo
        for i, (modo, _nombre, k, t, window) in enumerate(claves):
            key = (modo, k, t, window)
            pref = series[key]
            cobrado = pref[hi_t] - pref[lo]
            gasto_m = sorteos_m * t * 0.50
            saldo_i = round(cobrado - gasto_m, 2)
            euros = 0.0
            euros6 = 0.0
            n6 = 0
            for dia in grandes[key]:
                if lo < dia["i"] <= hi_t:
                    euros += dia["euros"]
                    n6 += dia["boletos"]["6"]
                    if dia["boletos"]["6"]:
                        euros6 += dia["euros"]
            saldo_t.append(saldo_i)
            sin5_t.append(round(saldo_i - euros, 2))
            seises_t.append(n6)
            premios_acc[i] += cobrado
            gasto_acc[i] += gasto_m
            euros6_acc[i] += euros6
            euros5_acc[i] += euros
            if saldo_i > 0:
                pos_acc[i] += 1
            rindes_mes[i].append(cobrado / gasto_m if gasto_m else 0.0)
        tramos.append({
            "mes": etiqueta,
            "desde": draws[start + lo][0].isoformat(),
            "hasta": draws[start + hi_t - 1][0].isoformat(),
            "saldo": saldo_t,
            "sin5": sin5_t,
            "seises": seises_t,
        })
    series_tramo = [{"nombre": PLAZO, "largo": "mes", "tramos": tramos}]
    top = {}
    for tramo in tramos:
        i = max(range(len(reglas)), key=lambda j: tramo["saldo"][j])
        r = reglas[i]
        marca = (r["modo"], r["cuantos"], r["apuestas"], r["mirados"])
        top[marca] = top.get(marca, 0) + 1
    frecuente = max(top, key=top.get)
    print(
        f"tramo {PLAZO} n={len(tramos)} gana={frecuente[0]} {frecuente[1]} "
        f"apuestas={frecuente[2]} mirados={frecuente[3]} veces={top[frecuente]}",
        flush=True,
    )
    cuadro = []
    for i, (modo, nombre, k, t, window) in enumerate(claves):
        prem = premios_acc[i]
        gasto = gasto_acc[i]
        saldo = prem - gasto
        cuadro.append({
            "modo": modo,
            "nombre": nombre,
            "cuantos": k,
            "apuestas": t,
            "mirados": window,
            "plazo": PLAZO,
            "meses": len(tramos),
            "meses_pos": pos_acc[i],
            "mediana": round(_mediana(rindes_mes[i]), 4),
            "premios": round(prem, 2),
            "gasto": round(gasto, 2),
            "rinde": round(prem / gasto, 4) if gasto else 0,
            "saldo": round(saldo, 2),
            "sin6": round(saldo - euros6_acc[i], 2),
            "sin5": round(saldo - euros5_acc[i], 2),
        })
    regular = max(cuadro, key=lambda r: (r["mediana"], r["meses_pos"], -r["apuestas"]))
    print(
        f"regular {regular['nombre']} {regular['cuantos']} mirados={regular['mirados']} "
        f"apuestas={regular['apuestas']} mediana={regular['mediana']} "
        f"rinde={regular['rinde']} meses+={regular['meses_pos']}/{regular['meses']} "
        f"saldo={regular['saldo']} sin6={regular['sin6']} sin5={regular['sin5']}",
        flush=True,
    )
    return {
        "sorteos": n,
        "desde": draws[start][0].isoformat(),
        "hasta": draws[-1][0].isoformat(),
        "mirados_min": miradas[0],
        "mirados_max": miradas[-1],
        "plazos": plazos,
        "pools": pools,
        "mejores": [mejores[PLAZO]],
        "tramos": {"reglas": reglas, "series": series_tramo},
        "cuadro": cuadro,
        "uno_saldo": uno_saldo,
        "uno_sin": uno_sin,
        "en_positivo": len(pools),
        "probadas": len(modos) * len(cfgs) * len(miradas),
        "rentable": False,
    }


def noche(draws, window=120):
    """El criterio: un boleto con los 6 calientes de los últimos 120 sorteos."""
    i = len(draws)
    prefix = [[0] * (N + 1) for _ in range(i + 1)]
    pos = [[] for _ in range(N + 1)]
    for j, (_d, nums) in enumerate(draws):
        nxt = prefix[j + 1]
        nxt[:] = prefix[j]
        for num in nums:
            nxt[num] += 1
            pos[num].append(j)
    counts = [prefix[i][n] - prefix[i - window][n] for n in range(N + 1)]
    gaps = _huecos(pos, i, window)
    return {
        "fecha": cf.nextBonolotoDate(draws[-1][0]).isoformat(),
        "desde": draws[i - window][0].isoformat(),
        "hasta": draws[-1][0].isoformat(),
        "mirados": window,
        "calientes": _topk(counts, k=12, reverse=True),
        "frios": _topk(counts, k=12, reverse=False),
        "retrasados": _topk(gaps, k=12, reverse=True),
    }


VENTANAS_CLARAS = (80, 120, 200, 320)
_TOP_CLARO = 12


def _hot(segment):
    score = [0] * (N + 1)
    for _d, nums in segment:
        for n in nums:
            score[n] += 1
    return sorted(range(1, N + 1), key=lambda n: (-score[n], n))


def _forma(nums):
    """3 pares y 3 impares, no seis seguidos, y al menos tres decenas."""
    if sum(n % 2 for n in nums) != 3:
        return False
    s = tuple(sorted(nums))
    if all(s[i + 1] - s[i] == 1 for i in range(5)):
        return False
    return len({(n - 1) // 10 for n in s}) >= 3


def _buscar(pool, vistos, debe=(), con=None, solape=None):
    if len(pool) < 6:
        return None
    pos = {n: i for i, n in enumerate(pool)}
    best = None
    best_heat = None
    for comb in combinations(pool, 6):
        if debe and any(n not in comb for n in debe):
            continue
        nums = tuple(sorted(comb))
        if nums in vistos or not _forma(nums):
            continue
        if con is not None and len(set(nums) & set(con)) != solape:
            continue
        heat = sum(pos[n] for n in nums)
        if best_heat is None or heat < best_heat:
            best_heat = heat
            best = nums
    return best


def _armar(modo, ranked, vistos):
    """una: 6 calientes con las reglas. dos: la segunda incluye el 7.º. nueve: dos boletos de los 9, con 3 en común."""
    primera = _buscar(ranked[:_TOP_CLARO], vistos) or _buscar(ranked[:18], vistos)
    if not primera:
        return []
    if modo == "una":
        return [primera]
    if modo == "dos":
        septimo = next(n for n in ranked if n not in primera)
        pool = list(dict.fromkeys([septimo] + ranked[:_TOP_CLARO]))[:_TOP_CLARO]
        segunda = _buscar(pool, vistos, debe=(septimo,))
        if not segunda or segunda == primera:
            return []
        return [primera, segunda]
    pool = ranked[:9]
    base = _buscar(pool, vistos) or primera
    segunda = _buscar(pool, vistos, con=base, solape=3)
    if not segunda:
        return []
    return [base, segunda]


def _euros_de(tabla, boletos, claves):
    sub = {k: boletos.get(k, 0) for k in claves}
    got = _euros(tabla, sub)
    return 0.0 if got is None else got


def _fila_clara(modo, nombre, mirados, meses):
    rindes = sorted(m["premio"] / m["gasto"] for m in meses if m["gasto"])
    bajos = sorted(m["bajo"] / m["gasto"] for m in meses if m["gasto"])
    premio = sum(m["premio"] for m in meses)
    bajo = sum(m["bajo"] for m in meses)
    gasto = sum(m["gasto"] for m in meses)
    mid = len(rindes) // 2
    return {
        "modo": modo,
        "nombre": nombre,
        "mirados": mirados,
        "apuestas": 1 if modo == "una" else 2,
        "mediana": round(rindes[mid], 4) if rindes else 0,
        "mediana_baja": round(bajos[mid], 4) if bajos else 0,
        "rinde": round(premio / gasto, 4) if gasto else 0,
        "rinde_bajo": round(bajo / gasto, 4) if gasto else 0,
        "saldo": round(premio - gasto, 2),
        "saldo_bajo": round(bajo - gasto, 2),
        "gasto": round(gasto, 2),
        "meses": len(rindes),
        "c3": sum(m["c3"] for m in meses),
        "c4": sum(m["c4"] for m in meses),
        "c5": sum(m["c5"] for m in meses),
        "c6": sum(m["c6"] for m in meses),
    }


def _claro(draws, comps, premios):
    """1 frente a 2 apuestas. El mes de en medio, sin el dinero del 5 y del 6."""
    modos = (("una", "Una apuesta"), ("dos", "Dos, con el 7.º"), ("nueve", "Dos, de nueve números"))
    start = max(max(VENTANAS_CLARAS), len(draws) - HISTORIA)
    seen = {tuple(sorted(nums)) for _d, nums in draws[:start]}
    meses = {(modo, ventana): {} for modo, _n in modos for ventana in VENTANAS_CLARAS}
    saltos = Counter()
    for i in range(start, len(draws)):
        d, nums = draws[i]
        tabla = premios.get(d.isoformat())
        if tabla:
            key = (d.year, d.month)
            for ventana in VENTANAS_CLARAS:
                ranked = _hot(draws[i - ventana:i])
                for modo, _nombre in modos:
                    tickets = _armar(modo, ranked, seen)
                    if not tickets:
                        saltos[(modo, ventana)] += 1
                        continue
                    _t5, _t6, bol = _boletos(tickets, nums, comps.get(d))
                    box = meses[(modo, ventana)].setdefault(
                        key, {"premio": 0.0, "bajo": 0.0, "gasto": 0.0, "c3": 0, "c4": 0, "c5": 0, "c6": 0}
                    )
                    box["premio"] += _euros_de(tabla, bol, ("3", "4", "5", "5c", "6"))
                    box["bajo"] += _euros_de(tabla, bol, ("3", "4"))
                    box["gasto"] += 0.50 * len(tickets)
                    box["c3"] += bol["3"]
                    box["c4"] += bol["4"]
                    box["c5"] += bol["5"] + bol["5c"]
                    box["c6"] += bol["6"]
        seen.add(tuple(sorted(nums)))
    filas = []
    for modo, nombre in modos:
        for ventana in VENTANAS_CLARAS:
            filas.append(_fila_clara(modo, nombre, ventana, list(meses[(modo, ventana)].values())))
    if saltos:
        print("dias sin boleto", dict(saltos))
    return filas


def _euro_es(n):
    return f"{n:.2f}".replace(".", ",")


def _detalle_boleto(nums):
    impares = sum(n % 2 for n in nums)
    dec = []
    for d in sorted({(n - 1) // 10 for n in nums}):
        dec.append("1-9" if d == 0 else f"{d * 10}-{d * 10 + 9}")
    return f"{impares} impares y {6 - impares} pares · decenas {', '.join(dec)}"


def build():
    draws, reins, orden, comps = load()
    premios = cargar_premios([d for d, _n in draws[-HISTORIA:]])
    oe = odd_even(draws)
    rein = reintegro_stats(reins)
    filas = _claro(draws, comps, premios)
    last = draws[-1][0]
    nxt = cf.nextBonolotoDate(last)
    por_modo = {}
    for fila in filas:
        por_modo.setdefault(fila["modo"], []).append(fila["rinde_bajo"])
    media = {m: sum(v) / len(v) for m, v in por_modo.items()}
    dos_modo = "dos" if media["dos"] >= media["nueve"] else "nueve"
    modo = dos_modo if media[dos_modo] > media["una"] + 0.02 else "una"
    candidatas = [f for f in filas if f["modo"] == modo]
    elegida = max(candidatas, key=lambda f: (f["rinde_bajo"], -abs(f["mirados"] - 120)))
    ranked = _hot(draws[-elegida["mirados"]:])
    vistos = {tuple(sorted(nums)) for _d, nums in draws}
    tickets = _armar(modo, ranked, vistos)
    cruda = _buscar(ranked[:_TOP_CLARO], set())
    septimo = next(n for n in ranked if n not in (tickets[0] if tickets else ()))
    forma3 = next(r for r in oe if r["impares"] == 3)
    top_rein = max(rein["filas"], key=lambda r: (r["count"], -r["n"]))
    reps = _repeats(draws)
    visibles = [f for f in filas if f["modo"] in ("una", dos_modo)]
    for f in visibles:
        f["elegida"] = f["modo"] == elegida["modo"] and f["mirados"] == elegida["mirados"]
    if modo == "una":
        titulo = "Una apuesta. La segunda no devuelve más."
    else:
        titulo = "Dos apuestas. Devuelven más por euro que una."
    dos_txt = "el 7.º en la segunda" if dos_modo == "dos" else "dos boletos sacados de los 9 más calientes"
    dos_media = _euro_es(media[dos_modo])
    if media["nueve"] + 0.02 < media["dos"]:
        cola = "por debajo de meter solo el 7.º."
    elif media["dos"] + 0.02 < media["nueve"]:
        cola = "por encima de meter solo el 7.º."
    else:
        cola = "parecido a meter solo el 7.º."
    cierre = "no mejora, y pagas el doble." if modo == "una" else "deja más por euro, así que hoy hay dos boletos."
    porque = (
        f"Una apuesta son 6 números y cuesta 0,50 €. "
        f"De 80, 120, 200 y 320 sorteos mirados, devuelve {_euro_es(media['una'])} € por euro. "
        f"Dos apuestas ({dos_txt}) devuelven {dos_media} € por euro: {cierre} "
        f"El boleto de hoy mira {elegida['mirados']}, la mirada que más devolvió "
        f"({_euro_es(elegida['rinde_bajo'])} € por euro), con {elegida['c3']} premios de 3 y {elegida['c4']} premios de 4. "
        f"Saldo de esa fila: {_euro_es(elegida['saldo_bajo'])} €. "
    )
    if elegida["c5"] == 0 and elegida["c6"] == 0:
        porque += "No hubo ningún 5 ni ningún 6, así que esto no sale de un premio gordo. "
    else:
        porque += (
            f"Con el 5 y el 6 el saldo sería {_euro_es(elegida['saldo'])} € "
            f"({elegida['c5']} cincos y {elegida['c6']} seises). Sin ellos, no se sostiene. "
        )
    porque += (
        f"Partir nueve números, del 1 al 6 y del 1 al 3 con el 7 al 9, devuelve {_euro_es(media['nueve'])} € por euro, {cola} "
        "Ninguna recupera el euro. Esto es la menos mala, no un sistema."
    )
    boletos = []
    for i, nums in enumerate(tickets, 1):
        boletos.append({
            "nombre": "Apuesta" if len(tickets) == 1 else f"Apuesta {i}",
            "numeros": list(nums),
            "detalle": _detalle_boleto(nums),
        })
    payload = {
        "juego": "Bonoloto",
        "fuente": "Hojas públicas Lotoideas"
        + (" y las fichas de los días que la hoja aún no traía" if orden.get("rellenos") else "")
        + (" y el tablero en vivo del último hueco" if orden.get("en_vivo") else ""),
        "desde": draws[0][0].isoformat(),
        "hasta": last.isoformat(),
        "proximo": nxt.isoformat(),
        "sorteos": len(draws),
        "titulo": titulo,
        "porque": porque.strip(),
        "reintegro": {
            "n": top_rein["n"],
            "nota": f"Reintegro: el {top_rein['n']}, si te obligan a marcar uno. La urna va del 0 al 9 y la diferencia cabe en el azar. Cualquiera vale.",
        },
        "complementario": (
            f"El complementario no se marca. Sale del bombo, entre los que no han salido. "
            f"Solo paga si ya tienes 5 y tu sexto número es ese. "
            f"El siguiente caliente, el {septimo}, "
            + ("va en la segunda apuesta." if len(tickets) == 2 else "no se juega: la segunda apuesta no compensa.")
        ),
        "septimo": septimo,
        "boletos": boletos,
        "descarto_repetida": bool(cruda and cruda in vistos),
        "reglas": [
            f"3 impares y 3 pares. Es la forma más común: el {round(forma3['obs'] * 100)} % de los sorteos, igual que el combinatorio.",
            "No van seis números seguidos. Eso casi no ha salido. Una pareja seguida sale más o menos la mitad de las veces, así que no la busco ni la evito.",
            "Al menos tres decenas, para no juntar los seis en el mismo bloque.",
            f"En {len(draws)} sorteos, una combinación de seis se ha repetido {reps} veces. El boleto de hoy no es ninguna de las ya salidas.",
        ],
        "tabla": visibles,
        "elegida": {"modo": elegida["modo"], "mirados": elegida["mirados"]},
    }
    out = Path("data")
    out.mkdir(exist_ok=True)
    (out / "bonoloto.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    nums = " | ".join(" ".join(map(str, b["numeros"])) for b in boletos)
    print(
        f"sorteos={payload['sorteos']} {payload['desde']}..{payload['hasta']} proximo={payload['proximo']} "
        f"{titulo} mirados={elegida['mirados']} {nums}"
    )
    for f in filas:
        print(
            f"{f['modo']:5} mirados={f['mirados']:3} por_euro={f['rinde_bajo']:.3f} "
            f"mediana={f['mediana_baja']:.3f} saldo={f['saldo_bajo']:.0f} "
            f"c3={f['c3']} c4={f['c4']} c5={f['c5']} c6={f['c6']}"
        )


def _repeats(draws):
    c = Counter(nums for _d, nums in draws)
    return sum(1 for v in c.values() if v > 1)


if __name__ == "__main__":
    build()
