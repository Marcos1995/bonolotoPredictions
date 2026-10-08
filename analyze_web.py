"""Bonoloto: patrones históricos y backtest walk-forward contra el azar.

Escribe data/bonoloto.json. Fuente: hojas públicas de raffles.py (mismo origen
que el sqlite local). Sin lookahead: cada regla usa solo sorteos anteriores.
"""
import datetime as dt
import json
import math
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


def _boletos(k, jugados, actual, comp):
    """Cuántos boletos de 6 cobran si se juegan todas las combinaciones del grupo."""
    pool = set(jugados)
    overlap = len(set(actual) & pool)
    non = k - overlap
    c_in = comp is not None and comp in pool
    b = {"3": 0, "4": 0, "5": 0, "5c": 0, "6": 0}
    if overlap >= 6:
        b["6"] = 1
        b["5c"] = 6 if c_in else 0
        b["5"] = 6 * (non - (1 if c_in else 0))
        b["4"] = math.comb(6, 4) * math.comb(non, 2)
        b["3"] = math.comb(6, 3) * math.comb(non, 3)
    elif overlap == 5:
        b["5c"] = 1 if c_in else 0
        b["5"] = non - (1 if c_in else 0)
        b["4"] = 5 * math.comb(non, 2)
        b["3"] = 10 * math.comb(non, 3)
    elif overlap == 4:
        b["4"] = math.comb(non, 2)
        b["3"] = 4 * math.comb(non, 3)
    elif overlap == 3:
        b["3"] = math.comb(non, 3)
    return overlap, b


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


def calientes_ventanas(draws, comps, premios):
    """Un año. Cobertura total: todas las combinaciones de 6 dentro de los k más calientes."""
    last = draws[-1][0]
    idx = next(i for i, (d, _n) in enumerate(draws) if d >= last - dt.timedelta(days=364))
    start = max(idx, 1)
    n = len(draws) - start
    pools = []
    for k in (6, 9, 12):
        apuestas = math.comb(k, 6)
        coste = n * apuestas * 0.50
        ventanas = []
        for window in range(50, 501, 50):
            cobrado = 0.0
            sin_precio = 0
            suma = {"3": 0, "4": 0, "5": 0, "5c": 0, "6": 0}
            dias5 = dias6 = 0
            premios_dias = []
            for i in range(start, len(draws)):
                actual = draws[i][1]
                comp = comps.get(draws[i][0])
                jugados = _pool(draws[:i], window, k)
                overlap, boletos = _boletos(k, jugados, actual, comp)
                if overlap == 5:
                    dias5 += 1
                elif overlap >= 6:
                    dias6 += 1
                for key, cnt in boletos.items():
                    suma[key] += cnt
                pago = _euros(premios.get(draws[i][0].isoformat()), boletos)
                if pago is None:
                    sin_precio += 1
                    continue
                cobrado += pago
                if overlap >= 5:
                    premios_dias.append({
                        "fecha": draws[i][0].isoformat(),
                        "salio": list(actual),
                        "numeros": sorted(jugados),
                        "dentro": overlap,
                        "complementario": comp,
                        "boletos": boletos,
                        "euros": round(pago, 2),
                    })
            ventanas.append({
                "ventana": window,
                "boletos": suma,
                "dias5": dias5,
                "dias6": dias6,
                "cobrado": round(cobrado, 2),
                "coste": round(coste, 2),
                "saldo": round(cobrado - coste, 2),
                "sin_precio": sin_precio,
                "premios": premios_dias,
            })
        ventanas.sort(key=lambda w: w["saldo"], reverse=True)
        pools.append({
            "cuantos": k,
            "apuestas": apuestas,
            "coste_dia": round(apuestas * 0.50, 2),
            "oficial": k <= 11,
            "ventanas": ventanas[:3],
        })
    mejor = max((w for p in pools for w in p["ventanas"]), key=lambda w: w["saldo"])
    return {
        "sorteos": n,
        "desde": draws[start][0].isoformat(),
        "pools": pools,
        "rentable": mejor["saldo"] > 0,
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
        "calientes_ventanas": calientes_ventanas(draws, comps, premios),
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
    for pool in v["pools"]:
        for w in pool["ventanas"]:
            print(
                f"k{pool['cuantos']} w{w['ventana']} dias5={w['dias5']} dias6={w['dias6']} "
                f"boletos={w['boletos']} cobrado={w['cobrado']} saldo={w['saldo']}"
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
