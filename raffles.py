"""Lotoideas historic CSVs (same source as the Bonoloto Google Sheet) + sqlite long rows."""
import datetime as dt
import pandas as pd

def _csv(pacx, gids=None):
    base = f"https://docs.google.com/spreadsheets/d/e/{pacx}/pub"
    if not gids:
        return [base + "?output=csv"]
    return [f"{base}?gid={g}&single=true&output=csv" for g in gids]


# Number games drawn ~every 1-3 days (plus Sunday Gordo). Lototurf 6/12 skipped: tiny urn.
GAMES = {
    "Bonoloto": {
        "n": 49, "w": 6, "weekdays": (0, 1, 2, 3, 4, 5),
        "csv": _csv("2PACX-1vQALTRaLDFfhXOAQmeONPqmFKm9yOiQ4W97rhWgR41BZ7czFsjK5YktD6fnETKHGB9YUnyQ4XBSbhZx", (0, 1)),
        "extra_types": ("Complementario", "Reintegro"),
    },
    "Primitiva": {
        "n": 49, "w": 6, "weekdays": (3, 5),
        "csv": _csv("2PACX-1vTov1BuA0nkVGTS48arpPFkc9cG7B40Xi3BfY6iqcWTrMwCBg5b50-WwvnvaR6mxvFHbDBtYFKg5IsJ", (0, 1)),
        "extra_types": ("Complementario", "Reintegro"),
    },
    "Euromillones": {
        "n": 50, "w": 5, "weekdays": (1, 4),
        "csv": _csv("2PACX-1vRy91wfK2JteoMi1ZOhGm0D1RKJfDTbEOj6rfnrB6-X7n2Q1nfFwBZBpcivHRdg3pSwxSQgLA3KpW7v"),
        "extra_types": ("Estrella1", "Estrella2"),
        "extra": {
            "name": "Euromillones-estrellas",
            "n": 12, "w": 2, "weekdays": (1, 4),
            "from": dt.date(2011, 5, 10),  # stars were 1-9 before this
        },
    },
    "ElGordo": {
        "n": 54, "w": 5, "weekdays": (6,),
        "csv": _csv("2PACX-1vRR678qNlN_3p2dAxRG0LULS6EYmBbEmpfVhCEmsYky6eiuEH3o_mCRc4c2_EevPru_3BJfSV0QwpG8"),
        "extra_types": ("Clave",),
    },
    "Eurodreams": {
        "n": 40, "w": 6, "weekdays": (0, 3),
        "csv": _csv("2PACX-1vTZzm-CTUj3li4EdfW1ImthPdc0AGIymq8tbuwPiqjW0OL4F1MWO5G6PfPEtNvLJcY8MpJo4apayTip"),
        "extra_types": ("Dream",),
    },
}


def load_rows(spec):
    """[(date, main_tuple, extra_tuple), ...] sorted. Main balls validated 1..n."""
    n, w = spec["n"], spec["w"]
    frames = []
    for url in spec["csv"]:
        df = pd.read_csv(url)
        date = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce").reset_index(drop=True)
        nums = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        frames.append(pd.concat([date.rename("RESULT_DATE"), nums], axis=1))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["RESULT_DATE"]).drop_duplicates("RESULT_DATE")
    out = []
    for _, row in df.iterrows():
        d = row["RESULT_DATE"]
        d = d.date() if hasattr(d, "date") else d
        vals = []
        for x in row.iloc[1:]:
            if pd.isna(x):
                continue
            try:
                vals.append(int(x))
            except (TypeError, ValueError):
                continue
        main = tuple(vals[:w])
        extra = tuple(vals[w:])
        if len(main) != w or len(set(main)) != w:
            continue
        if not all(1 <= x <= n for x in main):
            continue
        out.append((d, main, extra))
    return sorted(out, key=lambda x: x[0])


def load_draws(spec):
    """Main series + optional extra series (e.g. Euromillones stars)."""
    rows = load_rows(spec)
    mains = [(d, m) for d, m, _e in rows]
    extra_draws = []
    extra = spec.get("extra")
    if extra:
        en, ew = extra["n"], extra["w"]
        start = extra.get("from")
        for d, _m, e in rows:
            if start and d < start:
                continue
            balls = tuple(e[:ew])
            if len(balls) == ew and len(set(balls)) == ew and all(1 <= x <= en for x in balls):
                extra_draws.append((d, balls))
    return mains, extra_draws


def to_long(spec, raffle):
    """RAFFLE / RESULT_DATE / NUMBER_TYPE / NUMBER rows for sqlite ingest."""
    types = spec.get("extra_types") or ()
    recs = []
    for d, main, extra in load_rows(spec):
        for i, num in enumerate(main, 1):
            recs.append({"RAFFLE": raffle, "RESULT_DATE": d, "NUMBER_TYPE": f"N{i}", "NUMBER": num})
        for t, num in zip(types, extra):
            if abs(int(num)) > 99:  # skip Joker-like ids
                continue
            recs.append({"RAFFLE": raffle, "RESULT_DATE": d, "NUMBER_TYPE": t, "NUMBER": int(num)})
    return pd.DataFrame.from_records(recs)
