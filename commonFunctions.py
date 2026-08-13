import datetime as dt
import colorama

# ------------------------------------------------------------------------------------

# Format percentages (example: 2.3456789 --> 134.5679 %)
def formatPercentages(val):
    return round((val - 1) * 100, 4)


# Convert boolean values to int (True -> 1 and False -> 0). Else (just in case) -> -1
def boolToInt(val):
    if val == "True" or val == True:
        res = 1
    elif val == "False" or val == False:
        res = 0
    else:
        res = -1

    return res


# Print any data with the datetime to debug properly (really useful)
def printInfo(desc, color=""):
    print(f"{dt.datetime.now()} // {color}{desc}{colorama.Fore.RESET}")


def nextBonolotoDate(last_date):
    """Bonoloto draws Mon-Sat. Skip Sunday."""
    d = last_date + dt.timedelta(days=1)
    while d.weekday() == 6:
        d += dt.timedelta(days=1)
    return d


def topUniqueNumbers(counts, k=6, low=1, high=49):
    """Pick k unique ints from a value_counts-like Series, then fill  low..high."""
    picked = []
    for n in counts.index:
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if low <= n <= high and n not in picked:
            picked.append(n)
        if len(picked) == k:
            break
    for n in range(low, high + 1):
        if len(picked) >= k:
            break
        if n not in picked:
            picked.append(n)
    return sorted(picked[:k])
