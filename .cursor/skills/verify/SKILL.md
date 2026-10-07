---
name: verify
description: See the web you built before saying HECHO. Headless Chrome/Edge screenshots on mobile and desktop, light and dark; look at them and fix what looks off. Use after any UI change.
---

# Verify (UI)

Checking code is not checking the page. Look at it.

## 1. Screenshots

Browser: `chrome`, `msedge` or `chromium` (Windows: `"$env:ProgramFiles\Google\Chrome\Application\chrome.exe"` or `msedge`). Write PNGs outside the repo (temp dir), never commit them.

```sh
B=google-chrome   # or msedge / chromium / chrome.exe path
URL=file:///abs/path/index.html   # or http://localhost:PORT for dev servers
for s in "390,1600 movil" "1440,1000 escritorio"; do set -- $s
  "$B" --headless --no-sandbox --hide-scrollbars --window-size=$1 \
    --blink-settings=preferredColorScheme=1 --screenshot="$TMP/$2-claro.png" "$URL"
  "$B" --headless --no-sandbox --hide-scrollbars --window-size=$1 \
    --blink-settings=preferredColorScheme=0 --screenshot="$TMP/$2-oscuro.png" "$URL"
done
```

- `preferredColorScheme`: `1` light, `0` dark. Height = how much of the page you see; raise it for long pages.
- If the process doesn't exit, kill it after ~20 s: the PNG is already written. Avoid `--virtual-time-budget` with infinite animations (it hangs).

## 2. Look

Open every PNG and check:

- No horizontal overflow or cut text on mobile (the whole layout fits 390px).
- Hierarchy reads at a glance; spacing is even; nothing overlaps.
- Dark mode has real contrast (no dark-on-dark, no pure white glare).
- Fonts actually loaded (not a fallback serif/sans), icons render, images not broken.

## 3. Fix and repeat

Fix what looks off, take the screenshots again, and only then say HECHO. No browser available: say so in the reply; never claim you verified visually.
