---
name: web-design
description: Any page, landing, dashboard or visual change (HTML/CSS/JSX), or "diseño", "web bonita", "UI", "landing", "panel". Google Stitch designs; you integrate. Always.
---

# Web design

Google Stitch designs. You do not design the screen yourself.

Any new page, restyle, dashboard or visual change goes through Stitch. A typo, a data field or wiring that does not change layout does not.

## Style = `DESIGN.md`

Missing: create it (≤ 20 lines) with this default plus the user's wishes:

`Expresivo y verde. Titular display grande (Public Sans, muy negro). La cifra principal va sobre un bloque sólido #18E667 con texto #032612; el botón de la acción principal es #0A3D22 con texto claro. Tarjetas claras, esquinas amplias. Cada sección cambia de gesto: tamaño de tipo, bloque de color o imagen a sangre. Entre una y la siguiente hay transición (el bloque entra, el titular escala). Una imagen o un motivo decorativo por sección, para que un ranking no sea una lista plana. La fila se lee: nombre, una descripción y miniatura. No repitas el mismo truco en cada bloque. Sin objeto 3D ni WebGL de relleno. Una sola acción principal. Mobile first: el mismo layout nace a 360px y escala a 768 y 1280, sin una pantalla aparte de escritorio. Interruptor sol/luna: el punto se desliza al lado activo (por defecto el del sistema, y se recuerda). No un botón con la palabra Claro u Oscuro. Si hay una ruta o lugares, un solo mapa (OpenStreetMap, sin API de pago). Cada vez que se cita un lugar con web real, el nombre es el enlace a esa web; no inventes la URL. Un nombre que solo es una web enlaza a ella. En un ranking, cada fila tiene Ver mapa (lleva al mapa, resalta el punto y abre el popup) y Ruta a Google Maps (destino = ese punto; el origen lo elige quien viaja). El popup muestra el nombre (enlace a la web si la hay) y el botón Ruta. Ver mapa y Ruta, en la fila y en el popup, son fondo #18E667 y texto #032612 en claro y en oscuro; si salen grises o con texto blanco, se corrigen. Nada de degradados morados, arcoíris, glassmorphism, emojis como iconos, todo centrado ni plantillas genéricas.`

Keep the Stitch project id, design-system asset id and screen ids here.

Data page (prices, stats, rankings, metrics, panel): also put the anatomy from `dashboard.md` in this folder into the Stitch prompt (hero KPI, comparisons, charts, source line).

## Steps

1. Reuse the project id in `DESIGN.md`. Missing: `create_project` once, on MCP `stitch` (reads and `create_project` are fast; call them there).
2. Design system once per repo (reuse the asset id). Through `stitch_long`, tool `create_design_system`:
   - `colorMode: LIGHT`, `colorVariant: EXPRESSIVE`, `roundness: ROUND_TWELVE`
   - `customColor`: the accent hex from `DESIGN.md`, or `#18E667` if none
   - `bodyFont` and `headlineFont`: `PUBLIC_SANS`
   - `displayName`: the project name
3. New screen: `stitch_long` with tool `generate_screen_from_text`. Visual change to an existing screen: tool `edit_screens` (same screen id). Arguments:
   - `projectId`, `prompt` (purpose, real copy from the repo, the `DESIGN.md` line, and for data the `dashboard.md` anatomy)
   - `deviceType: MOBILE`, `modelId: GEMINI_3_8_FLASH` (the proxy forces both)
   - `designSystem`: `assets/<id>` from step 2
   - The prompt must say: the layout starts at 360px and the same structure scales to tablet and desktop; each section changes gesture (type size, color block, or full-bleed image) and the next section enters with a transition; a ranking row keeps the name, one description and a thumbnail; do not repeat the same card; if the page is a route, a trip, or places, include one map with those points and nothing else. No map when location is not the point. Map tiles: OpenStreetMap, no paid key. If there is a ranking of places, each row shows the name plus Ver mapa and Ruta; you wire those controls after download. Ver mapa and Ruta (row and popup) are solid #18E667 with text #032612 in both themes, never gray with white text. Do not invent URLs. Do not generate DESKTOP, TABLET or AGNOSTIC screens. You add the marker popup after download; do not ask Stitch to draw Google Maps.
4. If `stitch_long` returns `status: running`, call `stitch_wait(job)` until the result. Do not call generate/edit/variants on MCP `stitch` directly: those die at 60 s. Only past 20 min is a failure: report FALLO.
5. `get_screen` on MCP `stitch`. Download `htmlCode.downloadUrl` and publish that file as the page (`index.html`, or the entry the site actually serves). Do not redraw the layout, type, or colors in your own CSS. You may only fix a broken link, replace invented copy with the repo's real text, add the other color mode, wire places, and add the section transition in CSS (the block enters and the title scales as it comes into view; off under `prefers-reduced-motion`). No 3D object and no WebGL. Always add a sun/moon switch: a pill, thumb slides to the active side, sun sets light and moon sets dark. First visit follows the system, the choice is stored, both themes keep the same accent. Not a text button labeled Claro or Oscuro. `prefers-color-scheme` alone is not enough. Wire places after download; Stitch does not draw this behavior. No Maps API key, no embedded Google map; tiles are OpenStreetMap. Every time a place with a real website is named, the name is an `<a href>` to that exact URL. Do not invent a URL. A name that is only a website, and is not on the map, links to that site. A place with no website keeps a plain name that scrolls to the map. In a ranking, each row has **Ver mapa** (scrolls to the map, highlights that point, opens its popup) and **Ruta** (`https://www.google.com/maps/dir/?api=1&destination=<lat>,<lng>`, `target="_blank"`, `rel="noopener noreferrer"`). Destination is that point; Google Maps asks for the origin. The popup shows the same name link and **Ruta**. Row and popup controls use background `#18E667` and text `#032612` in light and dark; override Stitch if it painted them gray or with white text. If the site is generated (`build.py`), port this HTML into the template the build publishes. A second unused page does not count.
6. Write the ids back to `DESIGN.md`.
7. Stitch missing or failing: say so, then hand-build from `DESIGN.md`. Do not invent a second visual system.

## Before HECHO

360 / 768 / 1440 without horizontal scroll, WCAG AA, visible focus, `alt` on images, `prefers-reduced-motion`. Data page: the h1 states the finding, charts have a caption, no incomparable numbers ranked. Look with skill `verify`.
