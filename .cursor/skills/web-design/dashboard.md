# Dashboard brief (for the Stitch prompt)

Read when the page shows numbers that change (prices, stats, rankings, metrics). Put this anatomy in the `generate_screen_from_text` / `edit_screens` prompt. Stitch designs it; you integrate. A ranking is a sequence, not one dense table: a hero gesture, then each case as a row with name, one description and a thumbnail, then the map or chart at another scale and color. Each section changes gesture. Do not repeat the same card. Drop a chart that only repeats the finding. If the reader needs to see where something is (a route, stops, cities), one map with those points, OpenStreetMap, no paid key. Every time a place with a real website is named, the name links to that site; do not invent a URL. A name that is only a website links to that site. A place with no website keeps a plain name that scrolls to the map. Each ranking row has Ver mapa (scrolls to the map, highlights that point, opens the popup) and Ruta (Google Maps directions, destination = that point; the traveler picks the origin). The popup name links to the website when there is one, and repeats Ruta. Ver mapa and Ruta (row and popup) are solid `#18E667` with text `#032612` in light and dark; override a gray fill or white label. Otherwise no map. Reference result: `tarifas-espana` (`build.py`, `dashboard()`).

## Anatomy (always)

1. **Header**: kicker (scope · date) + h1 that states the finding, not the topic ("La luz sale más barata de 14 a 16 h", not "Precios de la luz") + chip with source/update.
2. **KPI row, 3-4 cards**: first one is `hero` (solid `#18E667`, big number, ink `#032612`). The main button is `#0A3D22` with light text. Each card: icon, label, number with small unit, and one comparison pill (`▲ +0,034 frente a la media` / `▼`). Arrow + words, never color alone.
3. **Two charts minimum**: one distribution (bars by hour/category, colored by level) and one comparison (line today vs reference, ranking, cost curve). Each card has a title that says what to read, a one-line caption, a legend.
4. **Controls** that change the numbers (slider, select, segmented Hoy/7 días). Update KPIs and charts together.
5. **Footer line**: source, date, what is NOT included.

Semantic scale: good `#0F7B5F`/`#4FD1A5`, warn `#D97706`/`#FBBF24`, bad `#C2410C`/`#F87171` (light/dark). Level = rank, not magic thresholds (e.g. 8 lowest good, 6 highest bad). Neutral series: `--muted`. Never use good/bad on data that is not comparable (different scope, different year): use one neutral accent and say why.

Layout: sidebar tabs on desktop (Resumen, then sections), top scrollable tabs < 768px. `grid` with `auto-fit minmax(14rem,1fr)` for KPIs, `1.6fr / 1fr` for the chart row, one column < 900px. Cards: `radius 14px`, 1px line, soft shadow in light mode only.

## CSS

```css
:root{--good:#0F7B5F;--warn:#D97706;--bad:#C2410C;--shadow:0 1px 2px rgb(0 0 0/.05),0 8px 24px rgb(0 0 0/.05)}
@media(prefers-color-scheme:dark){:root{--good:#4FD1A5;--warn:#FBBF24;--bad:#F87171;--shadow:none}}
.kpis{display:grid;gap:.9rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,14.5rem),1fr))}
.kpi{display:flex;flex-direction:column;gap:.2rem;padding:1.1rem 1.2rem;background:var(--surface);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
.kpi .ic{display:grid;place-items:center;width:2.25rem;height:2.25rem;border-radius:10px;background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)}
.kpi small{color:var(--muted);font-size:.875rem}
.kpi strong{font-size:clamp(1.7rem,1.2rem + 1.6vw,2.4rem);font-weight:650;letter-spacing:-.035em;line-height:1.05;font-variant-numeric:tabular-nums}
.kpi strong span{font-size:.5em;font-weight:500;color:var(--muted);margin-left:.2rem}
.kpi.hero{background:var(--accent);border-color:var(--accent);color:var(--on)}
.kpi.hero small,.kpi.hero strong span{color:inherit;opacity:.85}
.delta{align-self:flex-start;margin-top:.45rem;padding:.2rem .6rem;border-radius:99px;font-size:.8125rem;font-weight:600}
.delta.up{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
.delta.down{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.hero .delta{background:rgb(255 255 255/.2);color:inherit}
.row{display:grid;gap:.9rem;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr)}
@media(max-width:900px){.row{grid-template-columns:minmax(0,1fr)}}
.card{min-width:0;padding:1.15rem 1.25rem;background:var(--surface);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
.card h2{margin:0 0 .15rem;font-size:1.05rem}.card .cap{margin:0 0 .9rem;color:var(--muted);font-size:.875rem}
.chart svg{display:block;width:100%;height:auto}
.chart .grid{stroke:var(--line)}.chart .ax{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.b-good{fill:var(--good)}.b-mid{fill:var(--warn)}.b-bad{fill:var(--bad)}
.chart .ln{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.chart .ln.main{stroke:var(--accent)}.chart .ln.ref{stroke:var(--muted);stroke-dasharray:5 5;stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1rem;margin-top:.7rem;color:var(--muted);font-size:.8125rem}
.legend span::before{content:"";display:inline-block;width:.65rem;height:.65rem;margin-right:.35rem;border-radius:3px;background:var(--c);vertical-align:-1px}
.rk{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(4rem,1fr) auto;gap:.35rem .6rem;align-items:center;padding:.45rem;border-radius:8px;font-size:.9rem}
.rk i{display:block;height:.8rem;border-radius:99px;background:color-mix(in srgb,var(--accent) 55%,var(--line))}
.rk b{text-align:right;font-variant-numeric:tabular-nums}
.rk .acts{grid-column:1/-1;display:flex;gap:.35rem}
a.go,button.go{padding:.15rem .55rem;border-radius:99px;border:0;background:#18E667;color:#032612;font:inherit;font-size:.75rem;font-weight:650;text-decoration:none;cursor:pointer}
```

## Charts without libraries (inline SVG, no CDN)

```js
const fmt=(n,d=3)=>n.toLocaleString("es-ES",{minimumFractionDigits:d,maximumFractionDigits:d});
function levels(v,good=8,bad=6){const i=v.map((_,k)=>k).sort((a,b)=>v[a]-v[b]),l=v.map(()=>"mid");i.slice(0,good).forEach(k=>l[k]="good");i.slice(-bad).forEach(k=>l[k]="bad");return l}
function bars(el,labels,v,texts){ // el: .chart; v: numbers; texts: tooltip strings
  const W=720,H=270,L=44,R=8,T=14,B=28,max=Math.max(...v)*1.12,bw=(W-L-R)/v.length,lv=levels(v),sy=x=>T+(H-T-B)*(1-x/max);let o="";
  for(let i=0;i<=4;i++){const y=sy(max*i/4);o+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" class="grid"/><text x="${L-6}" y="${y+4}" class="ax" text-anchor="end">${fmt(max*i/4,2)}</text>`}
  v.forEach((x,i)=>{const px=L+i*bw+bw*.14,h=(H-T-B)*x/max;o+=`<rect x="${px}" y="${H-B-h}" width="${bw*.72}" height="${h}" rx="3" class="b-${lv[i]}"><title>${labels[i]} · ${texts[i]}</title></rect>`;if(i%3===0)o+=`<text x="${px+bw*.36}" y="${H-8}" class="ax" text-anchor="middle">${labels[i]}</text>`});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gráfico de barras">${o}</svg>`}
function lines(el,labels,main,ref){ // two series, same scale
  const W=720,H=250,L=44,R=10,T=14,B=28,max=Math.max(...main,...(ref||[0]))*1.12,sx=i=>L+(W-L-R)*i/(main.length-1),sy=x=>T+(H-T-B)*(1-x/max);
  const d=a=>a.map((x,i)=>(i?"L":"M")+sx(i).toFixed(1)+" "+sy(x).toFixed(1)).join(" ");let o="";
  for(let i=0;i<=4;i++){const y=sy(max*i/4);o+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" class="grid"/><text x="${L-6}" y="${y+4}" class="ax" text-anchor="end">${fmt(max*i/4,2)}</text>`}
  if(ref)o+=`<path d="${d(ref)}" class="ln ref"/>`;o+=`<path d="${d(main)}" class="ln main"/>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Gráfico de líneas">${o}</svg>`}
```

Ranking list: sort ascending, one `.rk` row per item (`<a class="nm" href="URL">name</a><i style="width:N%"></i><b>value</b><span class="acts"><button type="button" class="go" data-map="id">Ver mapa</button><a class="go" href="https://www.google.com/maps/dir/?api=1&destination=LAT,LNG" target="_blank" rel="noopener noreferrer">Ruta</a></span>`). Omit the name `href` when there is no real URL (the name then scrolls to the map). Omit Ver mapa and Ruta when the row is not a place. Highlight the selected row with `box-shadow:inset 0 0 0 1px var(--accent)`.

## Data honesty (non-negotiable)

- Show scope, year and source next to every number; "sin dato" beats a guess.
- Do not rank or color good/bad things measured differently (different taxes, different concepts). Say what each includes.
- Normalize before comparing (per month, per unit, same VAT); state the normalization.
- Check outliers against the source before they sit on the home page.
