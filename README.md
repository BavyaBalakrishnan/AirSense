<!-- filepath: /Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco/README.md -->
# 🌍 AirSense — Natural-Language Air-Quality Intelligence

Project built on the 
**Gemini ↔ MCP ↔ Prefab** pattern : instead of a
hard-coded agent loop, you **type a sentence** and the LLM picks the tools,
fetches live data from 8 public APIs, and composes a Prefab dashboard on
the fly. No widget names, no API names, no JSON authoring — just natural
English.

> *"Track Delhi, Mumbai, and Bengaluru. Compare them, show today's hourly
> trend, and add a 7-day heatmap for the most polluted city."*
>
> → 4 parallel HTTP fetches + dynamic resolution of "most polluted" + 4
> auto-generated dashboard tabs, in one prompt.

The killer feature: AirSense doesn't stop at *"the air is bad"* — it
cross-references **NASA satellite-detected fires** with **Open-Meteo wind
direction** to tell you **which fires upwind are causing today's smoke**.

---

## What it has
|---|---|
| **Internet** | 8 live data sources — Open-Meteo (3 endpoints), AQICN/CPCB stations, NASA FIRMS satellite fires, Google News RSS, Wikipedia REST, generic webpage scraper |
| **CRUD on a local file** | 4 tools on `eco_log.json` — `build_city_report` / `save_city_report` (Create/Update), `list_log` + `get_city_report` (Read), `remove_city` (Delete) |
| **UI via Prefab** | `show_dashboard()` returns a `PrefabApp` with 5 base tabs; the planner in `talk_eco.py` extends it with custom tabs per prompt |

---

## Architecture (3 layers, no hallucinations)

```
┌─────────────────────────────────────────────────────────────────┐
│  talk_eco.py  (the natural-language REPL)                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────┐   │
│  │ Intent          │ →  │ Planner LLM     │ →  │ Widget     │   │
│  │ classifier      │    │ (sees REAL data │    │ renderer   │   │
│  │ (add/remove/    │    │ from eco_log)   │    │ (Python    │   │
│  │  refresh JSON)  │    │                 │    │ → Prefab)  │   │
│  └─────────────────┘    └─────────────────┘    └────────────┘   │
│         ↓                       ↓                     ↓         │
└─────────│───────────────────────│─────────────────────│─────────┘
          │                       │                     │
          ▼                       ▼                     ▼
    MCP tools              eco_log.json           generated_eco_app.py
    (parallel HTTP)        (CRUD on disk)         (live Prefab UI)
```

**Why three layers?** Tools fetch raw data, the planner composes
*structure* (no values), and the renderer pulls real values from disk.
The LLM never sees raw HTML and never invents numbers. Every figure on
screen traces back to an HTTP response saved in `eco_log.json`.

---

## Live data sources (8 APIs)

| API | Key needed? | Used for |
|---|:-:|---|
| [Open-Meteo Geocoding](https://open-meteo.com/en/docs/geocoding-api) | no | city → lat/lon/timezone |
| [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) | no | current AQI + 168-hour history + 48-hour forecast + 6 pollutants |
| [Open-Meteo Forecast](https://open-meteo.com/en/docs) | no | temp / humidity / UV / wind speed + direction |
| [AQICN](https://aqicn.org/data-platform/token/) | free | official CPCB station-level AQI (second opinion vs interpolated grid) |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) | free MAP_KEY | VIIRS satellite fire detections (last 24h) |
| [Google News RSS](https://news.google.com/rss/search) | no | latest headlines per city |
| Wikipedia REST | no | clean lead-paragraph extracts (no nav chrome) |
| Generic HTML scraper | no | gov bulletins, blog posts, fallback |

---

## All 17 MCP tools

### 🌐 Live data fetch (5)

1. `fetch_aqi_now(city)` — current AQI + band + dominant pollutant + 6 pollutants
2. `fetch_aqi_24h(city)` — 24 hourly AQI + PM2.5, peak/min/avg
3. `fetch_aqi_7d(city)` — 7 days × 24 hours of AQI for the heatmap grid
4. `fetch_weather_now(city)` — temp, max/min, UV, humidity, condition, **wind speed + direction**
5. `build_city_report(city)` — runs 1+2+3+4 **in parallel** and saves to `eco_log.json` (one MCP call per city)

### 💾 Local CRUD on `eco_log.json` (4)

6. `save_city_report(...)` — Create / Update (replaces by city name)
7. `list_log()` — Read full log
8. `get_city_report(city)` — Read one city
9. `remove_city(city)` — Delete

### 🧮 Pure-Python analytics (4 — no internet)

10. `who_breach_summary(city)` — per-pollutant ratios over WHO 24h limits + worst breach
11. `time_of_day_profile(city)` — morning/afternoon/evening/night buckets + safest 3h window
12. `fetch_aqi_forecast(city, hours)` — next 1–48 forecast hours of AQI
13. `recommend_outdoor_window(city, duration_hours)` — lowest-AQI window in the next 24h

### 🛰️ Causal context (4)

14. `fetch_news_for(city, query, limit)` — Google News headlines (no API key)
15. `fetch_webpage(url, max_chars)` — generic page scraper; auto-uses Wikipedia REST summary for `*.wikipedia.org/wiki/...` URLs
16. `fetch_aqicn(city)` — official CPCB station data via AQICN (needs `AQICN_TOKEN`)
17. **`fire_hotspots_near(city, radius_km, days)`** — NASA FIRMS satellite fires in a bbox around the city, **cross-referenced with wind direction** to flag UPWIND vs DOWNWIND

### 🎨 UI (1)

* `show_dashboard()` — `@mcp.tool(app=True)` returning a `PrefabApp` with 5 tabs: **Right Now / Pollutants / 24h Trend / Hour Pattern / Compare Cities**

---

## How Prefab works (in 3 lines)

Prefab is **Python-as-a-UI-framework**. You write `Card`, `H1`, `BarChart`,
`Tabs`, `Ring`, `Badge` as nested `with` blocks — Prefab serves a tiny dev
server that hot-reloads the file and pushes updates to the browser. So
the LLM never writes HTML or React — it emits ~40 lines of Python and the
framework does the rest.

---

## What the dashboard actually looks like

The renderer in `talk_eco.py` is a small "design system" on top of Prefab —
the planner LLM only ever emits ~10 high-level **widget kinds**, and the
renderer expands each into the prettiest Prefab primitives:

| Widget kind the LLM picks | Renders as | Used for |
|---|---|---|
| `metric_grid` | responsive `Grid` of `Metric` cards (with delta arrows) | the hero "Cities tracked / Avg AQI / Cleanest / Worst" row |
| `metric` / `stat` | `Metric` (label + big value + description + trend) | single big-number cards |
| `alert` | `Alert` + `AlertTitle` + `AlertDescription` + icon | "PM10 17× WHO limit", "smoke blowing TOWARD city", best-window callouts |
| `radial` | `RadialChart` (half-donut gauge) | every city's AQI on one gauge |
| `radar` | `RadarChart` | "pollutant fingerprint" overlay per city |
| `area` | `AreaChart` (smooth, optional stacked) | preferred over `line` for 24h trends |
| `bar` / `line` / `pie` / `sparkline` | matching Prefab chart | one-line catch-alls |
| `heatstrip` / `heatmap_7d` | colored `Dot` cells in `Row`s | hour-of-day strip / 7×24 weekly grid |
| `table` / `data_table` | sortable, paginated, optional-search `DataTable` | hotspot tables, pollutant rows, scoreboards |
| `news_list` | `Card` per headline with clickable `Link` + `Icon("newspaper")` | Google News results |
| `markdown` / `text` | `Markdown` (supports **bold**, lists, links) | prose explainers |
| `accordion` | `Accordion` of `AccordionItem`s | collapsible "Methodology" / "Sources" sections |
| `who_breach`, `outdoor_window`, `pollution_source`, `aqicn_compare`, `webpage_summary` | placeholder widgets — the renderer fetches the data and composes a mini-dashboard | causal/health tabs |

The shell wrapping every dashboard:

* Soft `bg-gradient-to-br from-sky-50 via-white to-emerald-50` background, `max-w-7xl` content
* Hero band with `Icon("wind")` + `H1` title + `Lead` subtitle ("Live air-quality intelligence · updated …")
* `Card` with `CardDescription` ("Every number traces back to a real API response — zero hallucination") and a `CardFooter` crediting the 8 data sources

To pre-compute heavy or aggregation-prone payloads (so the LLM never
hallucinates), `talk_eco.py` ships ready-to-use chart data with each
prompt: `overview_metrics`, `aqi_radial`, `pollutant_radar`,
`dominant_per_city_bar`, `dominant_pollutant_pie`,
`dominant_pollutant_table`. The planner is told to copy these arrays
verbatim.

---

## Setup

### 1. Environment variables (`.env`)

```bash
# Required — Gemini for the planner LLM
GEMINI_API_KEY=AIza...
# gemini-2.5-flash-lite has the biggest free-tier daily cap (~1000/day).
# DO NOT use gemini-2.5-flash — only 20/day on free tier.
GEMINI_MODEL=gemini-2.5-flash-lite

# Optional but recommended — enables CPCB station data
AQICN_TOKEN=...        # https://aqicn.org/data-platform/token/

# Optional but recommended — enables the satellite fire feature
FIRMS_MAP_KEY=...      # https://firms.modaps.eosdis.nasa.gov/api/area/
```

`GEMINI_API_KEY` is read from `assignment_eco/.env` if present, otherwise
falls back to [`../assignment/.env`](../assignment/.env). Both AQICN and
FIRMS degrade gracefully — if the key is missing, the corresponding tool
returns a clear actionable error instead of crashing.

### 2. Run AirSense (the natural-language REPL)

```sh
cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"
~/.local/bin/uv run --with prefab-ui --with fastmcp \
    --with requests --with python-dotenv --with google-genai \
    python talk_eco.py
```

Then open the URL it prints (typically <http://127.0.0.1:5175>) **next
to** the terminal so you can see tool calls firing on the left and the
UI rebuilding on the right.

Type prompts like:

```
Track Delhi, Mumbai, and Bengaluru. Compare them and show today's hourly trend.
For Delhi build a Health tab with WHO breach, safest 2h outdoor window, news, and nearby fires with wind direction.
Drop the cleanest city, swap it for Pune, refresh the two most polluted.
```

### (Optional) Run the legacy hard-coded agent

`agent.py` is the original 4-step deterministic loop kept for reference.
It plans Delhi → Mumbai → Bengaluru, runs `build_city_report` per city,
calls `show_dashboard`, and emits a final answer.

```sh
~/.local/bin/uv run --with prefab-ui --with fastmcp \
    --with requests --with python-dotenv --with google-genai \
    --with "mcp[cli]" python agent.py
```

---

## Recommended demo (3 prompts, ~3 minutes, 100% feature coverage)

### 🥇 Prompt 1 — setup, comparison, dynamic city resolution

```
Track Delhi, Mumbai, and Bengaluru. Compare them, show today's hourly trend, and add a 7-day heatmap for the most polluted city.
```

**Tools fired:** `build_city_report` × 3 (parallel), `list_log`
**Output:** Overview tab (stat cards + bar) · 24h Trend (multi-series line) · 7-Day Heatmap (7×24 grid for the city the LLM identified as worst).

### 🥈 Prompt 2 — the 6-API health briefing

```
For Delhi build a Health tab with WHO breach, safest 2h outdoor window, latest news, nearby fires with wind direction, CPCB second opinion, and a Background tab with the Wikipedia page on air pollution in Delhi.
```

**Tools fired:** `who_breach_summary`, `recommend_outdoor_window`, `fetch_news_for`, `fire_hotspots_near`, `fetch_aqicn`, `fetch_webpage`
**Output:** Health tab with WHO breach table · outdoor window card · live news headlines · 🔥 fire hotspots with wind compass + UPWIND tags · CPCB-vs-Open-Meteo comparison. Plus a clean Wikipedia Background tab.

### 🥉 Prompt 3 — iterative editing + dynamic reasoning

```
Drop the cleanest city, swap it for Pune, refresh the two most polluted, and keep the Wikipedia tab.
```

**Tools fired:** `list_log`, `remove_city`, `build_city_report` × 3
**Output:** Delete + Create + 2× Update in one sentence. The LLM resolves *"cleanest"* and *"two most polluted"* by reading `eco_log.json` itself. Wikipedia tab is preserved (incremental edit, not regenerate).

---

## File layout

```
assignment_eco/
├── mcp_server.py          # 17 MCP tools + 5-tab Prefab dashboard
├── talk_eco.py            # 🌟 natural-language REPL (the real demo)
├── agent.py               # legacy hard-coded agent loop (reference only)
├── eco_log.json           # CRUD store — one record per city, persisted to disk
├── generated_eco_app.py   # Prefab app file rewritten on every prompt
├── prefab_server.log      # Prefab dev-server logs
├── .env                   # GEMINI_API_KEY + GEMINI_MODEL + AQICN_TOKEN + FIRMS_MAP_KEY
└── README.md
```



