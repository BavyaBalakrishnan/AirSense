# 🌍 AirSense — Natural-Language Air-Quality Intelligence Dashboard

A conversational agent that turns plain English into a live, browser-based air-quality dashboard for Indian cities. Type a sentence — the agent picks the tools, fetches real data, and composes a Prefab UI dashboard automatically.

**Three layers:** MCP tools fetch real data → Gemini agent decides what to render → Prefab turns Python components into a live web UI. Every number on screen traces back to a real HTTP response saved on disk — zero hallucination.

---

## 🚀 Quick Start

```bash
cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"

# Single command — uv handles all dependencies:
~/.local/bin/uv run --with prefab-ui --with fastmcp \
  --with requests --with python-dotenv --with google-genai --with mcp \
  python talk_eco.py
```

Open <http://127.0.0.1:5175> in a browser next to your terminal.

### Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key ([get one here](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL` | Model name (default: `gemini-2.0-flash`) |
| `AQICN_TOKEN` | Free token for CPCB station data ([sign up](https://aqicn.org/data-platform/token/)) |
| `FIRMS_MAP_KEY` | NASA FIRMS satellite fire API key ([sign up](https://firms.modaps.eosdis.nasa.gov/api/area/)) |
| `LLM_PRECALL_DELAY` | Seconds to sleep before each Gemini call (default: `10`, set `0` to disable) |

---

## 📁 File Structure

| File | Role |
|---|---|
| `talk_eco.py` | Main REPL — agent loop, planner, dashboard renderer, refresh server |
| `mcp_server.py` | 17 MCP tools (data fetch, CRUD, analytics, causal context) |
| `eco_log.json` | Persistent store of tracked cities (auto-created) |
| `generated_eco_app.py` | The Prefab app file rendered by the agent (auto-generated) |
| `smoke_boot.sh` / `smoke.py` | Quick smoke-test scripts |
| `DEMO.md` | 5-minute demo talk-track |

---

## 🔧 MCP Tools (17 total, 4 buckets)

### 🌐 Internet — Live Data Fetch (5+2 tools)

| Tool | Source | What it does |
|---|---|---|
| `fetch_aqi_now(city)` | Open-Meteo | Current AQI + 6 pollutants |
| `fetch_aqi_24h(city)` | Open-Meteo | Last 24 hourly AQI + PM2.5 |
| `fetch_aqi_7d(city)` | Open-Meteo | 7 days × 24 hours grid (heatmap) |
| `fetch_weather_now(city)` | Open-Meteo | Temp, UV, humidity, wind speed + direction |
| `build_city_report(city)` | Open-Meteo | All 4 above **in parallel** + save to disk |
| `fetch_aqi_forecast(city)` | Open-Meteo | Next 1–48 hour US AQI forecast |
| `fetch_news_for(city)` | Google News RSS | Recent headlines for a city |

### 💾 CRUD on `eco_log.json` (4 tools)

| Tool | Operation |
|---|---|
| `save_city_report(...)` | **C**reate / **U**pdate |
| `list_log()` | **R**ead full log |
| `get_city_report(city)` | **R**ead one city |
| `remove_city(city)` | **D**elete |

### 🧮 Pure-Python Analytics (4 tools)

| Tool | What it does |
|---|---|
| `who_breach_summary(city)` | Per-pollutant ratios over WHO 24h limits |
| `time_of_day_profile(city)` | Morning / afternoon / evening / night AQI buckets |
| `fetch_aqi_forecast(city, hours)` | Next 1–48 forecast hours |
| `recommend_outdoor_window(city, h)` | Lowest-AQI window in next 24h |

### 🛰️ Causal Context — *Why* the air is bad (4 tools)

| Tool | Source | What it does |
|---|---|---|
| `fetch_news_for(city)` | Google News RSS | Headlines + links |
| `fetch_webpage(url)` | Generic / Wikipedia REST | Clean text extract of any static page |
| `fetch_aqicn(city)` | AQICN / CPCB | Official station-level AQI (second opinion) |
| **`fire_hotspots_near(city)`** | **NASA FIRMS** | Satellite fires + wind cross-reference → **upwind** flags |

---

## 🎨 Dashboard Features

- **Gradient hero header** with title, live timestamp, and keyboard shortcut hint
- **Refresh button** — one click re-fetches all tracked cities in parallel, re-renders, and hot-reloads the browser (no page refresh needed)
- **Rich widget catalog** — radial gauges, area/bar/line/radar/pie charts, dot strips, 7-day heatmaps, WHO breach tables, outdoor window alerts, fire hotspot tables, news cards, Wikipedia summaries, CPCB comparisons, metric grids, sparklines, and more
- **Tabs** — agent picks tab names and composition based on your request
- **Incremental editing** — say "add a news tab" and only that tab changes; existing tabs are preserved

---

## 💬 Example Prompts

```
Track Delhi, Mumbai, and Bengaluru. Compare them and show a 7-day heatmap for the worst city.
```

```
For Delhi show a radial gauge, dot_strip for 24h trend, and a WHO breach tab.
```

```
Why is Delhi's air bad today? Show fires, wind direction, and news.
```

```
Drop the cleanest city, add Pune, and keep the Wikipedia tab.
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (http://127.0.0.1:5175)                    │
│  └── Prefab dev server (hot-reload on file change)  │
└──────────────────────┬──────────────────────────────┘
                       │ reads generated_eco_app.py
┌──────────────────────┴──────────────────────────────┐
│  talk_eco.py                                        │
│  ├── Agent loop (Gemini function-calling)           │
│  ├── Planner mode (/planner — legacy pipeline)      │
│  ├── _ag_dashboard() renderer (spec → Python file)  │
│  ├── PrefabServer (subprocess manager)              │
│  └── Refresh HTTP server (:5180)                    │
└──────────────────────┬──────────────────────────────┘
                       │ stdio JSON-RPC (MCP protocol)
┌──────────────────────┴──────────────────────────────┐
│  mcp_server.py (FastMCP)                            │
│  ├── Open-Meteo (AQI, weather, forecast)            │
│  ├── NASA FIRMS (satellite fires)                   │
│  ├── AQICN (CPCB station data)                      │
│  ├── Google News RSS                                │
│  ├── Wikipedia REST API                             │
│  └── eco_log.json (CRUD)                            │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **MCP over stdio** — tools are discovered via `list_tools` and called via `session.call_tool()` over JSON-RPC. This satisfies the "real MCP server" requirement.
2. **Agent-mode by default** — Gemini decides per-prompt whether to call tools, render a dashboard, or just text-reply. The legacy planner pipeline is still available via `/planner`.
3. **Server-hydrated widgets** — widgets like `who_breach`, `outdoor_window`, `pollution_source`, `news_list` are rendered at write-time (the renderer calls MCP tools and bakes the results into the Python file). No runtime callbacks needed.
4. **In-UI Refresh** — the browser's Refresh button POSTs to `:5180`, which re-fetches all cities in parallel, re-renders `generated_eco_app.py`, and bounces `prefab serve` so the browser hot-reloads.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `429 Resource Exhausted` | Increase `LLM_PRECALL_DELAY` in `.env` or switch to a model with higher RPM |
| Dashboard shows "Empty dashboard" | Check that `GEMINI_MODEL` is a valid, non-retired model name |
| Refresh button does nothing | Ensure `talk_eco.py` is still running (it hosts the `:5180` refresh server) |
| `prefab serve` won't start | Run `pkill -f "prefab serve"` and restart `talk_eco.py` |
| FIRMS/AQICN tools fail | Verify `FIRMS_MAP_KEY` and `AQICN_TOKEN` in `.env` |

---

## 📝 License

Educational project — EAG V3 Session 4 assignment.
