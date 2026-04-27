# 🎬 AirSense — Demo Script

A complete talk-track for showcasing AirSense in ~5 minutes. Read top-to-bottom.

---

## 1️⃣ What AirSense does (30 sec opener)

> *"This is **AirSense** — a natural-language air-quality intelligence dashboard for Indian cities. I type a sentence — like 'track Delhi and Mumbai and tell me why the air is bad' — and a browser-based dashboard rebuilds itself: stat cards, charts, tabs, satellite-fire maps, news headlines.*
>
> *I never name a widget, never name an API, never write JSON. The agent picks the tools, fetches live data, and the LLM composes the dashboard.*
>
> *Three layers: **MCP tools** fetch real data → **Gemini planner** decides what to render → **Prefab** turns Python components into a live web UI. Every number on screen traces back to a real HTTP response saved on disk — zero hallucination."*

---

## 2️⃣ The tools — 17 MCP tools in 4 buckets (45 sec)

> *"Open `mcp_server.py` and you'll see 17 MCP tools, grouped into 4 buckets:"*

### 🌐 Internet — live data fetch (5 tools)
| Tool | What it does |
|---|---|
| `fetch_aqi_now(city)` | Current AQI + 6 pollutants (Open-Meteo) |
| `fetch_aqi_24h(city)` | Last 24 hourly AQI + PM2.5 |
| `fetch_aqi_7d(city)` | 7 days × 24 hours grid for the heatmap |
| `fetch_weather_now(city)` | Temp, UV, humidity, **wind speed + direction** |
| `build_city_report(city)` | Runs all 4 above **in parallel** + saves to disk |

### 💾 CRUD on `eco_log.json` (4 tools)
| Tool | Operation |
|---|---|
| `save_city_report(...)` | **C**reate / **U**pdate (replaces by city name) |
| `list_log()` | **R**ead full log |
| `get_city_report(city)` | **R**ead one city |
| `remove_city(city)` | **D**elete |

### 🧮 Pure-Python analytics — no internet (4 tools)
| Tool | What it does |
|---|---|
| `who_breach_summary(city)` | Per-pollutant ratios over WHO 24h limits |
| `time_of_day_profile(city)` | Morning / afternoon / evening / night buckets |
| `fetch_aqi_forecast(city, hours)` | Next 1–48 forecast hours |
| `recommend_outdoor_window(city, h)` | Lowest-AQI window in next 24h |

### 🛰️ Causal context — *why* the air is bad (4 tools)
| Tool | Source |
|---|---|
| `fetch_news_for(city)` | Google News RSS |
| `fetch_webpage(url)` | Generic scraper + Wikipedia REST clean-extract |
| `fetch_aqicn(city)` | Official CPCB station data (second opinion) |
| **`fire_hotspots_near(city)`** | **NASA FIRMS satellite fires + wind cross-reference** |

### 🎨 UI (1 tool)
* `show_dashboard()` — `@mcp.tool(app=True)` returning a `PrefabApp` with 5 tabs (Right Now / Pollutants / 24h Trend / Hour Pattern / Compare Cities)

> **Talking point:** *"The killer tool is `fire_hotspots_near` — it pulls satellite fire detections from NASA FIRMS, cross-references them with wind direction from Open-Meteo, and tells you which fires are **upwind** of the city. Most AQI apps tell you the air is bad. This one tells you **why**."*

---

## 3️⃣ How Prefab works (30 sec)

> *"Prefab is **Python-as-a-UI-framework**. I write `Card`, `H1`, `BarChart`, `Tabs`, `Ring`, `Badge` as nested `with` blocks — no HTML, no React, no JSX. Prefab runs a tiny web server, hot-reloads the file on every save, and the browser updates automatically.*
>
> *So my LLM emits a small Python file like:*
> ```python
> with Card():
>     CardTitle("Delhi air today")
>     BarChart(data=[...], series=[...])
> ```
> *…and Prefab renders it live. The LLM never writes HTML — it writes 30–50 lines of Python and the framework does the rest. That's why the prompts work."*

---

## 4️⃣ The 3 demo prompts + outputs

### 🥇 Prompt 1 — Setup, comparison, dynamic resolution

```
Track Delhi, Mumbai, and Bengaluru. Compare them, show today's hourly trend, and add a 7-day heatmap for the most polluted city.
```

**Tools that fire (visible in terminal):**
```
· build_city_report("Delhi")      ← parallel
· build_city_report("Mumbai")     ← parallel
· build_city_report("Bengaluru")  ← parallel  ("Bangalore" auto-canonicalized)
· list_log()                      ← LLM finds the worst city
```

**What appears in browser:**
| Tab | Output |
|---|---|
| **Overview** | 3 stat cards (AQI ring + band + temp + UV per city) |
| **Pollutants** | Bar chart of PM2.5 / PM10 / NO₂ / O₃ / SO₂ / CO across all 3 cities |
| **24h Trend** | Multi-series line chart, one line per city, 24 points |
| **7-Day Heatmap** | 7×24 colored grid (168 cells) — only for Delhi (LLM picked it) |

**Talking point:** *"Notice I said 'most polluted city' — I didn't say which. The planner read `eco_log.json`, saw Delhi was highest, and put **only Delhi** in the heatmap. Dynamic resolution from a free-text phrase."*

---

### 🥈 Prompt 2 — The 6-API health briefing (your wow moment)

```
For Delhi build a Health tab with WHO breach, safest 2h outdoor window, latest news, nearby fires with wind direction, and a Background tab with the Wikipedia page on air pollution in Delhi.
```

**Tools that fire (6 different APIs, in parallel):**
```
· who_breach_summary("Delhi")              → WHO ratios from disk
· recommend_outdoor_window("Delhi", 2)     → Open-Meteo forecast
· fetch_news_for("Delhi", "air pollution") → Google News RSS
· fire_hotspots_near("Delhi", 200, 1)      → NASA FIRMS satellite
· fetch_aqicn("Delhi")                     → AQICN/CPCB stations
· fetch_webpage("https://en.wikipedia.org/wiki/Air_pollution_in_Delhi")
```

**What appears on the Health tab:**
| Card | Content |
|---|---|
| **WHO breach table** | "PM10 is **17× the WHO 24h limit**" — colored ratios per pollutant |
| **Outdoor window** | "Safest 2h block: **04:00–06:00**, avg AQI 142" |
| **News headlines** | Live Google News titles + sources + timestamps |
| 🔥 **Fire hotspots** | Big number ("**47 fires within 200km**") + wind compass + "smoke blowing TOWARD city" badge + top-5 hotspot table with **UPWIND** tags |
| **CPCB second opinion** | AQICN station-level AQI vs Open-Meteo grid value — side-by-side |

**Background tab:** Wikipedia lead paragraph (clean — no nav chrome, no infobox) via Wikipedia REST summary endpoint.

**Talking point (memorize this):**
> *"Most AQI apps stop at 'is it bad?'. AirSense answers **why**. NASA FIRMS gives me satellite-detected active fires. Open-Meteo gives me wind direction. The agent cross-references them — fires that sit in the direction the wind is **coming from** get flagged upwind. Smoke from those is what's making Delhi unbreathable today. **One sentence, six live data sources, causal explanation.**"*

---

### 🥉 Prompt 3 — Iterative editing + dynamic reasoning

```
Drop the cleanest city, swap it for Pune, refresh the two most polluted, and keep the Wikipedia tab.
```

**Tools that fire:**
```
· list_log()                      → LLM resolves "cleanest" + "two most polluted"
· remove_city("Bengaluru")        → DELETE
· build_city_report("Pune")       → CREATE
· build_city_report("Delhi")      → UPDATE  ← parallel
· build_city_report("Mumbai")     → UPDATE  ← parallel
```

**What appears:**
- Bengaluru disappears from every tab
- Pune appears with full data (AQI, pollutants, 24h trend, weather)
- Delhi and Mumbai numbers refresh (new `checked_at` timestamps)
- **Background / Wikipedia tab is preserved** — incremental edit, not regenerate-from-scratch

**Talking point:** *"That one sentence triggered a Delete + Create + two Updates + a Read — and the LLM resolved 'cleanest' and 'two most polluted' itself, by reading the JSON I never showed it. **Full CRUD coverage in one prompt.**"*

---



## 6️⃣ Closing line (15 sec)

> *"That's **AirSense**. **17 tools, 8 live data sources, 5 dashboard tabs, full CRUD, satellite-derived causal explanations** — driven by three plain English sentences. Every number traceable. Charts I never specified. Tabs I never named. And a working answer to **why** the air in Delhi is unbreathable today. Thank you."*

---



# 4. Launch
cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"
~/.local/bin/uv run --with prefab-ui --with fastmcp \
  --with requests --with python-dotenv --with google-genai python talk_eco.py
```

Open <http://127.0.0.1:5175> in a browser **next to** your terminal so the audience sees both the tool calls firing and the UI rebuilding.

---




## 🎯 The one talking-point to memorize

> *"Tools fetch, the planner composes, Prefab renders. Three layers, no hallucinations, every number traceable to disk."*

That sentence answers ~80% of any reviewer question. Good luck! 🚀
