"""
AirSense — type a sentence, see the air-quality dashboard morph.

Same architecture as prefab/04_talk_to_app/prompt_to_app.py, but specialised
for the climate/AQI domain and wired to the REAL eco-tracker MCP server:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ You type:  "Add Delhi and Mumbai with a 24-hour heatmap"             │
  │                       │                                              │
  │                       ▼                                              │
  │ 1. Pre-pass:  detect cities + verb (add / refresh / remove)          │
  │               → call mcp_server.build_city_report(city)              │
  │                 which hits Open-Meteo IN PARALLEL and writes         │
  │                 eco_log.json (CRUD: Create / Update)                 │
  │               → call mcp_server.remove_city(city) for "remove ..."   │
  │                                                                      │
  │ 2. Planner:   LLM gets your sentence + the FULL eco_log.json          │
  │               → returns a dashboard JSON spec using real numbers     │
  │                                                                      │
  │ 3. Writer:    render spec into generated_eco_app.py                   │
  │               restart `prefab serve` subprocess                      │
  │                                                                      │
  │ 4. Browser:   dashboard reflects real Open-Meteo data instantly      │
  └──────────────────────────────────────────────────────────────────────┘

Rubric mapping (this script alone exercises all 3 rules):
  • Internet           — build_city_report fetches Open-Meteo (no key)
  • CRUD on a file     — eco_log.json: Create/Update on add or refresh,
                         Delete on remove, Read on every dashboard render
  • UI via Prefab      — LLM-generated dashboard rendered through Prefab,
                         served by `prefab serve` in the browser

Run:
    cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"
    ~/.local/bin/uv run --with prefab-ui --with fastmcp \\
        --with requests --with python-dotenv --with google-genai \\
        python talk_eco.py

Open http://127.0.0.1:5175 and start typing prompts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Reuse the existing MCP-server logic: parallel Open-Meteo fetch + CRUD.
import mcp_server as eco

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / "assignment" / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GENERATED = HERE / "generated_eco_app.py"
LOG_PATH = HERE / "prefab_server.log"
BACKUP = HERE / ".last_good_eco_app.py"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# 1. Pre-pass — turn the sentence into MCP-server function calls.
#    Detects intent (add / refresh / remove / show) and any Indian cities.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Pre-pass — turn the sentence into MCP-server function calls.
#    Uses a small LLM classifier (much smarter than regex) to extract:
#      { add: [...], remove: [...], refresh: [...] }
#    so prompts like "swap Mumbai for Pune", "drop the cleanest two and add
#    Patna instead", "refresh whichever city has the highest AQI" all work.
# ---------------------------------------------------------------------------

INTENT_PROMPT = """You are an INTENT CLASSIFIER for a climate-tracker REPL.
Given the user's sentence and the list of cities currently in eco_log.json,
return a SINGLE JSON object describing which CRUD actions to take BEFORE
re-rendering the dashboard.

Output schema (no prose, no code fences):
  {{
    "add":     ["<canonical city name>", ...],   // new cities to fetch + save
    "remove":  ["<city>", ...],                  // cities to delete from log
    "refresh": ["<city>", ...]                   // cities to re-fetch (Update)
  }}

Rules:
- Use the CANONICAL Indian city name (Bengaluru not Bangalore, Mumbai not
  Bombay, Kolkata not Calcutta, Chennai not Madras).
- "swap A for B" means {{"remove": ["A"], "add": ["B"]}}.
- "refresh all" or "update everything" with no city named means refresh = ALL
  current cities in eco_log.json.
- "remove the cleanest" / "drop the worst city" — resolve using the
  current_cities list ordered by aqi (provided below).
- If the user is just asking to view/show/explain WITHOUT changing data,
  return all three lists empty.
- Cities mentioned for the FIRST TIME (not in current_cities) go in "add"
  even if the verb is "show" or "track".
- Don't put the same city in more than one list.

Current cities in eco_log.json (with aqi, sorted lowest→highest):
{current_cities}

User sentence:
{user_request}
"""


def classify_intent(user_request: str) -> dict:
    """Ask the LLM what CRUD actions the user wants. Falls back to {} on
    any error so the planner still runs."""
    entries = eco._load()
    sorted_entries = sorted(entries, key=lambda e: e.get("aqi", 1e9))
    current = [
        {"city": e.get("city"), "aqi": e.get("aqi"), "band": e.get("band")}
        for e in sorted_entries
    ]
    prompt = INTENT_PROMPT.format(
        current_cities=json.dumps(current, indent=2) if current else "[]",
        user_request=user_request,
    )
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        raw = (resp.text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]
        parsed = json.loads(raw)
        # Defensive: ensure all three keys exist as lists.
        for k in ("add", "remove", "refresh"):
            v = parsed.get(k) or []
            if isinstance(v, str):
                v = [v]
            parsed[k] = [str(c).strip() for c in v if c]
        return parsed
    except Exception as e:
        print(f"  (intent classifier failed: {e} — running as 'show only')")
        return {"add": [], "remove": [], "refresh": []}


def apply_data_changes(user_request: str) -> list[str]:
    """Run the real MCP-server CRUD based on the LLM intent classification.
    Returns a short list of human-readable change notes (for the planner)."""
    intent = classify_intent(user_request)
    notes: list[str] = []

    # (intent JSON suppressed — set ECO_VERBOSE=1 to see it)
    if os.getenv("ECO_VERBOSE"):
        print(f"  intent: {json.dumps(intent)}")

    for c in intent.get("remove", []):
        r = eco.remove_city(c)
        notes.append(f"DELETE {c}: {r}")
        print(f"  · remove_city({c!r}) → {r}")

    # Add and refresh are functionally the same call (build_city_report
    # creates or updates), so merge them.
    to_fetch: list[str] = []
    for c in intent.get("add", []) + intent.get("refresh", []):
        if c not in to_fetch:
            to_fetch.append(c)
    for c in to_fetch:
        print(f"  · build_city_report({c!r}) — fetching Open-Meteo …")
        r = eco.build_city_report(c)
        notes.append(f"FETCH {c}: {r[:140]}")
        print(f"    ↳ {r[:140]}")

    if not notes:
        notes.append("(no data changes — re-rendering with current eco_log.json)")
    return notes


# ---------------------------------------------------------------------------
# 2. Widget catalog (a focused subset of the Talk-to-App catalog, plus a
#    `heatstrip` widget for the 24-hour AQI row).
# ---------------------------------------------------------------------------

def widget_lines(w: dict) -> list[str]:
    kind = w.get("kind", "")

    if kind == "stat" or kind == "metric":
        # Use the purpose-built Metric component — looks much nicer than a
        # raw Column+Muted+H1 stack and supports description + delta/trend.
        label = str(w.get("label", ""))
        value = str(w.get("value", ""))
        sub = w.get("sub") or w.get("description") or ""
        delta = w.get("delta")
        trend = w.get("trend")  # "up" | "down" | "neutral"
        sentiment = w.get("trend_sentiment") or w.get("sentiment")
        bits = [f"label={label!r}", f"value={value!r}"]
        if sub:
            bits.append(f"description={str(sub)!r}")
        if delta is not None:
            bits.append(f"delta={str(delta)!r}")
        if trend in ("up", "down", "neutral"):
            bits.append(f"trend={trend!r}")
        if sentiment in ("positive", "negative", "neutral"):
            bits.append(f"trendSentiment={sentiment!r}")
        return [f"Metric({', '.join(bits)})"]

    if kind == "metric_grid":
        # A responsive grid of Metric cards. Use this for "overview" rows
        # (cities tracked / network avg / cleanest / most polluted) so the
        # cards sit SIDE-BY-SIDE on desktop and stack on mobile.
        items = w.get("items") or []
        cols = int(w.get("columns") or min(4, max(1, len(items))))
        out = [f'with Grid(columns={cols}, gap=4):']
        for it in items:
            sub_lines = widget_lines({"kind": "metric", **it})
            out.append('    with GridItem():')
            for ln in sub_lines:
                out.append('        ' + ln)
        return out

    if kind == "alert":
        # Eye-catching banner. Variant: default | info | success | warning |
        # destructive. Great for "smoke blowing TOWARD city", "WHO breach",
        # "best outdoor window" callouts.
        variant = w.get("variant", "info")
        title = w.get("title", "")
        body = w.get("body", "") or w.get("description", "")
        icon = w.get("icon")
        head = f'variant={variant!r}'
        if icon:
            head += f', icon={icon!r}'
        out = [f'with Alert({head}):']
        if title:
            out.append(f'    AlertTitle({title!r})')
        if body:
            out.append(f'    AlertDescription({body!r})')
        return out

    if kind == "ring":
        val = max(0, min(100, int(w.get("value", 0) or 0)))
        label = w.get("label", "") or f"{val}"
        out = ['with Column(gap=2):']
        title = w.get("title")
        if title:
            out.append(f'    H3({title!r})')
        out.append(f'    Ring(value={val}, label={label!r})')
        return out

    if kind == "badge":
        return [f'Badge({w.get("label","")!r}, '
                f'variant={w.get("variant","default")!r})']

    if kind == "heatstrip":
        # 24-cell colored row driven by `values` + thresholds. Defaults are
        # US AQI bands: <=50 good, <=100 default, <=150 warn, >150 bad.
        values = w.get("values", []) or []
        t1, t2, t3 = w.get("thresholds", [50, 100, 150])
        title = w.get("title", "")
        out = ['with Column(gap=2):']
        if title:
            out.append(f'    H3({title!r})')
        if not values:
            out.append('    Muted("(no data)")')
            return out
        # Legend so viewers know what each color means.
        out.append('    with Row(gap=3):')
        out.append('        Muted("Legend:")')
        out.append(f'        Badge({f"Good ≤{t1}"!r}, variant="success")')
        out.append(f'        Badge({f"Moderate ≤{t2}"!r}, variant="default")')
        out.append(f'        Badge({f"Unhealthy ≤{t3}"!r}, variant="warning")')
        out.append(f'        Badge({f"Very Unhealthy >{t3}"!r}, variant="destructive")')
        # Big colored cells with hour numbers underneath — much nicer than
        # tiny labelled badges.
        out.append('    with Row(gap=1):')
        for v in values:
            v = v or 0
            if v <= t1: var = "success"
            elif v <= t2: var = "info"
            elif v <= t3: var = "warning"
            else: var = "destructive"
            out.append(f'        Dot(variant={var!r}, size="lg", shape="rounded")')
        out.append('    with Row(gap=1):')
        for i in range(len(values)):
            out.append(f'        Muted({f"{i:02d}"!r})')
        return out

    if kind == "heatmap_7d":
        # True 7-day × 24-hour heatmap. Two ways to invoke:
        #   {"kind": "heatmap_7d", "city": "Delhi"}            ← preferred
        #     → renderer pulls aqi_7d_grid from eco_log.json
        #   {"kind": "heatmap_7d",
        #    "grid": [{"date": "...", "hourly":[24]}, ...]}    ← inline data
        # The placeholder form keeps the planner's JSON small (no 168-cell
        # array round-tripping through Gemini, which previously caused
        # JSONDecodeError on big multi-city dashboards).
        t1, t2, t3 = w.get("thresholds", [50, 100, 150])
        title = w.get("title", "")
        grid = w.get("grid")
        if not grid:
            city_name = (w.get("city") or "").strip()
            if city_name:
                for e in eco._load():
                    if e.get("city", "").lower() == city_name.lower():
                        grid = e.get("aqi_7d_grid") or []
                        if not title:
                            title = f"7-day AQI heatmap — {e.get('city')}"
                        break
        out = ['with Column(gap=3):']
        if title:
            out.append(f'    H3({title!r})')
        if not grid:
            out.append('    Muted("(no 7-day data — re-run build_city_report)")')
            return out
        # Legend — one labelled badge per US-AQI band so viewers know what
        # the colors mean (Good ≤50, Moderate ≤100, Unhealthy ≤150,
        # Very Unhealthy / Hazardous >150).
        out.append('    with Row(gap=3):')
        out.append('        Muted("Legend:")')
        out.append(f'        Badge({f"Good ≤{t1}"!r}, variant="success")')
        out.append(f'        Badge({f"Moderate ≤{t2}"!r}, variant="default")')
        out.append(f'        Badge({f"Unhealthy ≤{t3}"!r}, variant="warning")')
        out.append(f'        Badge({f"Very Unhealthy >{t3}"!r}, variant="destructive")')
        # Hour-of-day header row.
        out.append('    with Row(gap=1):')
        out.append('        Text("           ")')  # spacer for date column
        for h in range(24):
            out.append(f'        Muted({f"{h:02d}"!r})')
        for row in grid:
            date = str(row.get("date", ""))
            cells = row.get("hourly") or []
            out.append('    with Row(gap=1):')
            out.append(f'        Muted({date!r})')
            for v in cells:
                if v is None:
                    var = "muted"
                elif v <= t1: var = "success"
                elif v <= t2: var = "info"
                elif v <= t3: var = "warning"
                else: var = "destructive"
                out.append(f'        Dot(variant={var!r}, size="lg", shape="rounded")')
        return out

    if kind == "who_breach":
        # Per-pollutant ratio over WHO 24h limit for one city. Hydrated
        # by calling mcp_server.who_breach_summary at render time, so the
        # planner just emits {"kind":"who_breach","city":"Delhi"}.
        city_name = (w.get("city") or "").strip()
        title = w.get("title") or (f"WHO breach summary — {city_name}"
                                   if city_name else "WHO breach summary")
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not city_name:
            out.append('    Muted("(no city specified)")')
            return out
        try:
            data = json.loads(eco.who_breach_summary(city_name))
        except Exception as ex:
            out.append(f'    Muted({f"(error: {ex})"!r})')
            return out
        # Headline alert about the worst breach.
        if isinstance(data, dict) and data.get("worst"):
            wbest = data["worst"]
            ratio = wbest.get("ratio", 0)
            var = ("success" if ratio <= 1 else
                   "warning" if ratio <= 2 else "destructive")
            icon = ("check-circle" if ratio <= 1 else
                    "alert-triangle" if ratio <= 2 else "alert-octagon")
            atitle = f"{wbest.get('pollutant','')} is {ratio}× the WHO 24h limit"
            adesc = wbest.get("status", "")
            out.append(f'    with Alert(variant={var!r}, icon={icon!r}):')
            out.append(f'        AlertTitle({atitle!r})')
            out.append(f'        AlertDescription({adesc!r})')
        # Sortable, paginated DataTable of all pollutants.
        rows = []
        for row in (data.get("all") or []):
            rows.append({
                "pollutant": str(row.get("pollutant", "")),
                "value": str(row.get("value", "")),
                "limit": str(row.get("limit", "")),
                "ratio": f"{row.get('ratio', 0)}×",
                "status": str(row.get("status", "")),
            })
        out.append('    DataTable(')
        out.append('        columns=[')
        out.append('            DataTableColumn(key="pollutant", header="Pollutant", sortable=True),')
        out.append('            DataTableColumn(key="value", header="Current (µg/m³)", sortable=True, align="right"),')
        out.append('            DataTableColumn(key="limit", header="WHO 24h limit", align="right"),')
        out.append('            DataTableColumn(key="ratio", header="Ratio", sortable=True, align="right"),')
        out.append('            DataTableColumn(key="status", header="Status"),')
        out.append('        ],')
        out.append(f'        rows={rows!r},')
        out.append('    )')
        return out

    if kind == "outdoor_window":
        # Best N-hour outdoor window from forecast. Hydrated by calling
        # mcp_server.recommend_outdoor_window at render time. Planner emits
        # {"kind":"outdoor_window","city":"Delhi","duration_hours":2}.
        city_name = (w.get("city") or "").strip()
        dur = int(w.get("duration_hours") or 2)
        title = w.get("title") or (f"Best {dur}h outdoor window — {city_name}"
                                   if city_name else "Best outdoor window")
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not city_name:
            out.append('    Muted("(no city specified)")')
            return out
        try:
            data = json.loads(eco.recommend_outdoor_window(city_name, dur))
        except Exception as ex:
            out.append(f'    Muted({f"(forecast unavailable: {ex})"!r})')
            return out
        best = data.get("best") or {}
        if not best:
            out.append('    Muted("(no forecast data)")')
            return out
        avg = best.get("avg_aqi", 0)
        band = best.get("band", "")
        var = ("success" if avg <= 50 else "info" if avg <= 100
               else "warning" if avg <= 150 else "destructive")
        # Big headline alert.
        start = str(best.get("start_iso", ""))[:16].replace("T", " ")
        end = str(best.get("end_iso", ""))[:16].replace("T", " ")
        atitle = f"Best window: {start} → {end} (avg AQI {avg:.0f}, {band})"
        adesc = best.get("advisory") or ""
        out.append(f'    with Alert(variant={var!r}, icon="clock"):')
        out.append(f'        AlertTitle({atitle!r})')
        if adesc:
            out.append(f'        AlertDescription({adesc!r})')
        # Two side-by-side metric cards for big-number readability.
        out.append('    with Grid(columns=2, gap=4):')
        out.append('        with GridItem():')
        out.append(f'            Metric(label="Window starts", value={start!r}, description={f"ends {end}"!r})')
        out.append('        with GridItem():')
        out.append(f'            Metric(label="Avg AQI in window", value={f"{avg:.0f}"!r}, description={band!r})')
        # Alternative windows.
        alts = data.get("alternatives") or []
        if alts:
            out.append('    Muted("Other low-AQI windows:")')
            for a in alts:
                a_avg = a.get("avg_aqi", 0)
                a_var = ("success" if a_avg <= 50 else "default"
                         if a_avg <= 100 else "warning"
                         if a_avg <= 150 else "destructive")
                a_start = str(a.get("start_iso", ""))[:16].replace("T", " ")
                a_label = f"AQI {a_avg:.0f} · {a.get('band', '')}"
                out.append('    with Row(gap=3):')
                out.append(f'        Dot(variant={a_var!r}, size="lg")')
                out.append(f'        Text({a_start!r})')
                out.append(f'        Badge({a_label!r}, variant={a_var!r})')
        return out

    if kind == "news_list":
        # Recent news headlines for a city. Hydrated by calling
        # mcp_server.fetch_news_for at render time. Planner emits
        # {"kind":"news_list","city":"Delhi","query":"air pollution","limit":5}.
        city_name = (w.get("city") or "").strip()
        query = w.get("query") or "air pollution"
        limit = int(w.get("limit") or 5)
        title = w.get("title") or (f"Latest news — {city_name} ({query})"
                                   if city_name else "Latest news")
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not city_name:
            out.append('    Muted("(no city specified)")')
            return out
        try:
            data = json.loads(eco.fetch_news_for(city_name, query, limit))
        except Exception as ex:
            out.append(f'    Muted({f"(news fetch failed: {ex})"!r})')
            return out
        items = data.get("items") or []
        if not items:
            out.append('    Muted("(no headlines found)")')
            return out
        # Each headline is its own mini-card with a clickable Link.
        for it in items:
            t = (it.get("title") or "").strip()
            src = (it.get("source") or "").strip()
            pub = (it.get("published") or "").strip()
            url = (it.get("link") or it.get("url") or "").strip()
            out.append('    with Card(css_class="hover:shadow-md transition-shadow"):')
            out.append('        with CardContent():')
            out.append('            with Row(gap=3, css_class="items-start"):')
            out.append('                Icon("newspaper", size="default")')
            out.append('                with Column(gap=1):')
            if url:
                out.append(f'                    Link({t!r}, href={url!r}, bold=True)')
            else:
                out.append(f'                    Text({t!r}, bold=True)')
            meta = " · ".join(x for x in (src, pub) if x)
            if meta:
                out.append(f'                    Muted({meta!r})')
        return out

    if kind == "webpage_summary":
        # Fetch any static webpage and dump its cleaned text. Hydrated by
        # calling mcp_server.fetch_webpage at render time. Planner emits
        # {"kind":"webpage_summary","url":"https://...","max_chars":1200}.
        # For Wikipedia URLs the MCP tool auto-uses the REST summary API,
        # so the body is the lead paragraph (no nav chrome / infobox).
        url = (w.get("url") or "").strip()
        max_chars = int(w.get("max_chars") or 1200)
        title = w.get("title") or "Webpage summary"
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not url:
            out.append('    Muted("(no url specified)")')
            return out
        try:
            data = json.loads(eco.fetch_webpage(url, max_chars))
        except Exception as ex:
            out.append(f'    Muted({f"(fetch failed: {ex})"!r})')
            return out
        page_title = (data.get("title") or "").strip()
        # Compose a nicer card with H2 title, lead description, link, and
        # the body rendered as Markdown for proper paragraphs / typography.
        if page_title:
            out.append(f'    H2({page_title!r})')
        desc = (data.get("description") or "").strip()
        if desc:
            out.append(f'    Lead({desc!r})')
        out.append('    with Row(gap=2):')
        out.append('        Icon("link", size="sm")')
        out.append(f'        Link({url!r}, href={url!r})')
        out.append('    Separator()')
        body = (data.get("text") or "").strip()
        if not body:
            out.append('    Muted("(no readable text — likely a JS-rendered page)")')
            return out
        out.append(f'    Markdown({body!r})')
        if data.get("truncated"):
            out.append('    Muted("(showing first {} characters — pass max_chars to see more)"'
                       '.format({}))'.format(max_chars))
        src = (data.get("source") or "").strip()
        if src == "wikipedia-rest-summary":
            out.append('    Muted("(lead paragraph from Wikipedia REST API)")')
        return out

    if kind == "aqicn_compare":
        # Side-by-side AQI from Open-Meteo (eco_log) vs AQICN (CPCB stations)
        # for one city. Planner emits {"kind":"aqicn_compare","city":"Delhi"}.
        city_name = (w.get("city") or "").strip()
        title = w.get("title") or (f"Open-Meteo vs AQICN — {city_name}"
                                   if city_name else "Open-Meteo vs AQICN")
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not city_name:
            out.append('    Muted("(no city specified)")')
            return out
        try:
            aqicn = json.loads(eco.fetch_aqicn(city_name))
        except Exception as ex:
            out.append(f'    Muted({f"(AQICN fetch failed: {ex})"!r})')
            return out
        if isinstance(aqicn, dict) and aqicn.get("aqi") is None:
            out.append('    Muted("(no AQICN data)")')
            return out
        # Local Open-Meteo number from eco_log.
        local = next((e for e in (json.loads(Path("eco_log.json").read_text())
                                  if Path("eco_log.json").exists() else [])
                      if e.get("city", "").lower() == city_name.lower()), None)
        local_aqi = (local or {}).get("aqi", "n/a")
        cpcb_aqi = aqicn.get("aqi", "n/a")
        station = aqicn.get("station", "")
        # Compute delta if both numeric, so the AQICN card can show a trend
        # arrow vs the Open-Meteo grid value.
        try:
            delta_v = float(cpcb_aqi) - float(local_aqi)
            delta_label = f"{delta_v:+.0f} vs grid"
            trend = "up" if delta_v > 0 else "down" if delta_v < 0 else "neutral"
            sentiment = "negative" if delta_v > 0 else "positive" if delta_v < 0 else "neutral"
        except Exception:
            delta_label = None
            trend = sentiment = None
        out.append('    with Grid(columns=2, gap=4):')
        out.append('        with GridItem():')
        out.append(f'            Metric(label="Open-Meteo (interpolated grid)", '
                   f'value={str(local_aqi)!r}, '
                   f'description="Free, no-key public API")')
        out.append('        with GridItem():')
        cpcb_bits = [f'label="AQICN (CPCB station)"',
                     f'value={str(cpcb_aqi)!r}',
                     f'description={(f"Station: {station}" or "Ground truth")!r}']
        if delta_label:
            cpcb_bits.append(f'delta={delta_label!r}')
            cpcb_bits.append(f'trend={trend!r}')
            cpcb_bits.append(f'trendSentiment={sentiment!r}')
        out.append(f'            Metric({", ".join(cpcb_bits)})')
        return out

    if kind == "pollution_source":
        # NASA FIRMS active-fire detections + wind direction → "is the smoke
        # blowing toward this city?". Hydrated by mcp_server.fire_hotspots_near
        # at render time. Planner emits
        # {"kind":"pollution_source","city":"Delhi","radius_km":200,"days":1}.
        city_name = (w.get("city") or "").strip()
        radius_km = int(w.get("radius_km") or 200)
        days = int(w.get("days") or 1)
        title = w.get("title") or (
            f"Fire hotspots within {radius_km} km — {city_name}"
            if city_name else "Pollution source")
        out = ['with Column(gap=3):', f'    H3({title!r})']
        if not city_name:
            out.append('    Muted("(no city specified)")')
            return out
        raw = eco.fire_hotspots_near(city_name, radius_km, days)
        if isinstance(raw, str) and raw.lstrip().startswith("ERROR"):
            out.append(f'    Muted({raw!r})')
            out.append('    Muted("Get a free MAP_KEY at '
                       'https://firms.modaps.eosdis.nasa.gov/api/area/ '
                       'and add FIRMS_MAP_KEY=<key> to assignment_eco/.env")')
            return out
        try:
            data = json.loads(raw)
        except Exception as ex:
            out.append(f'    Muted({f"(FIRMS parse failed: {ex})"!r})')
            return out

        count = int(data.get("count", 0) or 0)
        upwind = int(data.get("upwind_count", 0) or 0)
        downwind = int(data.get("downwind_count", 0) or 0)
        wind = data.get("wind") or {}
        w_speed = wind.get("speed_kmh", 0) or 0
        w_from = wind.get("from_compass", "?")

        # Headline row: Metric cards + a coloured Alert callout.
        if count == 0:
            count_var, alert_var = "success", "success"
            headline = "No active fires detected"
            alert_icon = "check-circle"
        elif upwind > 0:
            count_var, alert_var = "destructive", "destructive"
            headline = f"{upwind} UPWIND fires of {count} total"
            alert_icon = "flame"
        elif count <= 5:
            count_var, alert_var = "warning", "warning"
            headline = f"{count} fires (none upwind)"
            alert_icon = "flame"
        else:
            count_var, alert_var = "warning", "warning"
            headline = f"{count} fires nearby (none upwind)"
            alert_icon = "flame"
        wind_label = (f"{w_speed:.0f} km/h from {w_from}"
                      if w_speed else "calm / no wind data")
        out.append('    with Grid(columns=3, gap=4):')
        out.append('        with GridItem():')
        out.append(f'            Metric(label={f"FIRMS · last {days}d · {radius_km} km"!r}, '
                   f'value={str(count)!r}, description="active fires detected")')
        out.append('        with GridItem():')
        out.append(f'            Metric(label="Wind", value={wind_label!r}, '
                   f'description="from this direction")')
        out.append('        with GridItem():')
        up_desc = (f"{upwind} upwind · {downwind} downwind"
                   if (upwind or downwind) else "no fires")
        out.append(f'            Metric(label="Upwind risk", value={str(upwind)!r}, '
                   f'description={up_desc!r})')
        # Big alert about the smoke direction.
        if upwind > 0 and w_speed > 1:
            atitle = "🔥 Smoke likely blowing TOWARD city"
        elif count > 0 and w_speed > 1:
            atitle = "✓ Smoke blowing AWAY from city"
        else:
            atitle = headline
        out.append(f'    with Alert(variant={alert_var!r}, icon={alert_icon!r}):')
        out.append(f'        AlertTitle({atitle!r})')

        # Plain-English insight line so non-experts understand the chart.
        if count == 0:
            insight = (f"No NASA FIRMS satellite fire detections within "
                       f"{radius_km} km in the last {days} day(s). Today's "
                       f"AQI is unlikely to be driven by biomass burning.")
        elif upwind > 0 and w_speed > 1:
            insight = (f"{upwind} of {count} fires sit in the direction the "
                       f"wind is coming FROM ({w_from}, {w_speed:.0f} km/h) — "
                       f"smoke from these is likely contributing to today's "
                       f"PM2.5 / PM10 levels.")
        elif count > 0 and w_speed > 1:
            insight = (f"{count} fires detected, but wind is from {w_from} "
                       f"@ {w_speed:.0f} km/h — they sit downwind, so smoke "
                       f"is blowing away from the city.")
        else:
            insight = (f"{count} fires detected within {radius_km} km. Wind "
                       f"data unavailable, so upwind/downwind direction "
                       f"cannot be inferred.")
        out.append(f'        AlertDescription({insight!r})')

        # Top hotspots table — sortable DataTable.
        top = data.get("top") or []
        if top:
            rows = []
            for f in top:
                rows.append({
                    "distance": f"{f.get('distance_km', 0):.0f} km",
                    "direction": f"{f.get('compass', '?')} ({int(f.get('bearing_deg', 0))}°)",
                    "date": str(f.get("acq_date", "")),
                    "brightness": f"{f.get('brightness', 0):.0f} K",
                    "upwind": "UPWIND" if bool(f.get("upwind")) else "downwind",
                })
            out.append('    H3("Nearest hotspots")')
            out.append('    DataTable(')
            out.append('        columns=[')
            out.append('            DataTableColumn(key="distance", header="Distance", sortable=True),')
            out.append('            DataTableColumn(key="direction", header="Direction"),')
            out.append('            DataTableColumn(key="date", header="Date", sortable=True),')
            out.append('            DataTableColumn(key="brightness", header="Brightness", sortable=True, align="right"),')
            out.append('            DataTableColumn(key="upwind", header="Upwind?"),')
            out.append('        ],')
            out.append(f'        rows={rows!r},')
            out.append('        paginated=True, pageSize=5,')
            out.append('    )')
        return out

    if kind == "bar":
        data = w.get("data", [])
        x_key = w.get("x_key", "x")
        y_keys = w.get("y_keys", ["y"])
        if isinstance(y_keys, str): y_keys = [y_keys]
        series = ", ".join(f'ChartSeries(data_key={yk!r}, label={yk!r})' for yk in y_keys)
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out += [f'    BarChart(data={data!r},',
                f'             series=[{series}],',
                f'             x_axis={x_key!r}, show_legend={len(y_keys) > 1})']
        return out

    if kind == "line":
        data = w.get("data", [])
        x_key = w.get("x_key", "x")
        y_keys = w.get("y_keys", ["y"])
        if isinstance(y_keys, str): y_keys = [y_keys]
        series = ", ".join(f'ChartSeries(data_key={yk!r}, label={yk!r})' for yk in y_keys)
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out += [f'    LineChart(data={data!r},',
                f'              series=[{series}],',
                f'              x_axis={x_key!r}, show_legend={len(y_keys) > 1})']
        return out

    if kind == "area":
        # Filled-area variant of LineChart — looks much nicer for trend tabs.
        data = w.get("data", [])
        x_key = w.get("x_key", "x")
        y_keys = w.get("y_keys", ["y"])
        if isinstance(y_keys, str): y_keys = [y_keys]
        series = ", ".join(f'ChartSeries(data_key={yk!r}, label={yk!r})' for yk in y_keys)
        stacked = bool(w.get("stacked", False))
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out += [f'    AreaChart(data={data!r},',
                f'              series=[{series}],',
                f'              xAxis={x_key!r}, curve="smooth",',
                f'              stacked={stacked!r}, showDots=False,',
                f'              showLegend={len(y_keys) > 1})']
        return out

    if kind == "radar":
        # Pollutant fingerprint: each city is one polygon over PM25/PM10/NO2/O3/SO2/CO.
        # data = [{"axis":"PM25","Delhi":312,"Mumbai":98}, ...]
        data = w.get("data", [])
        axis_key = w.get("axis_key", "axis")
        y_keys = w.get("y_keys", [])
        if isinstance(y_keys, str): y_keys = [y_keys]
        series = ", ".join(f'ChartSeries(data_key={yk!r}, label={yk!r})' for yk in y_keys)
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out += [f'    RadarChart(data={data!r},',
                f'               series=[{series}],',
                f'               axisKey={axis_key!r}, filled=True,',
                f'               showLegend={len(y_keys) > 1})']
        return out

    if kind == "radial":
        # Half-donut gauge — great for a single AQI / score readout.
        data = w.get("data", [])
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out.append(
            f'    RadialChart(data={data!r}, dataKey={w.get("value_key","value")!r}, '
            f'nameKey={w.get("name_key","name")!r}, innerRadius=50, '
            f'startAngle=180, endAngle=0)'
        )
        return out

    if kind == "pie":
        data = w.get("data", [])
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out.append(
            f'    PieChart(data={data!r}, data_key={w.get("value_key","value")!r}, '
            f'name_key={w.get("name_key","name")!r}, show_legend=True)'
        )
        return out

    if kind == "sparkline":
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out.append(f'    Sparkline(data={(w.get("values") or [])!r})')
        return out

    if kind == "table" or kind == "data_table":
        # Use the proper DataTable component — sortable, optionally
        # paginated/searchable, with column alignment.
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        cols = w.get("columns", [])
        # Allow either ["Header", ...] or [{key, header, sortable, align}, ...]
        norm_cols = []
        if cols and isinstance(cols[0], dict):
            norm_cols = cols
        else:
            for c in cols:
                key = re.sub(r"[^a-z0-9_]+", "_", str(c).lower()).strip("_") or "col"
                norm_cols.append({"key": key, "header": str(c), "sortable": True})
        # Normalise rows into list[dict].
        rows_in = w.get("rows", [])
        norm_rows = []
        for row in rows_in:
            if isinstance(row, dict):
                norm_rows.append({c["key"]: str(row.get(c["key"], row.get(c["header"], "")))
                                  for c in norm_cols})
            else:
                norm_rows.append({c["key"]: str(v) for c, v in zip(norm_cols, row)})
        searchable = bool(w.get("search", False))
        paginated = bool(w.get("paginated", len(norm_rows) > 8))
        page_size = int(w.get("page_size") or 8)
        out.append('    DataTable(')
        out.append('        columns=[')
        for c in norm_cols:
            bits = [f'key={c["key"]!r}', f'header={c["header"]!r}']
            if c.get("sortable", True):
                bits.append('sortable=True')
            if c.get("align"):
                bits.append(f'align={c["align"]!r}')
            out.append(f'            DataTableColumn({", ".join(bits)}),')
        out.append('        ],')
        out.append(f'        rows={norm_rows!r},')
        out.append(f'        search={searchable!r}, paginated={paginated!r}, '
                   f'pageSize={page_size},')
        out.append('    )')
        return out

    if kind == "markdown":
        # Render rich markdown (lists, bold, links, code) — much nicer than
        # cramming long explainers into Muted().
        body = w.get("body", "") or w.get("content", "")
        out = []
        title = w.get("title")
        if title:
            out.append(f'H3({title!r})')
        out.append(f'Markdown({body!r})')
        return out

    if kind == "accordion":
        # Collapsible FAQ-style sections. items = [{"title":..., "body":...}].
        items = w.get("items") or []
        multiple = bool(w.get("multiple", False))
        out = [f'with Accordion(multiple={multiple!r}, collapsible=True):']
        for it in items:
            t = str(it.get("title", "")) or "Section"
            body = str(it.get("body", ""))
            out.append(f'    with AccordionItem(title={t!r}):')
            if body:
                out.append(f'        Markdown({body!r})')
        return out

    if kind == "text":
        # Headline + body with much better typography. If "level" is "lead"
        # we use the Lead component (large muted intro paragraph).
        out = ['with Column(gap=2):']
        h = w.get("heading", "")
        body = w.get("body", "")
        level = str(w.get("level", "h3")).lower()
        if h:
            if level == "h1":
                out.append(f'    H1({h!r})')
            elif level == "h2":
                out.append(f'    H2({h!r})')
            elif level == "lead":
                out.append(f'    Lead({h!r})')
            else:
                out.append(f'    H3({h!r})')
        if body:
            # Render as Markdown so the planner can use **bold**, lists, links.
            out.append(f'    Markdown({body!r})')
        return out

    if kind == "image":
        # Inline image — useful for static maps (e.g. NASA FIRMS bbox PNG),
        # screenshots, logos. Planner emits {"kind":"image","src":"https://...",
        # "alt":"...", "caption":"..."}.
        src = (w.get("src") or "").strip()
        alt = w.get("alt") or w.get("caption") or "image"
        caption = w.get("caption") or ""
        out = ['with Column(gap=2):']
        title = w.get("title")
        if title:
            out.append(f'    H3({title!r})')
        if not src:
            out.append('    Muted("(no src specified)")')
            return out
        out.append(f'    Image(src={src!r}, alt={alt!r}, '
                   f'css_class="rounded-lg border border-slate-200 max-w-full")')
        if caption:
            out.append(f'    Muted({caption!r})')
        return out

    if kind == "carousel":
        # Auto-rotating carousel of items. Each item is a small Card with
        # heading + body + optional link. Great for "Today's headlines" or
        # rotating tips. Planner emits
        # {"kind":"carousel","items":[{"heading":"...","body":"...","link":"https://..."}]}.
        items = w.get("items") or []
        out = ['with Column(gap=3):']
        title = w.get("title")
        if title:
            out.append(f'    H3({title!r})')
        if not items:
            out.append('    Muted("(no items)")')
            return out
        out.append('    with Carousel(autoplay=True, interval=4000, loop=True):')
        for it in items:
            heading = str(it.get("heading", "")) or "Item"
            body = str(it.get("body", ""))
            link = str(it.get("link", ""))
            out.append('        with Card(css_class="p-4"):')
            out.append('            with Column(gap=2):')
            out.append(f'                H3({heading!r})')
            if body:
                out.append(f'                Muted({body!r})')
            if link:
                out.append(f'                Link({link!r}, href={link!r}, target="_blank")')
        return out

    if kind == "scatter":
        # Scatter plot — perfect for "PM2.5 vs wind speed" or any 2-variable
        # correlation. Planner emits
        # {"kind":"scatter","data":[{"x":1,"y":2,"label":"Delhi"}], "x_key":"x", "y_key":"y"}.
        data = w.get("data", [])
        x_key = w.get("x_key", "x")
        y_key = w.get("y_key", "y")
        out = ['with Column(gap=2):']
        title = w.get("title", "")
        if title:
            out.append(f'    H3({title!r})')
        out.append(f'    ScatterChart(data={data!r}, '
                   f'series=[ChartSeries(data_key={y_key!r}, label={y_key!r})], '
                   f'xAxis={x_key!r}, showLegend=False)')
        return out

    if kind == "calendar_picker":
        # Date picker calendar. Planner emits
        # {"kind":"calendar_picker","title":"Pick a day","name":"d","value":"2026-04-27"}.
        out = ['with Column(gap=2):']
        title = w.get("title")
        if title:
            out.append(f'    H3({title!r})')
        name = w.get("name") or "picked_date"
        value = w.get("value")
        bits = [f'name={name!r}']
        if value:
            bits.append(f'value={value!r}')
        out.append(f'    Calendar({", ".join(bits)})')
        return out

    if kind == "combobox" or kind == "select":
        # Searchable dropdown (Combobox) or simple Select. Planner emits
        # {"kind":"combobox","title":"Compare against","name":"city",
        #  "options":["Delhi","Mumbai","Pune"], "value":"Delhi"}.
        options = w.get("options") or []
        out = ['with Column(gap=2):']
        title = w.get("title")
        if title:
            out.append(f'    H3({title!r})')
        if not options:
            out.append('    Muted("(no options)")')
            return out
        name = w.get("name") or "selection"
        value = w.get("value") or (options[0] if options else "")
        comp = "Combobox" if kind == "combobox" else "Select"
        opt = "ComboboxOption" if kind == "combobox" else "SelectOption"
        out.append(f'    with {comp}(name={name!r}, value={value!r}):')
        for o in options:
            o_str = str(o)
            out.append(f'        {opt}({o_str!r}, value={o_str!r})')
        return out

    if kind == "blockquote" or kind == "quote":
        # Rendered as <blockquote>. Use for WHO advisories, expert quotes,
        # mission statements. Planner emits
        # {"kind":"blockquote","body":"...","cite":"WHO 2021 guidelines"}.
        body = str(w.get("body", "")) or ""
        cite = str(w.get("cite", "")) or ""
        out = []
        title = w.get("title")
        if title:
            out.append(f'H3({title!r})')
        if not body:
            out.append('Muted("(empty quote)")')
            return out
        out.append('with BlockQuote():')
        out.append(f'    Text({body!r})')
        if cite:
            out.append(f'    Muted({f"— {cite}"!r})')
        return out

    if kind == "kbd":
        # Inline keyboard-shortcut hints. Planner emits
        # {"kind":"kbd","keys":["R"],"label":"Refresh data"}.
        keys = w.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        label = w.get("label", "")
        out = ['with Row(gap=2, css_class="items-center"):']
        for k in keys:
            out.append(f'    Kbd({str(k)!r})')
        if label:
            out.append(f'    Muted({label!r})')
        return out

    return [f'Muted({f"Unknown widget kind: {kind!r}"!r})']


def render_dashboard(title: str, tabs: list[dict]) -> str:
    if not tabs:
        tabs = [{"name": "Empty", "widgets": [{"kind": "text", "heading": "Empty dashboard"}]}]

    TAB_INDENT = " " * 24
    from datetime import datetime as _dt
    subtitle = (
        f"Live air-quality intelligence · "
        f"updated {_dt.now().strftime('%b %d, %Y · %H:%M:%S')}"
    )
    parts = [
        "from prefab_ui.app import PrefabApp",
        "from prefab_ui.components import (",
        "    Accordion, AccordionItem, Alert, AlertDescription, AlertTitle,",
        "    Badge, BlockQuote, Button, Calendar, Card, CardContent,",
        "    CardDescription, CardFooter, CardHeader, CardTitle, Carousel,",
        "    Column, Combobox, ComboboxOption, DataTable, DataTableColumn,",
        "    Dot, Grid, GridItem, H1, H2, H3, Icon, Image, Kbd, Lead, Link,",
        "    Loader, Markdown, Metric, Muted, Ring, Row, Select, SelectOption,",
        "    Separator, Tab, Tabs, Text,",
        ")",
        "from prefab_ui.components.charts import (",
        "    AreaChart, BarChart, ChartSeries, LineChart, PieChart,",
        "    RadarChart, RadialChart, ScatterChart, Sparkline,",
        ")",
        # Declarative actions for the in-UI Refresh button. The button POSTs
        # to a tiny HTTP refresh endpoint (started by talk_eco.py on
        # http://127.0.0.1:5180/refresh) which re-runs build_city_report on
        # every tracked city and bumps the generated file's mtime — Prefab's
        # hot-reload watcher then re-renders the browser automatically.
        "from prefab_ui.actions import Fetch, ShowToast",
        "",
        # Soft gradient background + wider max-width for a more dashboard-y feel.
        'with PrefabApp(css_class="min-h-screen bg-gradient-to-br '
        'from-sky-50 via-white to-emerald-50 p-6") as app:',
        '    with Column(gap=4, css_class="max-w-7xl mx-auto"):',
        # Hero band above the main card.
        '        with Row(gap=3, css_class="items-center justify-between"):',
        '            with Row(gap=3, css_class="items-center"):',
        '                Icon("wind", size="lg")',
        '                with Column(gap=0):',
        f"                    H1({title!r})",
        f"                    Lead({subtitle!r})",
        # In-UI Refresh button — POSTs to the local refresh endpoint started
        # by talk_eco.py (see _RefreshHandler / start_refresh_server below).
        # The server re-runs build_city_report for every tracked city, then
        # touches generated_eco_app.py so Prefab hot-reloads the browser.
        # Purely declarative — no Python callable on the click path.
        '            with Row(gap=2, css_class="items-center"):',
        '                Kbd("R")',
        '                Button("Refresh data", icon="refresh-cw",',
        '                       variant="outline",',
        '                       on_click=Fetch(',
        '                           url="http://127.0.0.1:5180/refresh",',
        '                           method="POST",',
        '                           onSuccess=ShowToast(',
        '                               "Refreshing live data…",',
        '                               description="Re-fetching every tracked city. The dashboard will reload in a moment.",',
        '                               variant="success", duration=4000),',
        '                           onError=ShowToast(',
        '                               "Refresh failed",',
        '                               description="Is talk_eco.py still running? Check the terminal.",',
        '                               variant="error", duration=6000),',
        '                       ))',
        '        with Card(css_class="shadow-lg border-slate-200/70"):',
        "            with CardHeader():",
        f"                CardTitle({title!r})",
        '                CardDescription("Every number traces back to a real '
        'API response stored in eco_log.json — zero hallucination.")',
        "            with CardContent():",
    ]
    # Indent the rest one extra level (we wrapped in an extra Column).
    TAB_INDENT = " " * 28
    first_value = re.sub(r"[^a-z0-9_]+", "_", str(tabs[0].get("name", "tab1")).lower()).strip("_") or "tab1"
    parts.append(f"                with Tabs(value={first_value!r}):")
    for i, tab in enumerate(tabs):
        name = str(tab.get("name") or f"Tab {i+1}")
        value = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"tab_{i+1}"
        widgets = tab.get("widgets") or [{"kind": "text", "body": "(empty tab)"}]
        parts.append(f'                    with Tab({name!r}, value={value!r}):')
        parts.append("                        with Column(gap=6):")
        for w in widgets:
            for line in widget_lines(w):
                parts.append((TAB_INDENT + line) if line else "")
    # Footer with provenance.
    parts.append("            with CardFooter():")
    parts.append('                Muted("Sources: Open-Meteo · NASA FIRMS · '
                 'AQICN/CPCB · Google News · Wikipedia. Powered by Prefab UI.")')
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 3. Planner — LLM gets the user request + REAL eco_log.json + previous spec
#    and returns a dashboard JSON spec. Real numbers, no invention.
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You design climate / air-quality dashboards for Indian cities.
The dashboard MUST visualise the REAL data in `eco_log.json` provided below.
Do NOT invent any numbers — every value, label and series must come from the
provided data.

You produce ONE JSON object with this shape (no prose, no code fences):
  {{
    "title": "<short title>",
    "tabs": [
      {{"name": "<tab name>", "widgets": [<widget>, ...]}},
      ...
    ]
  }}

Each widget is one of:

  {{"kind": "stat",      "label": "<label>", "value": "<big value>", "sub": "<optional caption>"}}
        // Single big-number card. Renders as a Prefab Metric component.
  {{"kind": "metric",    "label": "...", "value": "...", "description": "...",
                         "delta": "+12 vs grid", "trend": "up|down|neutral",
                         "trend_sentiment": "positive|negative|neutral"}}
        // Same as `stat` but supports a delta arrow + colour. Use for
        // comparisons (Open-Meteo vs CPCB, today vs yesterday).
  {{"kind": "metric_grid", "columns": 4,
                           "items": [<stat-or-metric without "kind">, ...]}}
        // Responsive Grid of Metric cards. Use this for the hero "overview"
        // row so the cards sit SIDE-BY-SIDE on desktop. Always prefer this
        // over stacking 4 separate `stat` widgets.
  {{"kind": "alert",     "variant": "info|success|warning|destructive",
                         "title": "...", "body": "...", "icon": "<lucide-name>"}}
        // Big colored callout banner. Use for "smoke blowing TOWARD city",
        // "WHO PM10 17× over limit", "Best window 04:00-06:00", etc. Icons:
        // "alert-triangle", "alert-octagon", "check-circle", "info", "flame",
        // "wind", "clock", "newspaper", "link".
  {{"kind": "ring",      "title": "<optional>", "value": <0..100>, "label": "<optional>"}}
  {{"kind": "badge",     "label": "...", "variant": "default|success|warning|destructive"}}
  {{"kind": "heatstrip", "title": "<optional>", "values": [<24 numbers>], "thresholds": [50, 100, 150]}}
        // 24-cell colored row, one cell per HOUR-of-day. Use for "calendar
        // heatmap" or "24-hour AQI heatmap" requests on a single city.
  {{"kind": "heatmap_7d","title": "<optional>", "city": "<city in eco_log>"}}
        // TRUE 7×24 heatmap (rows = days, cols = hours). Use when the user
        // asks for "weekly heatmap", "7-day pattern", "this week's hours".
        // ALWAYS use the {{"city": "..."}} placeholder form — the renderer
        // will pull aqi_7d_grid from eco_log.json. Do NOT inline the grid
        // data, it is too large to round-trip safely.
  {{"kind": "who_breach", "title": "<optional>", "city": "<city>"}}
        // Per-pollutant ratio over the WHO 24h limit + worst breach badge.
        // The renderer calls mcp_server.who_breach_summary(city). Use when
        // the user asks "why is X polluted", "WHO limits", "worst pollutant".
  {{"kind": "outdoor_window", "title": "<optional>",
                              "city": "<city>", "duration_hours": 2}}
        // Best low-AQI window in the next 24 forecast hours. The renderer
        // calls mcp_server.recommend_outdoor_window. Use when the user asks
        // "when should I run", "safest time to go out", "best window".
  {{"kind": "news_list", "title": "<optional>", "city": "<city>",
                         "query": "air pollution", "limit": 5}}
        // Recent Google News headlines for the city. The renderer calls
        // mcp_server.fetch_news_for. Use when the user asks for "news",
        // "headlines", "what's happening", or wants real-world context.
  {{"kind": "webpage_summary", "title": "<optional>",
                               "url": "https://...", "max_chars": 2000}}
        // Fetch any STATIC webpage (Wikipedia, gov bulletin, blog, news
        // article) and show its cleaned text. The renderer calls
        // mcp_server.fetch_webpage. Use when the user pastes a URL or asks
        // to "read" / "summarise" a page. NOT for JS-rendered SPAs
        // (aqi.in, accuweather.com) — those return empty shells.
  {{"kind": "aqicn_compare", "title": "<optional>", "city": "<city>"}}
        // Side-by-side comparison: Open-Meteo (interpolated grid) vs AQICN
        // (CPCB ground stations — same data aqi.in shows). The renderer
        // calls mcp_server.fetch_aqicn. Use when the user asks for "CPCB",
        // "aqi.in", "station data", or "second opinion" on AQI.
  {{"kind": "pollution_source", "title": "<optional>", "city": "<city>",
                                "radius_km": 200, "days": 1}}
        // NASA FIRMS active-fire detections within `radius_km` of the city,
        // cross-referenced with current wind direction to flag fires that
        // sit UPWIND (likely smoke source). The renderer calls
        // mcp_server.fire_hotspots_near. Use when the user asks "why is
        // AQI high", "where is the smoke from", "fires near X", "stubble
        // burning", "pollution source", or wants a CAUSAL explanation
        // for a PM2.5 / PM10 spike. radius_km defaults to 200 (range
        // 10-1000), days defaults to 1 (range 1-10).
  {{"kind": "bar",       "title": "<optional>", "data": [...], "x_key": "city", "y_keys": ["aqi"]}}
  {{"kind": "line",      "title": "<optional>", "data": [...], "x_key": "hour", "y_keys": ["Delhi","Mumbai"]}}
  {{"kind": "area",      "title": "<optional>", "data": [...], "x_key": "hour",
                         "y_keys": ["Delhi","Mumbai"], "stacked": false}}
        // Smooth filled-area chart — preferred over `line` for 24h trend tabs;
        // looks much nicer.
  {{"kind": "radar",     "title": "<optional>", "data": [...], "axis_key": "axis",
                         "y_keys": ["Delhi","Mumbai"]}}
        // Pollutant fingerprint — one polygon per city across the 6
        // pollutants. ALWAYS use the pre-computed `pollutant_radar` payload.
  {{"kind": "radial",    "title": "<optional>", "data": [...],
                         "name_key": "name", "value_key": "value"}}
        // Half-donut gauge. Use the pre-computed `aqi_radial` payload to
        // show every city's AQI on one beautiful gauge.
  {{"kind": "pie",       "title": "<optional>", "data": [{{"name":"...","value":<n>}}], "name_key": "name", "value_key": "value"}}
  {{"kind": "sparkline", "title": "<optional>", "values": [<numbers>]}}
  {{"kind": "table",     "title": "<optional>", "columns": ["..."], "rows": [["...","..."], ...]}}
        // Renders as a sortable, paginated DataTable. Set "search": true to
        // add a search box for long tables.
  {{"kind": "markdown",  "title": "<optional>", "body": "## any **markdown**"}}
        // For longer prose explainers (lists, bold, links, headings).
  {{"kind": "accordion", "items": [{{"title": "Q1", "body": "A1 (markdown ok)"}}, ...],
                         "multiple": false}}
        // Collapsible FAQ-style sections. Great for "Methodology", "Sources",
        // or per-pollutant deep dives that would otherwise crowd a tab.
  {{"kind": "text",      "heading": "<optional>", "body": "<optional>", "level": "h1|h2|h3|lead"}}
        // body is rendered as Markdown. level="lead" gives a large muted
        // intro paragraph for the top of a tab.
  {{"kind": "image",     "src": "https://...", "alt": "...", "caption": "<optional>"}}
        // Inline image. Use for static maps (NASA FIRMS PNG bbox screenshot,
        // OpenStreetMap tile), org logos, satellite imagery.
  {{"kind": "carousel",  "title": "<optional>",
                         "items": [{{"heading": "...", "body": "...", "link": "https://..."}}]}}
        // Auto-rotating cards. Great for "Today's headlines" — same data as
        // news_list but more cinematic. 4-second autoplay.
  {{"kind": "scatter",   "title": "<optional>", "data": [{{"x":1,"y":2}}],
                         "x_key": "x", "y_key": "y"}}
        // Scatter plot — perfect for "PM2.5 vs wind speed" correlation,
        // or "AQI vs temperature" causal hints.
  {{"kind": "calendar_picker", "title": "<optional>", "name": "d", "value": "2026-04-27"}}
        // Date-picker calendar. Use when the user asks for "pick a day",
        // "compare two days", "drill into a specific date".
  {{"kind": "combobox",  "title": "<optional>", "name": "city",
                         "options": ["Delhi","Mumbai","Pune"], "value": "Delhi"}}
        // Searchable dropdown. Use for "Compare against …", "Focus city",
        // "AQI scale (US EPA / European / NAQI)". Use {{"kind":"select"}}
        // for the simpler non-searchable variant.
  {{"kind": "blockquote", "body": "...", "cite": "<optional source>"}}
        // <blockquote> — use for WHO advisories, expert quotes, mission
        // statements, anything you'd want to visually pull off the page.
  {{"kind": "kbd",       "keys": ["R"], "label": "<optional caption>"}}
        // Render keyboard-shortcut hints inline.

Composition rules:
- Always start with a high-impact "Overview" tab using the pre-computed
  `overview_metrics` payload as a `metric_grid` widget (4 cards in one row),
  followed by the pre-computed `aqi_radial` as a `radial` gauge AND the
  `dominant_per_city_bar` payload as a `bar` chart. This gives an instant
  visual read on the network. NEVER stack 4 separate `stat` widgets — use
  `metric_grid`.
- For "trend" / "24h" / "history" prefer `area` (smooth, filled) over `line`
  (x_key="hour", y_keys = list of city names). Use `line` only if the user
  explicitly asks for "line chart".
- For "pollutant breakdown" / "fingerprint" / "compare pollutants" use a
  `radar` widget driven by the pre-computed `pollutant_radar` payload —
  much more striking than a grouped bar.
- For "heatmap" / "hour-of-day" / "calendar" requests on ONE city use
  `heatstrip`. For "weekly heatmap" / "7-day pattern" use `heatmap_7d`
  (the data is in `aqi_7d_grid`).
- For ANY important headline finding (worst breach, best outdoor window,
  smoke direction, fire upwind warning) use an `alert` widget with a
  matching variant + icon. Don't bury these in `text`.
- For long lists of items (hotspots, pollutant rows, news stations) prefer
  the `table` (DataTable) widget — it is sortable + paginated.
- ALWAYS add a small "Why" / "Insight" `text` widget at the END of each tab
  using the EXACT body provided in `tab_insights` below.
- For "headlines" / "news" you may use either `news_list` (a vertical list
  of cards with clickable links — best for ≤ 5 items) or `carousel` (an
  auto-rotating banner — best for "show me the latest" demos).
- For static maps, satellite imagery, or any external PNG (e.g. NASA FIRMS
  bbox map) use `image`. The renderer will lazy-load it and show a caption.
- For "drill into a specific day" or "compare two dates" use
  `calendar_picker`. For a "focus city" / "compare against" picker use
  `combobox` (searchable) or `select` (simple).
- For pulled-out quotes (WHO advisories, expert opinions) use `blockquote`.
- Add a `kbd` widget anywhere you want to advertise a keyboard shortcut
  (e.g. {{"kind":"kbd","keys":["R"],"label":"Refresh data"}}).
- Pick tab names that fit the user's request (e.g. "Overview", "Pollutants",
  "24h Trend", "7-Day Heatmap", "Compare", "Health", "Background").
- For "why is X polluted", "WHO limits", "what's the worst pollutant in X" →
  use `who_breach`. For "when should I run", "safest time outside",
  "best outdoor window" → use `outdoor_window`. For "news" / "headlines" /
  "what's happening" → use `news_list`. For a pasted URL or "read this
  page" → use `webpage_summary`. For "CPCB" / "aqi.in" / "station data" /
  "second opinion on AQI" → use `aqicn_compare`. For "why is AQI high",
  "where is the smoke from", "fires near X", "stubble burning",
  "pollution source", "what's causing the spike" → use `pollution_source`.
  These are PLACEHOLDER widgets: emit only the small set of fields shown —
  the renderer fetches the data.
- AQI is on the US EPA scale (0-500). Color-coded `badge` mapping:
  Good (≤50)  → success; Moderate (≤100) → default;
  Unhealthy for Sensitive Groups (≤150) / Unhealthy (≤200) → warning;
  Very Unhealthy / Hazardous (>200) → destructive.
- If the user is MODIFYING a previous dashboard, keep unchanged tabs/widgets
  intact and only edit what they asked for.

eco_log.json (REAL data — use exactly these numbers):
{eco_data}

Authoritative summary (PRE-COMPUTED — use these EXACT numbers for any
"cities tracked", "network avg AQI", "cleanest city", "most polluted city"
stat cards. Do NOT recount or recompute):
{summary}

Tab insights (PRE-COMPUTED 1-line plain-English takeaways — drop the body
into a `text` widget with heading "Why" or "Insight" at the end of the
matching tab. Use the EXACT wording, do not rewrite):
{tab_insights}

Derived chart payloads (PRE-COMPUTED, chart-ready — do NOT aggregate or
recount yourself, just COPY these arrays verbatim):
  - overview_metrics: a metric_grid widget. Use it like
      {{"kind":"metric_grid", "columns": <columns>, "items": <items>}}
    Drop this at the very TOP of the Overview tab.
  - aqi_radial: a radial-gauge widget. Use it like
      {{"kind":"radial", "title":"AQI gauge",
        "data": <data>, "name_key":"name", "value_key":"value"}}
  - pollutant_radar: a radar widget. Use it like
      {{"kind":"radar", "title":"Pollutant fingerprint",
        "data": <data>, "axis_key":"axis", "y_keys": <y_keys>}}
  - dominant_per_city_bar: a bar widget. Use it like
      {{"kind":"bar", "title":"Dominant pollutant by city",
        "data": <rows>, "x_key": "city", "y_keys":["aqi"]}}
    Each row already has a `dominant` field — mention it in the surrounding
    `text` widget rather than the chart itself.
  - dominant_pollutant_pie: a pie widget. Use it like
      {{"kind":"pie", "title":"Dominant pollutant share",
        "data": <data>, "name_key":"name", "value_key":"value"}}
  - dominant_pollutant_table: a table widget. Use it like
      {{"kind":"table", "title":"Dominant pollutant per city",
        "columns": <columns>, "rows": <rows>}}
For ANY widget about "overview", "dominant pollutant", or "pollutant
fingerprint", you MUST use one of these pre-built payloads. Never emit an
empty `data: []` for these — if a payload is missing it means there are no
cities yet, in which case skip the widget entirely:
{derived_charts}

Pre-pass change notes (already applied to eco_log.json before you saw it):
{change_notes}

Previous dashboard spec (or null if first run):
{current_spec}

User request:
{user_request}
"""


def _compute_insights(entries: list[dict]) -> dict[str, str]:
    """Pre-compute one-line plain-English takeaways for each likely tab.
    Returned as {suggested_tab_name: body_string} — the planner is told to
    drop these verbatim into a `text` widget at the end of the matching tab.
    Doing this in Python (not in the LLM) means no hallucinated numbers."""
    out: dict[str, str] = {}
    if not entries:
        return out

    n = len(entries)
    cleanest = min(entries, key=lambda e: e.get("aqi", 1e9))
    worst = max(entries, key=lambda e: e.get("aqi", -1))
    avg = round(sum(e.get("aqi", 0) for e in entries) / n, 1)

    # Overview / Right Now
    out["Overview"] = (
        f"Across {n} city/cities the network avg AQI is {avg:.0f}. "
        f"{cleanest.get('city')} is cleanest right now ({cleanest.get('aqi'):.0f}, "
        f"{cleanest.get('band')}); {worst.get('city')} is worst "
        f"({worst.get('aqi'):.0f}, {worst.get('band')})."
    )

    # Pollutants — flag the worst WHO-limit breach across all cities.
    who = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40, "co": 4000}
    breaches = []
    for e in entries:
        for k, limit in who.items():
            v = e.get(k, 0) or 0
            if v > limit:
                breaches.append((v / limit, e.get("city"), k.upper(), v, limit))
    if breaches:
        breaches.sort(reverse=True)
        ratio, city, p, v, limit = breaches[0]
        out["Pollutants"] = (
            f"Worst WHO breach: {city} {p} = {v} µg/m³, "
            f"{ratio:.1f}× the 24-hour guideline of {limit}."
        )
    else:
        out["Pollutants"] = "No city exceeds WHO 24-hour pollutant guidelines."

    # 24h Trend — find the city + hour with the single highest AQI.
    peak_city, peak_hour, peak_aqi = None, 0, -1.0
    for e in entries:
        h = e.get("hourly_aqi") or []
        if not h:
            continue
        local_peak = max(h)
        if local_peak > peak_aqi:
            peak_aqi = local_peak
            peak_city = e.get("city")
            peak_hour = h.index(local_peak)
    if peak_city is not None:
        out["24h Trend"] = (
            f"The single worst hour observed across all cities was "
            f"{peak_city} at {peak_hour:02d}:00 (AQI {peak_aqi:.0f})."
        )

    # Hour pattern — call out the most common worst-hour across cities.
    worst_hours = []
    for e in entries:
        h = e.get("hourly_aqi") or []
        if h:
            worst_hours.append(h.index(max(h)))
    if worst_hours:
        from collections import Counter
        common_hour, count = Counter(worst_hours).most_common(1)[0]
        out["Hour Pattern"] = (
            f"In {count} of {len(worst_hours)} cities the daily peak occurs "
            f"around {common_hour:02d}:00 — schedule outdoor activity earlier."
        )

    # 7-day heatmap — describe the worst day across all cities.
    worst_day, worst_day_avg, worst_day_city = None, -1.0, None
    for e in entries:
        for day in (e.get("aqi_7d_grid") or []):
            vals = [v for v in (day.get("hourly") or []) if v is not None]
            if not vals:
                continue
            d_avg = sum(vals) / len(vals)
            if d_avg > worst_day_avg:
                worst_day_avg = d_avg
                worst_day = day.get("date")
                worst_day_city = e.get("city")
    if worst_day:
        out["7-Day Heatmap"] = (
            f"Worst day in the last week: {worst_day} in {worst_day_city} "
            f"(daily avg AQI {worst_day_avg:.0f})."
        )

    # Compare
    if n >= 2:
        spread = worst.get("aqi", 0) - cleanest.get("aqi", 0)
        out["Compare"] = (
            f"AQI spread across the {n} tracked cities is {spread:.0f} points "
            f"({cleanest.get('city')} {cleanest.get('aqi'):.0f} → "
            f"{worst.get('city')} {worst.get('aqi'):.0f})."
        )

    # Health — point to whichever city has the worst advisory.
    out["Health"] = (
        f"Highest-risk city: {worst.get('city')} — "
        f"{worst.get('advisory', 'limit outdoor activity')}"
    )

    return out


def plan(user_request: str, current_spec: dict | None,
         change_notes: list[str]) -> dict:
    entries = eco._load()
    eco_data = json.dumps(entries, indent=2)
    if len(eco_data) > 12000:
        eco_data = eco_data[:12000] + "\n... (truncated)"

    # Pre-compute authoritative summary stats. The LLM is unreliable at
    # counting list items / picking min-max, so we hand it the answers.
    if entries:
        cleanest = min(entries, key=lambda e: e.get("aqi", 1e9))
        worst = max(entries, key=lambda e: e.get("aqi", -1))
        avg_aqi = round(sum(e.get("aqi", 0) for e in entries) / len(entries), 1)
        summary = {
            "cities_tracked": len(entries),
            "city_names": [e.get("city", "?") for e in entries],
            "network_avg_aqi": avg_aqi,
            "cleanest_city": cleanest.get("city"),
            "cleanest_aqi": cleanest.get("aqi"),
            "most_polluted_city": worst.get("city"),
            "most_polluted_aqi": worst.get("aqi"),
        }
    else:
        summary = {"cities_tracked": 0, "city_names": [],
                   "network_avg_aqi": 0,
                   "cleanest_city": None, "cleanest_aqi": None,
                   "most_polluted_city": None, "most_polluted_aqi": None}

    # Pre-compute "derived" chart payloads. The LLM is unreliable at any
    # form of aggregation (counting cities per dominant pollutant, summing
    # bands, etc.), so we ship the chart-ready arrays here and tell the
    # planner to copy them verbatim.
    derived_charts: dict = {}
    if entries:
        # 1) Dominant pollutant per city — bar chart, x_key="city",
        #    y_keys=["aqi"]. The LLM needs only this rectangular shape.
        derived_charts["dominant_per_city_bar"] = {
            "x_key": "city",
            "y_keys": ["aqi"],
            "rows": [
                {
                    "city": e.get("city", "?"),
                    "dominant": (e.get("dominant", "") or "").upper(),
                    "aqi": e.get("aqi", 0),
                }
                for e in entries
            ],
        }
        # 2) Dominant-pollutant city counts — perfect for a Pie chart with
        #    name_key="name", value_key="value".
        from collections import Counter
        counts = Counter(
            (e.get("dominant", "") or "?").upper() for e in entries
        )
        derived_charts["dominant_pollutant_pie"] = {
            "name_key": "name", "value_key": "value",
            "data": [{"name": k, "value": v} for k, v in counts.most_common()],
        }
        # 3) Dominant pollutant table rows.
        derived_charts["dominant_pollutant_table"] = {
            "columns": ["City", "Dominant pollutant", "Current AQI", "Band"],
            "rows": [
                [e.get("city", ""), (e.get("dominant", "") or "?").upper(),
                 f"{e.get('aqi', 0):.0f}", e.get("band", "")]
                for e in entries
            ],
        }
        # 4) Pollutant radar — one polygon per city across the 6 pollutants.
        #    The LLM should drop this into a `radar` widget so each city's
        #    "pollutant fingerprint" is visually obvious.
        pollutants = ["pm25", "pm10", "no2", "o3", "so2", "co"]
        radar_rows = []
        for p in pollutants:
            row = {"axis": p.upper()}
            for e in entries:
                row[e.get("city", "?")] = e.get(p, 0) or 0
            radar_rows.append(row)
        derived_charts["pollutant_radar"] = {
            "axis_key": "axis",
            "y_keys": [e.get("city", "?") for e in entries],
            "data": radar_rows,
        }
        # 5) Overview metric grid — 4 big "Metric" cards for the hero row.
        derived_charts["overview_metrics"] = {
            "columns": min(4, max(1, len(entries))),
            "items": [
                {"label": "Cities tracked", "value": str(len(entries)),
                 "description": ", ".join(e.get("city", "") for e in entries)[:60]},
                {"label": "Network avg AQI", "value": f"{avg_aqi:.0f}",
                 "description": "average across all tracked cities"},
                {"label": "Cleanest city",
                 "value": str(cleanest.get("city", "")),
                 "description": f"AQI {cleanest.get('aqi', 0):.0f} · {cleanest.get('band', '')}",
                 "trend": "down", "trend_sentiment": "positive"},
                {"label": "Most polluted",
                 "value": str(worst.get("city", "")),
                 "description": f"AQI {worst.get('aqi', 0):.0f} · {worst.get('band', '')}",
                 "trend": "up", "trend_sentiment": "negative"},
            ],
        }
        # 6) Half-donut RadialChart payload — visual gauge of every city's AQI
        #    on a single chart (capped at 500 for readability).
        derived_charts["aqi_radial"] = {
            "value_key": "value", "name_key": "name",
            "data": [
                {"name": e.get("city", "?"),
                 "value": min(500, int(e.get("aqi", 0) or 0))}
                for e in entries
            ],
        }

    prompt = PLANNER_PROMPT.format(
        user_request=user_request,
        current_spec=json.dumps(current_spec) if current_spec else "null",
        change_notes="\n".join(change_notes) if change_notes else "(none)",
        eco_data=eco_data,
        summary=json.dumps(summary, indent=2),
        tab_insights=json.dumps(_compute_insights(entries), indent=2),
        derived_charts=json.dumps(derived_charts, indent=2),
    )

    def _parse_planner_json(text: str) -> dict:
        """Tolerant parser: strip code fences, slice to outer braces, parse."""
        s = (text or "").strip()
        if s.startswith("```"):
            # ```json ... ```  or  ``` ... ```
            s = s.strip("`")
            if "\n" in s:
                first, rest = s.split("\n", 1)
                if first.lower().strip() in ("json", ""):
                    s = rest
            if s.endswith("```"):
                s = s[:-3]
        # Slice to the outermost JSON object — drops any leading prose.
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            s = s[i:j + 1]
        return json.loads(s)

    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = response.text or ""
    try:
        return _parse_planner_json(raw)
    except json.JSONDecodeError as e:
        print(f"  (planner JSON malformed: {e}; retrying with stricter ask)")
        # Retry once with an explicit small-output reminder.
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: your previous reply was not valid JSON "
              "(error: " + str(e) + "). Return ONLY a single JSON object, "
              "no prose, no code fences. Keep arrays SHORT — for "
              "heatmap_7d widgets pass only {\"kind\":\"heatmap_7d\","
              "\"city\":\"<city>\"} (the data will be filled in from "
              "eco_log.json by the renderer)."
        )
        response = client.models.generate_content(model=MODEL, contents=retry_prompt)
        return _parse_planner_json(response.text or "")


# ---------------------------------------------------------------------------
# 4. Writer — render the spec to disk, syntax-check, restart prefab serve.
# ---------------------------------------------------------------------------

def write_app(spec: dict) -> None:
    title = spec.get("title", "🌍 India Climate Tracker")
    tabs = spec.get("tabs", [])
    source = render_dashboard(title, tabs)
    compile(source, "<generated_eco_app>", "exec")  # syntax check
    GENERATED.write_text(source, encoding="utf-8")
    os.utime(GENERATED, None)
    print(f"  → wrote {GENERATED.name} ({source.count(chr(10))} lines)")


def save_backup() -> None:
    if GENERATED.exists():
        BACKUP.write_text(GENERATED.read_text(encoding="utf-8"), encoding="utf-8")


def restore_backup() -> bool:
    if BACKUP.exists():
        GENERATED.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    return False


def tail_log(n: int = 30) -> str:
    try:
        return "\n".join(LOG_PATH.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-n:])
    except Exception as e:
        return f"(log unreadable: {e})"


class PrefabServer:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._log = None

    def start(self):
        self._log = open(LOG_PATH, "a")
        self._log.write("\n===== restart =====\n"); self._log.flush()
        self._proc = subprocess.Popen(
            ["prefab", "serve", str(GENERATED)],
            cwd=GENERATED.parent, stdout=self._log, stderr=subprocess.STDOUT,
        )

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try: self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill(); self._proc.wait()
            self._proc = None
        if self._log is not None:
            self._log.close(); self._log = None

    def restart(self):
        self.stop(); self.start()


# ---------------------------------------------------------------------------
# 4b. In-UI Refresh server — a tiny HTTP endpoint the generated dashboard
#     POSTs to when the user clicks the "Refresh data" button. Runs in a
#     background daemon thread so it doesn't block the REPL.
#
#     Pattern: Prefab's Fetch action fires from the browser → this handler
#     re-runs build_city_report for every tracked city in parallel → it
#     `os.utime`s generated_eco_app.py so Prefab's file-watcher detects a
#     change and hot-reloads the open browser tab. Zero Python callables on
#     Prefab's click path (which it does not allow); everything declarative.
# ---------------------------------------------------------------------------

import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

REFRESH_PORT = 5180

# Set by main() after each successful plan() so the refresh handler can
# re-render the dashboard with a fresh `updated …` timestamp + freshly
# hydrated live widgets (heatmap_7d, who_breach, outdoor_window, news_list,
# aqicn_compare, pollution_source). Without this the file's mtime bumps
# but the hero subtitle and any render-time-fetched data stay frozen.
_LAST_SPEC: dict | None = None

# Set by main() so the refresh handler can bounce `prefab serve` after
# rewriting generated_eco_app.py. `prefab serve` doesn't reliably hot-reload
# on a plain mtime bump from another process, so we mirror what the REPL
# does after every prompt: write file → restart subprocess → browser sees
# the new app on its next websocket reconnect.
_PREFAB_SERVER: "PrefabServer | None" = None


class _RefreshHandler(BaseHTTPRequestHandler):
    """Tiny request handler with permissive CORS so the Prefab dev server
    (5175) can call this server (5180) from the browser."""

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:        # noqa: N802  (BaseHTTPRequestHandler API)
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:           # noqa: N802
        if self.path.rstrip("/") != "/refresh":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        cities: list[str] = []
        try:
            cities = [e.get("city") for e in eco._load() if e.get("city")]
        except Exception as ex:
            print(f"  [refresh] cannot read eco_log.json: {ex}")

        results: list[str] = []
        if cities:
            print(f"  [refresh] re-fetching {len(cities)} city/cities in parallel: "
                  f"{', '.join(cities)}")
            # Use a small thread-pool so all cities are re-fetched concurrently
            # — same speedup the talk_eco REPL gets for its parallel ADD path.
            with ThreadPoolExecutor(max_workers=min(8, len(cities))) as pool:
                futures = {pool.submit(eco.build_city_report, c): c for c in cities}
                for fut in futures:
                    c = futures[fut]
                    try:
                        fut.result(timeout=30)
                        results.append(f"OK {c}")
                    except Exception as ex:
                        results.append(f"FAIL {c}: {ex}")
                        print(f"    ↳ refresh failed for {c}: {ex}")

        # Re-render the dashboard so the hero "updated …" timestamp advances
        # AND any render-time-hydrated widgets (heatmap_7d, who_breach,
        # outdoor_window, news_list, aqicn_compare, pollution_source) pick
        # up the freshly fetched numbers. Falls back to a plain mtime bump
        # if no spec has been planned yet.
        try:
            if _LAST_SPEC is not None:
                write_app(_LAST_SPEC)
            elif GENERATED.exists():
                os.utime(GENERATED, None)
        except Exception as ex:
            print(f"  [refresh] re-render failed ({ex}); falling back to mtime bump")
            try:
                os.utime(GENERATED, None)
            except Exception:
                pass

        # Bounce `prefab serve` so the browser actually picks up the new
        # file. Without this the websocket stays connected to the old
        # process and the page never reloads even though mtime changed.
        if _PREFAB_SERVER is not None:
            try:
                _PREFAB_SERVER.restart()
                print("  [refresh] prefab serve restarted — reload imminent.")
            except Exception as ex:
                print(f"  [refresh] prefab restart failed: {ex}")

        body = json.dumps({"refreshed": cities, "results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)
        print(f"  [refresh] done — browser will hot-reload momentarily.")

    # Suppress the noisy default access log; we print our own one-liner.
    def log_message(self, fmt: str, *args) -> None:    # noqa: D401
        return


def start_refresh_server() -> HTTPServer:
    """Spawn the refresh HTTP server in a daemon thread on REFRESH_PORT."""
    httpd = HTTPServer(("127.0.0.1", REFRESH_PORT), _RefreshHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="refresh-server")
    t.start()
    return httpd


# ---------------------------------------------------------------------------
# 5. REPL.
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 66)
    print("🌍 AirSense — describe an air-quality view, watch the browser morph.")
    print("=" * 66)
    print("Internet  : Open-Meteo (free, no key)")
    print("CRUD file : eco_log.json (Create/Update on add/refresh, Delete on remove)")
    print("UI        : Prefab dashboard regenerated on every prompt")
    print()
    print("Try prompts like:")
    print("  • Add Delhi, Mumbai, and Bangalore.")
    print("  • Show the 24-hour AQI as a heatstrip for each city.")
    print("  • Compare pollutants across all cities.")
    print("  • Refresh Delhi only.")
    print("  • Remove Mumbai.")
    print("  • A health-advisory dashboard for the worst city.")
    print("Type 'quit' to exit.\n")

    # Boot placeholder app so prefab has something to serve.
    GENERATED.write_text(
        "from prefab_ui.app import PrefabApp\n"
        "from prefab_ui.components import Card, CardContent, CardHeader, CardTitle, Muted\n\n"
        'with PrefabApp(css_class="max-w-md mx-auto p-6") as app:\n'
        "    with Card():\n"
        "        with CardHeader():\n"
        '            CardTitle("AirSense")\n'
        "        with CardContent():\n"
        '            Muted("Waiting for your first prompt in the terminal...")\n',
        encoding="utf-8",
    )
    LOG_PATH.write_text("")
    server = PrefabServer()
    print(f"Starting Prefab dev server (logs → {LOG_PATH.name}) …")
    server.start(); time.sleep(1.5)
    global _PREFAB_SERVER
    _PREFAB_SERVER = server  # let the refresh handler bounce it
    refresh_httpd = start_refresh_server()
    print(f"Refresh endpoint up on http://127.0.0.1:{REFRESH_PORT}/refresh "
          "(used by the in-UI Refresh button).")
    print("Open http://127.0.0.1:5175 in your browser.\n")

    current_spec: dict | None = None
    try:
        while True:
            try:
                prompt = input("\nWhat do you want to see? ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                # Empty Enter is a no-op — keep the Prefab server alive so
                # the in-UI Refresh button (and the open browser tab) keep
                # working. Use "quit" / "exit" / "q" / Ctrl-D to leave.
                continue
            if prompt.lower() in {"quit", "exit", "q"}:
                break

            try:
                # 1) data layer first — real Open-Meteo + real CRUD
                print("  [data]")
                notes = apply_data_changes(prompt)

                # 2) plan + render
                print("  [plan]")
                spec = plan(prompt, current_spec, notes)
                short = json.dumps(spec)
                print(f"  spec: {short[:200]}{'...' if len(short) > 200 else ''}")
                save_backup()
                write_app(spec)
                server.restart(); time.sleep(1.5)

                # 3) crash detection
                tail = tail_log(30)
                broken = ((server._proc and server._proc.poll() is not None)
                          or "traceback" in tail.lower()
                          or "exception" in tail.lower())
                if broken:
                    print("\n  ✗ Prefab did not come up cleanly.")
                    print("  --- last lines of prefab_server.log ---")
                    print(tail)
                    print("  ----------------------------------------")
                    if restore_backup():
                        print("  Reverting to last working app …")
                        server.restart(); time.sleep(1.0)
                else:
                    current_spec = spec
                    global _LAST_SPEC
                    _LAST_SPEC = spec  # so the Refresh button can re-render
                    print("  (browser will reconnect in a moment)")
            except Exception as e:
                print(f"  error: {e}")
    finally:
        print("\nShutting down Prefab server …")
        server.stop()
        try:
            refresh_httpd.shutdown()
            refresh_httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
