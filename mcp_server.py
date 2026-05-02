"""
Climate & Air-Quality MCP server for Indian cities.

All internet calls use FREE, NO-KEY public APIs:
  - Open-Meteo Geocoding   https://geocoding-api.open-meteo.com/
  - Open-Meteo Forecast    https://api.open-meteo.com/
  - Open-Meteo Air Quality https://air-quality-api.open-meteo.com/

Tools:
  Internet (4):
    1. fetch_aqi_now(city)      -> JSON {city, aqi, band, dominant, pm25, pm10, no2, o3, so2, co}
    2. fetch_aqi_24h(city)      -> JSON {city, hourly:[...], peak_hour, peak_aqi, min_hour, min_aqi, avg_aqi}
    3. fetch_aqi_7d(city)       -> JSON {city, days:[{date, hourly:[24]}], min, max, avg}
    4. fetch_weather_now(city)  -> JSON {city, temp_c, tmax_c, tmin_c, uv_index, humidity, condition}
    +  build_city_report(city)  -> one-shot parallel fetch + save (1+2+3+4)
    +  fetch_aqi_forecast(city) -> next-N-hour US AQI forecast
    +  fetch_news_for(city)     -> Google News RSS headlines (different domain)

  Local CRUD (4) on eco_log.json:
    save_city_report / list_log / get_city_report / remove_city

  Analytics (4 — pure Python, no internet):
    who_breach_summary(city)         -> per-pollutant ratio over WHO 24h limit
    time_of_day_profile(city)        -> morning/afternoon/evening/night + safest 3h
    recommend_outdoor_window(city)   -> lowest-AQI N-hour block in next 24h
    (compare/aggregate done in talk_eco.py)

  UI (1):
    show_dashboard()         -> @mcp.tool(app=True) Prefab dashboard (5 tabs)

Run:
    python mcp_server.py                 # stdio (used by agent.py)
    fastmcp dev apps mcp_server.py       # browse the dashboard at http://localhost:6274
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load .env BEFORE any tool runs. Critical when this file is launched as a
# stdio MCP subprocess (e.g. by talk_eco.py) — in that case nothing else has
# run python-dotenv on our behalf, so FIRMS_MAP_KEY / AQICN_TOKEN would be
# silently missing and tools like fire_hotspots_near / fetch_aqicn would fail.
load_dotenv(Path(__file__).parent / ".env")
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge, Card, CardContent, CardHeader, CardTitle,
    Column, H1, H2, H3, Muted, Ring, Row, Separator, Tab, Tabs, Text,
)
from prefab_ui.components.charts import (
    BarChart, ChartSeries, LineChart, PieChart, Sparkline,
)


HERE = Path(__file__).parent
LOG_FILE = HERE / "eco_log.json"

mcp = FastMCP("ClimateTrackerServer")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(entries: list[dict]) -> None:
    LOG_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                        encoding="utf-8")


# ---------------------------------------------------------------------------
# AQI band helpers (US EPA AQI scale, 0-500)
#   0-50    Good
#   51-100  Moderate
#   101-150 Unhealthy for Sensitive Groups
#   151-200 Unhealthy
#   201-300 Very Unhealthy
#   301+    Hazardous
# This matches what CPCB / news in India report and what most users expect.
# ---------------------------------------------------------------------------

def _band(aqi: float) -> tuple[str, str, str]:
    """Return (band_name, prefab_variant, advisory) for a given US AQI."""
    a = aqi or 0
    if a <= 50:
        return ("Good", "success",
                "Air quality is satisfactory; little or no risk.")
    if a <= 100:
        return ("Moderate", "default",
                "Acceptable, but unusually sensitive people should reduce "
                "prolonged outdoor exertion.")
    if a <= 150:
        return ("Unhealthy for Sensitive Groups", "warning",
                "Children, elderly and people with heart/lung disease "
                "should limit outdoor exertion.")
    if a <= 200:
        return ("Unhealthy", "warning",
                "Everyone may begin to feel effects; sensitive groups "
                "should avoid prolonged outdoor exertion.")
    if a <= 300:
        return ("Very Unhealthy", "destructive",
                "Health alert: serious effects possible. Limit outdoor "
                "activity; wear an N95 if going out.")
    return ("Hazardous", "destructive",
            "Health emergency: stay indoors with windows closed, run an "
            "air purifier, wear an N95 if you must go out.")


# Aliases → canonical city name. Keys are lowercased and stripped on lookup.
# Prevents the "Bangalore vs Bengaluru" duplication in eco_log.json.
_CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengalooru": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi",
    "mysore": "Mysuru",
    "mangalore": "Mangaluru",
    "vizag": "Visakhapatnam",
    "pondicherry": "Puducherry",
    "baroda": "Vadodara",
    "poona": "Pune",
    "gurgaon": "Gurugram",
    "allahabad": "Prayagraj",
}


def _canonical(name: str) -> str:
    """Map common aliases to the canonical Indian city name."""
    return _CITY_ALIASES.get((name or "").strip().lower(), name)


def _geocode(city: str) -> dict:
    # Resolve aliases BEFORE hitting the API so geocoding hits the same lat/lon
    # for both "Bangalore" and "Bengaluru".
    query = _canonical(city)
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = (r.json() or {}).get("results") or []
    if not results:
        raise ValueError(f"city '{city}' not found")
    g = results[0]
    # Force the canonical name even if the API returns a different spelling.
    canonical_name = _canonical(g.get("name", query)) or query
    return {
        "name": canonical_name,
        "country": g.get("country", ""),
        "lat": g["latitude"],
        "lon": g["longitude"],
        "timezone": g.get("timezone", "Asia/Kolkata"),
    }


# ---------------------------------------------------------------------------
# 1. Internet — current air quality
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_aqi_now(city: str) -> str:
    """Fetch the current air quality for a city. Returns JSON with the
    US AQI (0-500), band ('Good'..'Hazardous'), dominant pollutant, and
    pm25 / pm10 / no2 / o3 / so2 / co values (µg/m³)."""
    try:
        g = _geocode(city)
    except Exception as e:
        return f"ERROR: {e}"

    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": g["lat"], "longitude": g["lon"],
            "current": "us_aqi,pm10,pm2_5,carbon_monoxide,"
                       "nitrogen_dioxide,sulphur_dioxide,ozone",
            "timezone": g["timezone"],
        },
        timeout=10,
    )
    r.raise_for_status()
    cur = (r.json() or {}).get("current", {}) or {}
    aqi = cur.get("us_aqi") or 0
    pm25 = cur.get("pm2_5") or 0
    pm10 = cur.get("pm10") or 0
    no2 = cur.get("nitrogen_dioxide") or 0
    o3 = cur.get("ozone") or 0
    so2 = cur.get("sulphur_dioxide") or 0
    co = cur.get("carbon_monoxide") or 0

    # Dominant pollutant = highest ratio over its WHO 24h guideline.
    who = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40, "co": 4000}
    vals = {"pm25": pm25, "pm10": pm10, "no2": no2, "o3": o3, "so2": so2, "co": co}
    dominant = max(vals, key=lambda k: (vals[k] or 0) / who[k])

    band, _, advisory = _band(aqi)
    out = {
        "city": g["name"], "country": g["country"],
        "lat": g["lat"], "lon": g["lon"],
        "aqi": round(aqi, 1), "band": band, "advisory": advisory,
        "dominant": dominant,
        "pm25": round(pm25, 1), "pm10": round(pm10, 1),
        "no2": round(no2, 1), "o3": round(o3, 1),
        "so2": round(so2, 1), "co": round(co, 1),
        "fetched_at": cur.get("time", datetime.utcnow().isoformat()),
    }
    return json.dumps(out)


# ---------------------------------------------------------------------------
# 2. Internet — 24-hour AQI history (hourly)
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_aqi_24h(city: str) -> str:
    """Fetch the last 24 hours of hourly US AQI for a city. Returns
    JSON with the hourly array (24 values), peak hour/value, min hour/value,
    and 24-hour average. Also returns hourly PM2.5 for sparklines."""
    try:
        g = _geocode(city)
    except Exception as e:
        return f"ERROR: {e}"

    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": g["lat"], "longitude": g["lon"],
            "hourly": "us_aqi,pm2_5",
            "timezone": g["timezone"],
            "past_days": 1, "forecast_days": 1,
        },
        timeout=10,
    )
    r.raise_for_status()
    h = (r.json() or {}).get("hourly", {}) or {}
    times = h.get("time") or []
    aqis = h.get("us_aqi") or []
    pm25s = h.get("pm2_5") or []

    # Take the most recent 24 entries that are <= "now" in city time.
    now_iso = datetime.utcnow().isoformat()
    # Pair and sort, keep last 24.
    paired = list(zip(times, aqis, pm25s))
    paired = [p for p in paired if p[1] is not None][-24:]
    if not paired:
        return f"ERROR: no hourly data for {city}"

    hourly = [
        {"time": t, "hour": int(t.split("T")[1].split(":")[0]),
         "aqi": round(a, 1), "pm25": round(p or 0, 1)}
        for t, a, p in paired
    ]
    aqis_only = [row["aqi"] for row in hourly]
    peak_idx = aqis_only.index(max(aqis_only))
    min_idx = aqis_only.index(min(aqis_only))
    avg = round(sum(aqis_only) / len(aqis_only), 1)

    out = {
        "city": g["name"],
        "hourly": hourly,
        "peak_hour": hourly[peak_idx]["hour"],
        "peak_aqi": hourly[peak_idx]["aqi"],
        "min_hour": hourly[min_idx]["hour"],
        "min_aqi": hourly[min_idx]["aqi"],
        "avg_aqi": avg,
    }
    return json.dumps(out)


# ---------------------------------------------------------------------------
# 2b. Internet — 7 days × 24 hours of AQI for the heatmap grid.
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_aqi_7d(city: str) -> str:
    """Fetch the last 7 days of hourly US AQI for a city — 168 values shaped
    as a 7×24 grid (rows = days oldest→newest, cols = hours 00..23).
    Returns JSON {city, days:[{date, hourly:[24 values]}], min, max, avg}."""
    try:
        g = _geocode(city)
    except Exception as e:
        return f"ERROR: {e}"

    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": g["lat"], "longitude": g["lon"],
            "hourly": "us_aqi",
            "timezone": g["timezone"],
            "past_days": 7, "forecast_days": 0,
        },
        timeout=12,
    )
    r.raise_for_status()
    h = (r.json() or {}).get("hourly", {}) or {}
    times = h.get("time") or []
    aqis = h.get("us_aqi") or []
    if not times:
        return f"ERROR: no 7-day hourly data for {city}"

    # Group by date (YYYY-MM-DD); keep only complete-ish days (>=12 hours).
    by_date: dict[str, list[float | None]] = {}
    for t, a in zip(times, aqis):
        date = t.split("T")[0]
        hour = int(t.split("T")[1].split(":")[0])
        row = by_date.setdefault(date, [None] * 24)
        if 0 <= hour < 24:
            row[hour] = round(a, 1) if a is not None else None

    days = []
    for date in sorted(by_date):
        row = by_date[date]
        if sum(1 for v in row if v is not None) >= 12:
            days.append({"date": date, "hourly": row})
    days = days[-7:]  # most recent 7 days

    flat = [v for d in days for v in d["hourly"] if v is not None]
    out = {
        "city": g["name"],
        "days": days,
        "min": round(min(flat), 1) if flat else 0,
        "max": round(max(flat), 1) if flat else 0,
        "avg": round(sum(flat) / len(flat), 1) if flat else 0,
    }
    return json.dumps(out)


# ---------------------------------------------------------------------------
# 3. Internet — current weather
# ---------------------------------------------------------------------------

_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Heavy showers", 82: "Violent showers", 95: "Thunderstorm",
    96: "Thunderstorm w/ hail", 99: "Severe thunderstorm",
}


@mcp.tool()
def fetch_weather_now(city: str) -> str:
    """Fetch current weather + today's max/min temp + max UV index for a city.
    Returns JSON with temp_c, tmax_c, tmin_c, uv_index, humidity, condition."""
    try:
        g = _geocode(city)
    except Exception as e:
        return f"ERROR: {e}"

    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": g["lat"], "longitude": g["lon"],
            # wind_speed_10m + wind_direction_10m make the FIRMS fire-hotspot
            # insight actionable ("smoke blowing TOWARD city" vs "AWAY").
            "current": ("temperature_2m,relative_humidity_2m,weather_code,"
                        "wind_speed_10m,wind_direction_10m"),
            "daily": "temperature_2m_max,temperature_2m_min,uv_index_max",
            "timezone": g["timezone"], "forecast_days": 1,
        },
        timeout=10,
    )
    r.raise_for_status()
    j = r.json() or {}
    cur = j.get("current", {}) or {}
    daily = j.get("daily", {}) or {}
    code = cur.get("weather_code") or 0

    out = {
        "city": g["name"],
        "lat": g["lat"], "lon": g["lon"],
        "temp_c": round(cur.get("temperature_2m") or 0, 1),
        "humidity": round(cur.get("relative_humidity_2m") or 0, 1),
        "tmax_c": round((daily.get("temperature_2m_max") or [0])[0], 1),
        "tmin_c": round((daily.get("temperature_2m_min") or [0])[0], 1),
        "uv_index": round((daily.get("uv_index_max") or [0])[0], 1),
        "condition": _WEATHER_CODES.get(code, f"code {code}"),
        # Wind: Open-Meteo reports the direction the wind is COMING FROM,
        # in degrees clockwise from true north (0° = N, 90° = E, …).
        "wind_speed_kmh": round(cur.get("wind_speed_10m") or 0, 1),
        "wind_dir_deg":   round(cur.get("wind_direction_10m") or 0, 0),
    }
    return json.dumps(out)


# ---------------------------------------------------------------------------
# 3b. Internet — one-shot batch fetch (parallel) + save.
# Lets the agent build a full city report in ONE MCP call instead of four.
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor  # noqa: E402


@mcp.tool()
def build_city_report(city: str) -> str:
    """Fetch current AQI + 24h hourly AQI + current weather for a city IN
    PARALLEL, then save the merged report to eco_log.json. Replaces the
    4-step (fetch_aqi_now → fetch_aqi_24h → fetch_weather_now →
    save_city_report) sequence with a single MCP call.

    Returns a short summary JSON: {city, aqi, band, peak_hour, peak_aqi,
    avg_aqi, temp_c, action: 'added'|'updated'}."""
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_now = ex.submit(fetch_aqi_now, city)
        f_24h = ex.submit(fetch_aqi_24h, city)
        f_wx = ex.submit(fetch_weather_now, city)
        f_7d = ex.submit(fetch_aqi_7d, city)
        now_s = f_now.result()
        h24_s = f_24h.result()
        wx_s = f_wx.result()
        d7_s = f_7d.result()

    if any(isinstance(s, str) and s.startswith("ERROR:")
           for s in (now_s, h24_s, wx_s)):
        return (f"ERROR: build_city_report failed for {city!r}: "
                f"now={now_s} | 24h={h24_s} | wx={wx_s}")

    now = json.loads(now_s)
    h24 = json.loads(h24_s)
    wx = json.loads(wx_s)
    # The 7-day fetch is best-effort; if it fails we still save the rest.
    d7 = None
    if isinstance(d7_s, str) and not d7_s.startswith("ERROR:"):
        try:
            d7 = json.loads(d7_s)
        except Exception:
            d7 = None

    entry = {
        "city": now["city"],
        "aqi": now["aqi"], "band": now["band"], "dominant": now["dominant"],
        "pm25": now["pm25"], "pm10": now["pm10"], "no2": now["no2"],
        "o3": now["o3"], "so2": now["so2"], "co": now["co"],
        "temp_c": wx["temp_c"], "tmax_c": wx["tmax_c"], "tmin_c": wx["tmin_c"],
        "uv_index": wx["uv_index"], "humidity": wx["humidity"],
        "condition": wx["condition"],
        # Wind + lat/lon power the FIRMS fire-hotspot widget. lat/lon are
        # also handy for any future map / geo widget.
        "lat": wx.get("lat"), "lon": wx.get("lon"),
        "wind_speed_kmh": wx.get("wind_speed_kmh", 0),
        "wind_dir_deg":   wx.get("wind_dir_deg", 0),
        "advisory": now["advisory"],
        "hourly_aqi": [r["aqi"] for r in h24.get("hourly", [])],
        "hourly_pm25": [r["pm25"] for r in h24.get("hourly", [])],
        "peak_hour": h24.get("peak_hour", 0),
        "peak_aqi": h24.get("peak_aqi", 0),
        "avg_aqi": h24.get("avg_aqi", 0),
        # 7d × 24h grid for the heatmap. List of {date, hourly:[24]}.
        "aqi_7d_grid": (d7 or {}).get("days", []),
        "aqi_7d_min": (d7 or {}).get("min", 0),
        "aqi_7d_max": (d7 or {}).get("max", 0),
        "aqi_7d_avg": (d7 or {}).get("avg", 0),
        "checked_at": datetime.utcnow().isoformat(timespec="seconds"),
    }

    entries = _load()
    action = "added"
    for i, e in enumerate(entries):
        if e.get("city", "").lower() == entry["city"].lower():
            entries[i] = entry
            action = "updated"
            break
    else:
        entries.append(entry)
    _save(entries)

    return json.dumps({
        "action": action,
        "city": entry["city"],
        "aqi": entry["aqi"],
        "band": entry["band"],
        "peak_hour": entry["peak_hour"],
        "peak_aqi": entry["peak_aqi"],
        "avg_aqi": entry["avg_aqi"],
        "temp_c": entry["temp_c"],
        "total_cities_in_log": len(entries),
    })


# ---------------------------------------------------------------------------
# 4-7. Local CRUD on eco_log.json
# ---------------------------------------------------------------------------

@mcp.tool()
def save_city_report(
    city: str,
    aqi: float,
    band: str,
    dominant: str,
    pm25: float,
    pm10: float,
    no2: float,
    o3: float,
    so2: float,
    co: float,
    temp_c: float,
    tmax_c: float,
    tmin_c: float,
    uv_index: float,
    humidity: float,
    condition: str,
    advisory: str,
    hourly_aqi: list[float],
    hourly_pm25: list[float],
    peak_hour: int,
    peak_aqi: float,
    avg_aqi: float,
) -> str:
    """Create or update one city's full report in eco_log.json.
    The hourly_aqi / hourly_pm25 lists should be 24 floats (oldest -> newest)."""
    entries = _load()
    entry = {
        "city": city,
        "aqi": aqi, "band": band, "dominant": dominant,
        "pm25": pm25, "pm10": pm10, "no2": no2, "o3": o3,
        "so2": so2, "co": co,
        "temp_c": temp_c, "tmax_c": tmax_c, "tmin_c": tmin_c,
        "uv_index": uv_index, "humidity": humidity, "condition": condition,
        "advisory": advisory,
        "hourly_aqi": list(hourly_aqi)[-24:],
        "hourly_pm25": list(hourly_pm25)[-24:],
        "peak_hour": peak_hour, "peak_aqi": peak_aqi,
        "avg_aqi": avg_aqi,
        "checked_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    for i, e in enumerate(entries):
        if e.get("city", "").lower() == city.lower():
            entries[i] = entry
            _save(entries)
            return f"Updated '{city}' in eco_log ({len(entries)} cities total)."
    entries.append(entry)
    _save(entries)
    return f"Added '{city}' to eco_log ({len(entries)} cities total)."


@mcp.tool()
def list_log() -> str:
    """Return the entire eco_log.json as a JSON string."""
    return json.dumps(_load(), indent=2)


@mcp.tool()
def get_city_report(city: str) -> str:
    """Return one city's stored report, or ERROR if not found."""
    for e in _load():
        if e.get("city", "").lower() == city.lower():
            return json.dumps(e)
    return f"ERROR: '{city}' not in eco_log"


@mcp.tool()
def remove_city(city: str) -> str:
    """Remove one city from eco_log.json (case-insensitive)."""
    entries = _load()
    before = len(entries)
    entries = [e for e in entries if e.get("city", "").lower() != city.lower()]
    _save(entries)
    if len(entries) == before:
        return f"'{city}' was not in eco_log."
    return f"Removed '{city}' ({len(entries)} remain)."


# ---------------------------------------------------------------------------
# 7b. Analytics tools — pure-Python derivations on top of eco_log.json.
# These give the agent / Talk-to-Eco planner real "analysis" affordances
# (ranking, breaches, time-of-day, forecast windows, news context) without
# any aggregation having to happen LLM-side.
# ---------------------------------------------------------------------------

# WHO 24-hour pollutant guidelines (µg/m³ for all except CO which is in
# µg/m³ on Open-Meteo too — 4000 µg/m³ ≈ 4 mg/m³, the WHO 24h CO limit).
_WHO_24H = {"pm25": 15, "pm10": 45, "no2": 25,
            "o3": 100, "so2": 40, "co": 4000}


def _find(city: str) -> dict | None:
    cl = (city or "").strip().lower()
    for e in _load():
        if e.get("city", "").lower() == cl:
            return e
    return None


@mcp.tool()
def who_breach_summary(city: str) -> str:
    """Per-pollutant ratio over the WHO 24-hour guideline for one city,
    plus the single worst breach. Reads eco_log.json — no internet call.

    Returns JSON {city, worst:{pollutant, value, limit, ratio, status},
                  all:[{pollutant, value, limit, ratio, status}, ...]}."""
    e = _find(city)
    if not e:
        return f"ERROR: '{city}' not in eco_log"
    rows = []
    for k, limit in _WHO_24H.items():
        v = e.get(k, 0) or 0
        ratio = round(v / limit, 2) if limit else 0
        status = "OK" if ratio <= 1 else f"{ratio:.1f}× over"
        rows.append({
            "pollutant": k.upper(),
            "value": round(v, 1), "limit": limit,
            "ratio": ratio, "status": status,
        })
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return json.dumps({
        "city": e.get("city"),
        "worst": rows[0] if rows else None,
        "all": rows,
    })


@mcp.tool()
def time_of_day_profile(city: str) -> str:
    """Bucket the city's last 24h hourly AQI into morning (06-11),
    afternoon (12-17), evening (18-23), night (00-05). Returns JSON with
    each bucket's avg/peak AQI plus the SAFEST 3-hour window today.
    Useful for 'when can I run outdoors?' type questions. No internet call."""
    e = _find(city)
    if not e:
        return f"ERROR: '{city}' not in eco_log"
    hourly = e.get("hourly_aqi") or []
    if len(hourly) < 24:
        return f"ERROR: hourly_aqi for '{city}' has only {len(hourly)} of 24 entries"

    def stats(hours: list[int]) -> dict:
        vals = [hourly[h] for h in hours if 0 <= h < len(hourly)]
        return {
            "hours": hours,
            "avg": round(sum(vals) / len(vals), 1) if vals else 0,
            "peak": round(max(vals), 1) if vals else 0,
        }

    buckets = {
        "night":     stats(list(range(0, 6))),
        "morning":   stats(list(range(6, 12))),
        "afternoon": stats(list(range(12, 18))),
        "evening":   stats(list(range(18, 24))),
    }
    # Safest 3-hour window — sliding mean.
    best_start, best_avg = 0, 1e9
    for start in range(0, len(hourly) - 2):
        window = hourly[start:start + 3]
        m = sum(window) / 3
        if m < best_avg:
            best_avg, best_start = m, start
    safest = {
        "start_hour": best_start,
        "end_hour": best_start + 2,
        "label": f"{best_start:02d}:00–{best_start + 3:02d}:00",
        "avg_aqi": round(best_avg, 1),
        "band": _band(best_avg)[0],
    }
    return json.dumps({
        "city": e.get("city"),
        "buckets": buckets,
        "safest_window_3h": safest,
    })


@mcp.tool()
def fetch_aqi_forecast(city: str, hours: int = 24) -> str:
    """Fetch the NEXT N hours (default 24, max 48) of forecast US AQI for a
    city from Open-Meteo. Returns JSON {city, forecast:[{hour_offset,
    iso_time, aqi}], avg, peak, min}. Internet call."""
    try:
        g = _geocode(city)
    except Exception as ex:
        return f"ERROR: {ex}"
    n = max(1, min(int(hours or 24), 48))

    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": g["lat"], "longitude": g["lon"],
            "hourly": "us_aqi",
            "timezone": g["timezone"],
            "past_days": 0, "forecast_days": 2,
        },
        timeout=12,
    )
    r.raise_for_status()
    h = (r.json() or {}).get("hourly", {}) or {}
    times = h.get("time") or []
    aqis = h.get("us_aqi") or []
    now_iso = datetime.utcnow().isoformat()
    # Keep only timestamps strictly in the future, take first N.
    future = [(t, a) for t, a in zip(times, aqis)
              if t > now_iso and a is not None][:n]
    if not future:
        return f"ERROR: no forecast hours for {city}"
    forecast = [
        {"hour_offset": i, "iso_time": t, "aqi": round(a, 1)}
        for i, (t, a) in enumerate(future)
    ]
    vals = [row["aqi"] for row in forecast]
    return json.dumps({
        "city": g["name"],
        "forecast": forecast,
        "avg": round(sum(vals) / len(vals), 1),
        "peak": round(max(vals), 1),
        "min": round(min(vals), 1),
    })


@mcp.tool()
def recommend_outdoor_window(city: str, duration_hours: int = 2) -> str:
    """Find the lowest-AQI window of `duration_hours` (default 2, max 6) in
    the next 24 forecast hours. Returns JSON {city, best:{start_iso,
    end_iso, hour_offset, avg_aqi, band, advisory}, alternatives:[...]}.
    Calls fetch_aqi_forecast under the hood."""
    d = max(1, min(int(duration_hours or 2), 6))
    fc_s = fetch_aqi_forecast(city, hours=24)
    if isinstance(fc_s, str) and fc_s.startswith("ERROR:"):
        return fc_s
    fc = json.loads(fc_s)
    series = fc.get("forecast") or []
    if len(series) < d:
        return f"ERROR: only {len(series)} forecast hours, need {d}"

    windows = []
    for i in range(len(series) - d + 1):
        chunk = series[i:i + d]
        avg = sum(c["aqi"] for c in chunk) / d
        band, _, advisory = _band(avg)
        windows.append({
            "start_iso": chunk[0]["iso_time"],
            "end_iso": chunk[-1]["iso_time"],
            "hour_offset": chunk[0]["hour_offset"],
            "avg_aqi": round(avg, 1),
            "band": band,
            "advisory": advisory,
        })
    windows.sort(key=lambda w: w["avg_aqi"])
    return json.dumps({
        "city": fc.get("city"),
        "duration_hours": d,
        "best": windows[0],
        "alternatives": windows[1:4],
    })


@mcp.tool()
def fetch_news_for(city: str, query: str = "air pollution",
                   limit: int = 5) -> str:
    """Pull recent news headlines mentioning the city + query from Google
    News' free RSS feed (no API key). Returns JSON {city, query, items:[
    {title, source, link, published}]}. Adds a second internet domain
    (news, not weather) so the dashboard feels current."""
    n = max(1, min(int(limit or 5), 10))
    q = f"{city} {query}".strip()
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (eco-tracker)"},
        )
        r.raise_for_status()
    except Exception as ex:
        return f"ERROR: {ex}"

    # Tiny RSS parser — avoids a feedparser dependency.
    import re
    xml = r.text or ""
    items_xml = re.findall(r"<item>(.*?)</item>", xml, flags=re.DOTALL)[:n]

    def grab(tag: str, blob: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", blob, flags=re.DOTALL)
        if not m:
            return ""
        s = m.group(1).strip()
        # CDATA + entity unescape.
        s = re.sub(r"^<!\[CDATA\[|\]\]>$", "", s)
        return (s.replace("&amp;", "&").replace("&lt;", "<")
                 .replace("&gt;", ">").replace("&quot;", '"')
                 .replace("&#39;", "'"))

    items = []
    for blob in items_xml:
        items.append({
            "title":     grab("title", blob),
            "source":    grab("source", blob),
            "link":      grab("link", blob),
            "published": grab("pubDate", blob),
        })
    return json.dumps({"city": city, "query": q, "items": items})


# ---------------------------------------------------------------------------
# 7c. Generic webpage fetcher — gives the agent access to any static URL
# (Wikipedia, gov bulletins, blog posts, news articles). Strips HTML and
# returns clean text the LLM can summarise. NOT for JS-rendered SPAs like
# aqi.in or weather.com — those need a headless browser; use AQICN instead.
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """Fetch a static webpage and return its visible text with HTML stripped.
    Good for Wikipedia, gov bulletins, news articles, blogs. Returns JSON
    {url, title, text, truncated, summary}. Will NOT work on JS-rendered
    single-page apps (aqi.in, accuweather.com) — those return an empty shell.

    Special-cases en.wikipedia.org by hitting the REST summary endpoint,
    which returns just the lead paragraph (no nav chrome, no infobox)."""
    import re
    from urllib.parse import urlparse, unquote

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    # ----- Wikipedia fast-path: use the REST summary API ------------------
    # Returns {extract: "<lead paragraph>", description, ...} — already
    # boilerplate-free, so the dashboard shows the actual gist of the page
    # instead of "Jump to content Main menu Main menu …".
    if host.endswith("wikipedia.org") and "/wiki/" in parsed.path:
        slug = unquote(parsed.path.split("/wiki/", 1)[1].split("#", 1)[0])
        lang = host.split(".", 1)[0] if host.count(".") >= 2 else "en"
        api = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            wr = requests.get(api, timeout=10,
                              headers={"User-Agent": "eco-tracker/1.0"})
            wr.raise_for_status()
            wj = wr.json() or {}
            extract = (wj.get("extract") or "").strip()
            if extract:
                truncated = len(extract) > max_chars
                if truncated:
                    extract = extract[:max_chars] + "…"
                return json.dumps({
                    "url": url,
                    "title": wj.get("title") or "",
                    "description": wj.get("description") or "",
                    "text": extract,
                    "char_count": len(extract),
                    "truncated": truncated,
                    "source": "wikipedia-rest-summary",
                })
        except Exception:
            # Fall through to the generic HTML scraper below.
            pass

    # ----- Generic HTML scraper (with boilerplate trimming) ---------------
    try:
        r = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (eco-tracker)"},
        )
        r.raise_for_status()
    except Exception as ex:
        return f"ERROR: {ex}"

    html = r.text or ""
    # Title.
    m = re.search(r"<title[^>]*>(.*?)</title>", html,
                  flags=re.DOTALL | re.IGNORECASE)
    title = (m.group(1).strip() if m else "").replace("\n", " ")

    # Strip <script>, <style>, and common boilerplate containers BEFORE
    # tag stripping so their text doesn't leak into the output. Order
    # matters: outer-most chrome first.
    for tag in ("script", "style", "nav", "header", "footer", "aside",
                "form", "noscript", "svg"):
        html = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", " ", html,
                      flags=re.DOTALL | re.IGNORECASE)

    # Wikipedia / MediaWiki: drop sidebar, edit links, references list,
    # navbox, infobox, table of contents — keep only article prose.
    for cls in ("mw-jump-link", "mw-editsection", "navbox", "vector-menu",
                "vector-header", "vector-page-toolbar", "mw-portlet",
                "noprint", "reference", "reflist", "catlinks",
                "mw-references-wrap", "toc", "infobox", "sidebar",
                "thumbcaption", "shortdescription"):
        html = re.sub(
            rf'<(\w+)[^>]*class="[^"]*\b{cls}\b[^"]*"[^>]*>.*?</\1>',
            " ", html, flags=re.DOTALL | re.IGNORECASE)

    # If a <main> or <article> exists, keep ONLY that — most modern sites
    # mark the real article body this way.
    main_match = re.search(r"<(main|article)\b[^>]*>(.*?)</\1>", html,
                           flags=re.DOTALL | re.IGNORECASE)
    if main_match:
        html = main_match.group(2)

    # Strip all remaining tags.
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities.
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'"))
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Final boilerplate trim: many pages still leak a "Jump to … Search …
    # Donate Create account Log in" prefix. Drop everything before the first
    # sentence-like chunk (capital letter starting a long run of words).
    skip_match = re.search(r"[A-Z][a-z]{2,}[^.!?]{40,}[.!?]", text)
    if skip_match and skip_match.start() < 600:
        text = text[skip_match.start():]

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "…"

    return json.dumps({
        "url": url,
        "title": title,
        "text": text,
        "char_count": len(text),
        "truncated": truncated,
        "source": "html-scrape",
    })


# ---------------------------------------------------------------------------
# 7d. AQICN — official station-level AQI for any Indian city. This is the
# same CPCB data that aqi.in scrapes, but via a documented free API.
# Sign up at https://aqicn.org/data-platform/token/  →  put the token in
# .env as AQICN_TOKEN=...   If no token is set, the tool returns a clear
# ERROR explaining how to get one (no silent failures).
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_aqicn(city: str) -> str:
    """Fetch CPCB station-level AQI for a city via the AQICN.org free API
    (the same data source aqi.in shows). Returns JSON {city, station, aqi,
    dominant, pm25, pm10, no2, o3, so2, co, time, url}.

    Requires AQICN_TOKEN in .env (free, get one at
    https://aqicn.org/data-platform/token/). Use this when the user asks
    for 'CPCB', 'aqi.in', 'station data', or wants a second opinion vs
    Open-Meteo's interpolated grid."""
    token = os.environ.get("AQICN_TOKEN", "").strip()
    if not token:
        return ("ERROR: AQICN_TOKEN not set. Get a free token at "
                "https://aqicn.org/data-platform/token/ and add "
                "AQICN_TOKEN=<token> to assignment_eco/.env")

    query = _canonical(city)
    try:
        r = requests.get(
            f"https://api.waqi.info/feed/{query}/",
            params={"token": token},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as ex:
        return f"ERROR: {ex}"

    j = r.json() or {}
    if j.get("status") != "ok":
        return f"ERROR: AQICN returned status={j.get('status')} data={j.get('data')}"

    d = j.get("data") or {}
    iaqi = d.get("iaqi") or {}

    def _v(key: str) -> float:
        return round((iaqi.get(key) or {}).get("v", 0) or 0, 1)

    out = {
        "city": query,
        "station": (d.get("city") or {}).get("name", ""),
        "url": (d.get("city") or {}).get("url", ""),
        "aqi": d.get("aqi") or 0,
        "dominant": d.get("dominentpol", ""),
        "pm25": _v("pm25"),
        "pm10": _v("pm10"),
        "no2":  _v("no2"),
        "o3":   _v("o3"),
        "so2":  _v("so2"),
        "co":   _v("co"),
        "temp_c":   _v("t"),
        "humidity": _v("h"),
        "time": (d.get("time") or {}).get("s", ""),
    }
    return json.dumps(out)


# ---------------------------------------------------------------------------
# 7e. NASA FIRMS — active-fire detections from MODIS/VIIRS satellites.
# Free public API; needs a MAP_KEY (instant signup at
# https://firms.modaps.eosdis.nasa.gov/api/area/). Crucial for India: a
# huge fraction of seasonal PM2.5 spikes (Oct-Nov stubble burning, Apr-May
# forest fires) come from biomass burning that satellites catch as fire
# pixels. Combined with wind direction (already in eco_log.json), this
# turns "AQI is bad" into "47 fires upwind in Punjab — that's why".
# ---------------------------------------------------------------------------

import math


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees from north, clockwise) from point1 → point2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


_COMPASS_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass(deg: float) -> str:
    """Compass label for a bearing in degrees (16-point rose)."""
    idx = int((deg + 11.25) % 360 // 22.5)
    return _COMPASS_16[idx]


@mcp.tool()
def fire_hotspots_near(city: str, radius_km: int = 200,
                       days: int = 1) -> str:
    """Count NASA FIRMS active-fire detections within `radius_km` of `city`
    over the last `days` (1-10). Combines fire location with the city's
    current wind direction (from eco_log.json) to label each hotspot
    'upwind' (likely smoke source) or 'downwind' (smoke blowing away).

    Returns JSON {city, radius_km, days, count, upwind_count,
    downwind_count, nearest: {distance_km, bearing_deg, compass, brightness,
    acq_date, upwind: bool}, top: [...up to 5...], wind: {speed_kmh, from_deg,
    from_compass}}.

    Requires FIRMS_MAP_KEY in .env (free, get one at
    https://firms.modaps.eosdis.nasa.gov/api/area/). Use this when the user
    asks 'why is AQI high', 'fires near X', 'smoke source', 'pollution
    source', or 'is the smoke coming from somewhere'."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        return ("ERROR: FIRMS_MAP_KEY not set. Get a free MAP_KEY at "
                "https://firms.modaps.eosdis.nasa.gov/api/area/ and add "
                "FIRMS_MAP_KEY=<key> to assignment_eco/.env")

    # Pull lat/lon from eco_log.json first (avoids a second geocoding call).
    entry = next((e for e in _load()
                  if e.get("city", "").lower() == _canonical(city).lower()),
                 None)
    if entry and entry.get("lat") is not None:
        lat, lon = float(entry["lat"]), float(entry["lon"])
        wind_speed = float(entry.get("wind_speed_kmh") or 0)
        wind_from = float(entry.get("wind_dir_deg") or 0)
        canonical_city = entry["city"]
    else:
        try:
            g = _geocode(city)
        except Exception as ex:
            return f"ERROR: {ex}"
        lat, lon, canonical_city = g["lat"], g["lon"], g["name"]
        wind_speed, wind_from = 0.0, 0.0

    days = max(1, min(int(days or 1), 10))
    radius_km = max(10, min(int(radius_km or 200), 1000))

    # FIRMS area-CSV endpoint. Use VIIRS_SNPP_NRT (best resolution, near-
    # real-time, ~12h latency). Bounding box is west,south,east,north.
    # 1° latitude ≈ 111 km; 1° longitude ≈ 111·cos(lat) km.
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    bbox = f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{key}/VIIRS_SNPP_NRT/{bbox}/{days}")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as ex:
        return f"ERROR: FIRMS request failed: {ex}"

    text = (r.text or "").strip()
    # FIRMS returns CSV with a header row; if no fires, just the header.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "latitude" not in lines[0].lower():
        return json.dumps({
            "city": canonical_city, "radius_km": radius_km, "days": days,
            "count": 0, "upwind_count": 0, "downwind_count": 0,
            "nearest": None, "top": [],
            "wind": {"speed_kmh": wind_speed, "from_deg": wind_from,
                     "from_compass": _compass(wind_from)},
            "note": "no FIRMS rows returned (or unexpected response)",
        })

    header = [h.strip().lower() for h in lines[0].split(",")]
    try:
        i_lat = header.index("latitude")
        i_lon = header.index("longitude")
        i_bri = header.index("bright_ti4") if "bright_ti4" in header \
                else header.index("brightness")
        i_date = header.index("acq_date")
        i_time = header.index("acq_time") if "acq_time" in header else None
        i_conf = header.index("confidence") if "confidence" in header else None
    except ValueError as ex:
        return f"ERROR: unexpected FIRMS header: {ex} ({header})"

    fires: list[dict] = []
    # Wind comes FROM `wind_from` (meteorological convention). A fire is
    # "upwind" of the city if it sits in the direction the wind is coming
    # FROM, i.e. its bearing-from-city is within ±45° of wind_from.
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < len(header):
            continue
        try:
            flat = float(parts[i_lat])
            flon = float(parts[i_lon])
        except ValueError:
            continue
        dist = _haversine_km(lat, lon, flat, flon)
        if dist > radius_km:
            continue  # bbox is square; trim to circle
        bearing = _bearing_deg(lat, lon, flat, flon)
        delta = abs(((bearing - wind_from) + 180) % 360 - 180)
        upwind = wind_speed > 1 and delta <= 45
        fires.append({
            "lat": round(flat, 3), "lon": round(flon, 3),
            "distance_km": round(dist, 1),
            "bearing_deg": round(bearing, 0),
            "compass": _compass(bearing),
            "brightness": float(parts[i_bri]) if parts[i_bri] else 0,
            "acq_date": parts[i_date],
            "acq_time": parts[i_time] if i_time is not None else "",
            "confidence": parts[i_conf] if i_conf is not None else "",
            "upwind": upwind,
        })

    fires.sort(key=lambda f: f["distance_km"])
    upwind_count = sum(1 for f in fires if f["upwind"])

    return json.dumps({
        "city": canonical_city,
        "radius_km": radius_km,
        "days": days,
        "count": len(fires),
        "upwind_count": upwind_count,
        "downwind_count": len(fires) - upwind_count,
        "nearest": fires[0] if fires else None,
        "top": fires[:5],
        "wind": {"speed_kmh": wind_speed, "from_deg": wind_from,
                 "from_compass": _compass(wind_from)},
    })


# ---------------------------------------------------------------------------
# 8. Prefab dashboard
# ---------------------------------------------------------------------------

def _stat(label: str, value: str, sub: str = "") -> None:
    with Column(gap=1):
        Muted(label)
        H1(value)
        if sub:
            Muted(sub)


def _aqi_variant(aqi: float) -> str:
    """Map AQI to a Prefab badge variant for color coding."""
    if aqi < 40:
        return "success"
    if aqi < 60:
        return "default"
    if aqi < 80:
        return "warning"
    return "destructive"


def _table_row(cells: list[str]) -> None:
    with Row(gap=4):
        for c in cells:
            Text(str(c))


def _city_overview(e: dict) -> None:
    """Render the 'Right Now' tab content for one city."""
    aqi = e.get("aqi", 0)
    band = e.get("band", "?")
    with Column(gap=5):
        with Row(gap=8):
            with Column(gap=2):
                H3(f"📍 {e.get('city', '')}")
                Muted(f"Updated {e.get('checked_at', '')}")
                # The headline ring. Prefab Ring expects 0-100 so we
                # cap-display at 100 but show the real number below.
                Ring(value=int(min(aqi, 100)), label=f"AQI {aqi:.0f}")
                Badge(band, variant=_aqi_variant(aqi))
            with Column(gap=2):
                _stat("Temperature", f"{e.get('temp_c', 0)}°C",
                      f"high {e.get('tmax_c', 0)}° / low {e.get('tmin_c', 0)}°")
                _stat("UV Index", f"{e.get('uv_index', 0)}",
                      "max today (0-11+)")
                _stat("Humidity", f"{e.get('humidity', 0)}%",
                      e.get("condition", ""))
            with Column(gap=2):
                _stat("Dominant pollutant", e.get("dominant", "?").upper(),
                      "highest vs WHO guideline")
                _stat("24h average AQI", f"{e.get('avg_aqi', 0):.0f}",
                      f"peak {e.get('peak_aqi', 0):.0f} @ {e.get('peak_hour', 0):02d}:00")
        Separator()
        H3("Health advisory")
        Text(e.get("advisory", ""))


def _all_cities_overview(entries: list[dict]) -> None:
    """Right Now tab — one block per city, stacked top-to-bottom."""
    if not entries:
        Muted("No cities saved yet — call build_city_report first.")
        return
    with Column(gap=8):
        # Network-level summary at the top.
        n = len(entries)
        avg = round(sum(e.get("aqi", 0) for e in entries) / n, 1)
        cleanest = min(entries, key=lambda x: x.get("aqi", 1e9))
        worst = max(entries, key=lambda x: x.get("aqi", -1))
        with Row(gap=8):
            _stat("Cities tracked", str(n), "in eco_log.json")
            _stat("Network avg AQI", f"{avg:.0f}", "across all cities")
            _stat("Cleanest", cleanest.get("city", ""),
                  f"AQI {cleanest.get('aqi', 0):.0f} · {cleanest.get('band', '')}")
            _stat("Most polluted", worst.get("city", ""),
                  f"AQI {worst.get('aqi', 0):.0f} · {worst.get('band', '')}")
        Separator()
        # Per-city block.
        for e in entries:
            _city_overview(e)
            Separator()


def _all_cities_pollutants(entries: list[dict]) -> None:
    """Pollutants tab — one grouped bar chart per city."""
    if not entries:
        Muted("No cities saved yet.")
        return
    who = {"PM2.5": 15, "PM10": 45, "NO₂": 25, "O₃": 100, "SO₂": 40, "CO": 4000}

    with Column(gap=8):
        H3("All-city pollutant comparison (current values, µg/m³)")
        # One row per pollutant, one series per city — the LLM-friendly shape.
        keys = [("PM2.5", "pm25"), ("PM10", "pm10"), ("NO₂", "no2"),
                ("O₃", "o3"), ("SO₂", "so2"), ("CO", "co")]
        bar_rows = []
        for label, k in keys:
            row = {"pollutant": label}
            for e in entries:
                row[e.get("city", "?")] = e.get(k, 0)
            bar_rows.append(row)
        BarChart(
            data=bar_rows,
            series=[ChartSeries(data_key=e.get("city", "?"),
                                label=e.get("city", "?"))
                    for e in entries],
            x_axis="pollutant",
            show_legend=True,
        )
        Separator()
        H3("Per-city detail vs WHO 24-hour limits")
        for e in entries:
            H3(f"📍 {e.get('city', '')}")
            cur = {
                "PM2.5": e.get("pm25", 0), "PM10": e.get("pm10", 0),
                "NO₂": e.get("no2", 0), "O₃": e.get("o3", 0),
                "SO₂": e.get("so2", 0), "CO": e.get("co", 0),
            }
            _table_row(["Pollutant", "Current", "WHO 24h limit", "Status"])
            for k in cur:
                ratio = (cur[k] or 0) / who[k] if who[k] else 0
                status = "OK" if ratio <= 1 else f"{ratio:.1f}× over"
                _table_row([k, f"{cur[k]}", f"{who[k]}", status])
            Separator()


def _all_cities_trend(entries: list[dict]) -> None:
    """24h Trend tab — one combined multi-series line chart + per-city
    sparkline + peak/min/avg stats stacked underneath."""
    if not entries:
        Muted("No cities saved yet.")
        return
    with Column(gap=8):
        H3("AQI across the last 24 hours — all cities")
        max_len = max((len(e.get("hourly_aqi") or []) for e in entries), default=0)
        line_rows = []
        for i in range(max_len):
            row = {"hour": f"{i:02d}h"}
            for e in entries:
                arr = e.get("hourly_aqi") or []
                if i < len(arr):
                    row[e.get("city", "?")] = arr[i]
            line_rows.append(row)
        if line_rows:
            LineChart(
                data=line_rows,
                series=[ChartSeries(data_key=e.get("city", "?"),
                                    label=e.get("city", "?"))
                        for e in entries],
                x_axis="hour", show_legend=True,
            )
        Separator()
        # Per-city PM2.5 sparkline + stats.
        for e in entries:
            H3(f"📍 {e.get('city', '')}")
            with Row(gap=6):
                _stat("Peak AQI", f"{e.get('peak_aqi', 0):.0f}",
                      f"at {e.get('peak_hour', 0):02d}:00")
                hourly = e.get("hourly_aqi") or []
                _stat("Lowest AQI", f"{min(hourly):.0f}" if hourly else "-",
                      "calmest hour")
                _stat("24h average", f"{e.get('avg_aqi', 0):.0f}", "AQI")
            pm25 = e.get("hourly_pm25") or []
            if pm25:
                Muted("PM2.5 trend (24h sparkline)")
                Sparkline(data=pm25)
            Separator()


def _all_cities_hour_pattern(entries: list[dict]) -> None:
    """Hour Pattern tab — one 24-cell heatstrip per city, stacked."""
    if not entries:
        Muted("No cities saved yet.")
        return
    with Column(gap=6):
        H3("Hour-of-day pollution pattern — all cities")
        Muted("Each cell is one hour. Color = AQI band "
              "(green Good · grey Moderate · amber Poor · red Very Poor).")
        for e in entries:
            hourly_aqi = e.get("hourly_aqi") or []
            with Column(gap=2):
                with Row(gap=4):
                    Text(f"📍 {e.get('city', '')}")
                    Muted(f"avg {e.get('avg_aqi', 0):.0f} · "
                          f"peak {e.get('peak_aqi', 0):.0f} @ "
                          f"{e.get('peak_hour', 0):02d}:00")
                with Row(gap=1):
                    for i, aqi in enumerate(hourly_aqi):
                        Badge(f"{i:02d}", variant=_aqi_variant(aqi))
        Separator()
        # Distribution as a stacked-style table: hours per band, per city.
        H3("Hours spent in each band, per city")
        _table_row(["City", "Good", "Fair", "Moderate",
                    "Poor", "Very Poor", "Extremely Poor"])
        order = ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"]
        for e in entries:
            counts = {k: 0 for k in order}
            for aqi in (e.get("hourly_aqi") or []):
                name, _, _ = _band(aqi)
                counts[name] = counts.get(name, 0) + 1
            _table_row([e.get("city", "")] + [str(counts[k]) for k in order])


def _city_pollutants(e: dict) -> None:
    """Tab 2 — pollutant bar chart + table vs WHO limits."""
    who = {"PM2.5": 15, "PM10": 45, "NO₂": 25, "O₃": 100, "SO₂": 40, "CO": 4000}
    cur = {
        "PM2.5": e.get("pm25", 0), "PM10": e.get("pm10", 0),
        "NO₂": e.get("no2", 0), "O₃": e.get("o3", 0),
        "SO₂": e.get("so2", 0), "CO": e.get("co", 0),
    }
    bar_data = [
        {"pollutant": k, "current": cur[k], "who_24h_limit": who[k]}
        for k in cur
    ]
    with Column(gap=5):
        H3(f"Pollutants in {e.get('city', '')} (µg/m³, CO in µg/m³)")
        BarChart(
            data=bar_data,
            series=[
                ChartSeries(data_key="current", label="Current"),
                ChartSeries(data_key="who_24h_limit", label="WHO 24h limit"),
            ],
            x_axis="pollutant",
            show_legend=True,
        )
        Separator()
        H3("Detail")
        _table_row(["Pollutant", "Current", "WHO 24h limit", "Status"])
        for k in cur:
            ratio = (cur[k] or 0) / who[k] if who[k] else 0
            status = "OK" if ratio <= 1 else f"{ratio:.1f}× over"
            _table_row([k, f"{cur[k]}", f"{who[k]}", status])


def _city_trend(e: dict) -> None:
    """Tab 3 — 24h hourly AQI line chart + sparkline + peak/min stats."""
    hourly_aqi = e.get("hourly_aqi", []) or []
    hourly_pm25 = e.get("hourly_pm25", []) or []
    line_data = [
        {"hour": f"{i:02d}h", "aqi": v}
        for i, v in enumerate(hourly_aqi)
    ]
    with Column(gap=5):
        with Row(gap=6):
            _stat("Peak AQI", f"{e.get('peak_aqi', 0):.0f}",
                  f"at {e.get('peak_hour', 0):02d}:00")
            _stat("Lowest AQI", f"{min(hourly_aqi):.0f}" if hourly_aqi else "-",
                  "calmest hour")
            _stat("24h average", f"{e.get('avg_aqi', 0):.0f}", "European AQI")
        Separator()
        H3(f"AQI across the last 24 hours — {e.get('city', '')}")
        if line_data:
            LineChart(
                data=line_data,
                series=[ChartSeries(data_key="aqi", label="AQI")],
                x_axis="hour",
                show_legend=False,
            )
        else:
            Muted("(no hourly data)")
        Separator()
        H3("PM2.5 trend (sparkline)")
        if hourly_pm25:
            Sparkline(data=hourly_pm25)
        else:
            Muted("(no PM2.5 data)")


def _city_hour_pattern(e: dict) -> None:
    """Tab 4 — 24-cell colored badge row (the 'viewable heatmap').

    Each cell = one hour-of-day, color-coded by AQI band. Reads at a glance:
    'mornings green, evenings red'.
    """
    hourly_aqi = e.get("hourly_aqi", []) or []
    with Column(gap=5):
        H3(f"Hour-of-day pollution pattern — {e.get('city', '')}")
        Muted("Each cell is one hour. Color = AQI band "
              "(green Good · grey Moderate · amber Poor · red Very Poor).")
        # Render as a single Row of 24 small Badges.
        with Row(gap=1):
            for i, aqi in enumerate(hourly_aqi):
                Badge(f"{i:02d}", variant=_aqi_variant(aqi))
        Separator()
        with Row(gap=8):
            _stat("Worst hour", f"{e.get('peak_hour', 0):02d}:00",
                  f"AQI {e.get('peak_aqi', 0):.0f}")
            _stat("Best hour",
                  f"{hourly_aqi.index(min(hourly_aqi)):02d}:00" if hourly_aqi else "-",
                  f"AQI {min(hourly_aqi):.0f}" if hourly_aqi else "")
            _stat("Range",
                  f"{(max(hourly_aqi) - min(hourly_aqi)):.0f}" if hourly_aqi else "-",
                  "AQI swing across the day")
        Separator()
        # Distribution as a pie.
        bands: dict[str, int] = {}
        for aqi in hourly_aqi:
            name, _, _ = _band(aqi)
            bands[name] = bands.get(name, 0) + 1
        if bands:
            H3("Hours spent in each band")
            PieChart(
                data=[{"name": k, "value": v} for k, v in bands.items()],
                data_key="value", name_key="name", show_legend=True,
            )


def _compare_tab(entries: list[dict]) -> None:
    """Tab 5 — multi-city comparison."""
    with Column(gap=5):
        if not entries:
            Muted("No cities saved yet — call save_city_report first.")
            return
        H3("Current AQI per city")
        bar_data = [
            {"city": e.get("city", ""), "aqi": e.get("aqi", 0)}
            for e in entries
        ]
        bar_data_sorted = sorted(bar_data, key=lambda x: x["aqi"])
        BarChart(
            data=bar_data_sorted,
            series=[ChartSeries(data_key="aqi", label="AQI")],
            x_axis="city", show_legend=False,
        )
        Separator()
        H3("24-hour AQI trend, all cities")
        # Build a multi-series line: one column per city.
        max_len = max((len(e.get("hourly_aqi") or []) for e in entries), default=0)
        line_rows = []
        for i in range(max_len):
            row = {"hour": f"{i:02d}h"}
            for e in entries:
                arr = e.get("hourly_aqi") or []
                if i < len(arr):
                    row[e.get("city", "?")] = arr[i]
            line_rows.append(row)
        if line_rows:
            LineChart(
                data=line_rows,
                series=[ChartSeries(data_key=e.get("city", "?"),
                                    label=e.get("city", "?"))
                        for e in entries],
                x_axis="hour", show_legend=True,
            )
        Separator()
        H3("Comparison table")
        _table_row(["City", "AQI", "Band", "Dominant", "Temp °C",
                    "UV", "Peak hr"])
        # Sort by aqi ascending so cleanest city is first.
        for e in sorted(entries, key=lambda x: x.get("aqi", 0)):
            _table_row([
                e.get("city", ""),
                f"{e.get('aqi', 0):.0f}",
                e.get("band", ""),
                e.get("dominant", "").upper(),
                f"{e.get('temp_c', 0)}",
                f"{e.get('uv_index', 0)}",
                f"{e.get('peak_hour', 0):02d}:00",
            ])
        Separator()
        # Best / worst callout.
        sorted_e = sorted(entries, key=lambda x: x.get("aqi", 0))
        best, worst = sorted_e[0], sorted_e[-1]
        with Row(gap=6):
            with Column(gap=1):
                Muted("Cleanest right now")
                H2(f"🌿 {best.get('city', '')}")
                Muted(f"AQI {best.get('aqi', 0):.0f} · {best.get('band', '')}")
            with Column(gap=1):
                Muted("Most polluted right now")
                H2(f"⚠️  {worst.get('city', '')}")
                Muted(f"AQI {worst.get('aqi', 0):.0f} · {worst.get('band', '')}")


@mcp.tool(app=True)
def show_dashboard(focus_city: str = "") -> PrefabApp:
    """Render the climate / AQI dashboard. EVERY tab now shows EVERY city
    saved in eco_log.json — no per-city focus needed. The `focus_city`
    argument is accepted for backwards-compat but ignored."""
    entries = _load()
    n = len(entries)
    avg_all = round(sum(e.get("aqi", 0) for e in entries) / n, 1) if n else 0

    with PrefabApp(css_class="max-w-6xl mx-auto p-6") as app:
        with Card():
            with CardHeader():
                CardTitle("🌍 AirSense — India Air-Quality Intelligence")
                Muted(f"{n} city(ies) tracked · network avg AQI {avg_all} · "
                      "live data from Open-Meteo (free, no key)")
            with CardContent():
                with Tabs(value="now"):

                    with Tab("Right Now", value="now"):
                        _all_cities_overview(entries)

                    with Tab("Pollutants", value="pollutants"):
                        _all_cities_pollutants(entries)

                    with Tab("24h Trend", value="trend"):
                        _all_cities_trend(entries)

                    with Tab("Hour Pattern", value="pattern"):
                        _all_cities_hour_pattern(entries)

                    with Tab("Compare Cities", value="compare"):
                        _compare_tab(entries)
    return app


if __name__ == "__main__":
    mcp.run()
