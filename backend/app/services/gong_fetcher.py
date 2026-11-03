"""
GONG Live Frame Fetcher
=======================
Polls https://gong2.nso.edu/HA/has/ for real-time H-alpha solar observations.
No API key, no registration — fully public NSO server.

Directory URL pattern:
    https://gong2.nso.edu/HA/has/YYYYMM/YYYYMMDD/

File naming:
    YYYYMMDDHHMMSSXh.jpg   (X = station code: B, C, L, M, T, U)

Station codes:
    Bh = Big Bear Solar Observatory, California
    Ch = Cerro Tololo Inter-American Observatory, Chile
    Lh = Learmonth Solar Observatory, Western Australia
    Mh = Mauna Loa Solar Observatory, Hawaii
    Th = Teide Observatory, Canary Islands
    Uh = Udaipur Solar Observatory, India
"""

import re
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, List

GONG_HAS_BASE = "https://gong2.nso.edu/HA/has"

STATION_CODES = {
    "Bh": {"name": "Big Bear Solar Observatory",              "location": "California, USA",      "station_id": "BBSO"},
    "Ch": {"name": "Cerro Tololo Inter-American Observatory", "location": "La Serena, Chile",     "station_id": "CTIO"},
    "Lh": {"name": "Learmonth Solar Observatory",             "location": "Western Australia",    "station_id": "LEAR"},
    "Mh": {"name": "Mauna Loa Solar Observatory",             "location": "Hawaii, USA",          "station_id": "MLSO"},
    "Th": {"name": "Teide Observatory",                       "location": "Canary Islands, Spain","station_id": "OT"},
    "Uh": {"name": "Udaipur Solar Observatory",               "location": "Rajasthan, India",     "station_id": "USO"},
}

# In-memory caches
# dt_key -> (list_of_all_files, fetched_at)
_dir_cache: Dict[str, Tuple[List[str], float]] = {}
DIR_CACHE_TTL = 30.0  # cache directory listing for 30s

# station_code -> (filename, image_bytes, dt_utc_str, fetched_at)
_frame_cache: Dict[str, Tuple[str, bytes, str, float]] = {}
FRAME_CACHE_TTL = 25.0  # cache image bytes for 25s


def _directory_url(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return f"{GONG_HAS_BASE}/{dt.strftime('%Y%m')}/{dt.strftime('%Y%m%d')}/"


def _list_all_frames_for_date(dt: Optional[datetime] = None) -> List[str]:
    """Fetches and caches the full list of files for a specific UTC date."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    date_key = dt.strftime("%Y%m%d")
    now = time.time()

    if date_key in _dir_cache:
        cached_files, ts = _dir_cache[date_key]
        if now - ts < DIR_CACHE_TTL:
            return cached_files

    url = _directory_url(dt)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SolarFilamentPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode("utf-8", errors="replace")
        files = re.findall(r'href="(\d{14}\w+\.jpg)"', html)
        files.sort()
        _dir_cache[date_key] = (files, now)
        return files
    except Exception as e:
        # If cache exists (even expired), return it on error
        if date_key in _dir_cache:
            return _dir_cache[date_key][0]
        raise RuntimeError(f"GONG directory listing failed ({url}): {e}")


def _list_frames(station: Optional[str] = None, dt: Optional[datetime] = None) -> List[str]:
    """Returns sorted list of JPEG filenames from GONG, optionally filtered by station."""
    all_files = _list_all_frames_for_date(dt)
    if not station or station == "all":
        return all_files
    code = station[:2].lower()
    return [f for f in all_files if f[14:16].lower() == code]


def fetch_latest_frame(station: str = "all", force: bool = False) -> Tuple[str, bytes, str]:
    """
    Fetches the most recent available JPEG frame from GONG live server.
    If the requested station has no frames today, falls back to yesterday or any active station.

    Returns:
        (filename, image_bytes, observation_time_utc_str)
    """
    cache_key = station or "all"
    now = time.time()

    if not force and cache_key in _frame_cache:
        fname, data, obs_time, fetched_at = _frame_cache[cache_key]
        if now - fetched_at < FRAME_CACHE_TTL:
            return fname, data, obs_time

    dt_utc = datetime.now(timezone.utc)
    frames = _list_frames(station, dt_utc)
    date_used = dt_utc

    # Try yesterday for this station if none today
    if not frames:
        yesterday = datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=timezone.utc) - timedelta(days=1)
        frames = _list_frames(station, yesterday)
        if frames:
            date_used = yesterday

    # Fallback to "all" (active station) if this specific station has no frames
    if not frames and station != "all":
        frames = _list_frames("all", dt_utc)
        date_used = dt_utc
        if not frames:
            yesterday = datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=timezone.utc) - timedelta(days=1)
            frames = _list_frames("all", yesterday)
            if frames:
                date_used = yesterday

    if not frames:
        raise RuntimeError("No GONG frames currently available from live server.")

    latest = frames[-1]
    img_url = _directory_url(date_used) + latest

    try:
        req = urllib.request.Request(img_url, headers={"User-Agent": "SolarFilamentPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            image_bytes = r.read()
    except Exception as e:
        # If cache exists, fall back to it
        if cache_key in _frame_cache:
            fname, data, obs_time, _ = _frame_cache[cache_key]
            return fname, data, obs_time
        raise RuntimeError(f"Failed to download live GONG frame ({img_url}): {e}")

    obs_time = _parse_obs_time(latest)
    _frame_cache[cache_key] = (latest, image_bytes, obs_time, now)
    return latest, image_bytes, obs_time


def list_available_stations_today() -> List[dict]:
    """
    Returns which GONG stations have frames today, using cached directory listing.
    """
    results = []
    dt = datetime.now(timezone.utc)
    all_today = _list_all_frames_for_date(dt)

    yesterday = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) - timedelta(days=1)
    all_yesterday = []

    for code, info in STATION_CODES.items():
        st_lower = code.lower()
        st_frames = [f for f in all_today if f[14:16].lower() == st_lower]
        if not st_frames:
            if not all_yesterday:
                try:
                    all_yesterday = _list_all_frames_for_date(yesterday)
                except Exception:
                    all_yesterday = []
            st_frames = [f for f in all_yesterday if f[14:16].lower() == st_lower]

        results.append({
            "code": code,
            **info,
            "frames_today": len(st_frames),
            "latest_frame": _parse_obs_time(st_frames[-1]) if st_frames else None,
            "is_live": len(st_frames) > 0,
        })

    return results


def _parse_obs_time(filename: str) -> str:
    """Parses 'YYYYMMDDHHMMSSXh.jpg' -> 'YYYY-MM-DD HH:MM:SS UTC'"""
    try:
        d = filename[:14]
        if len(d) == 14 and d.isdigit():
            return f"{d[0:4]}-{d[4:6]}-{d[6:8]} {d[8:10]}:{d[10:12]}:{d[12:14]} UTC"
    except Exception:
        pass
    return "LIVE OBSERVATION"
