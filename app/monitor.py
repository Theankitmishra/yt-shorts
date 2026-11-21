from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.config import DATA_DIR, JOBS_DIR
from app.transcriber import run_cmd

MONITOR_PATH = DATA_DIR / "monitor.json"

COPYRIGHT_WATCHPARTY_RE = re.compile(
    r"\b(bigg?\s*boss|wkv|weekend\s*ka\s*vaar|watch\s*party|watch\s*along|watchparty|"
    r"watchalong|hotstar|jiocinema|star\s*plus|colors\s*tv|sony\s*liv|netflix|"
    r"prime\s*video|splitsvilla|roadies|shark\s*tank|kbc|full\s*movie|"
    r"match\s*highlights|ipl\s*live|cricket\s*match|series\s*watch|episode\s*\d+)\b",
    re.IGNORECASE,
)
GAMING_TITLE_RE = re.compile(
    r"\b(gta\s*6|gta\s*5|gta|yatra\s*rp|yatrarp|gaming|gameplay|walkthrough|"
    r"heist|mission|vice\s*city|rp|valorant|bgmi|pubg|minecraft)\b",
    re.IGNORECASE,
)


def is_copyright_risk_title(title: str) -> bool:
    """Return True if a stream/chapter title looks like a copyrighted TV/movie/watchparty broadcast."""
    return bool(COPYRIGHT_WATCHPARTY_RE.search(str(title or "")))


def is_gaming_title(title: str) -> bool:
    """Return True if a stream title clearly indicates gameplay content."""
    return bool(GAMING_TITLE_RE.search(str(title or "")))


def _default_autopilot() -> dict[str, Any]:
    return {
        "enabled": False,
        "interval_minutes": 60,
        "auto_upload_youtube": False,
        "youtube_privacy": "private",
        "max_jobs_per_day": 2,
        "only_finished_vods": True,
        "min_duration_sec": 900,
        "last_run_at": "",
        "last_action": "Idle (Auto-Pilot OFF)",
    }


def _default_monitor_state() -> dict[str, Any]:
    return {
        "autopilot": _default_autopilot(),
        "creators": [
            {
                "creator_id": "rakazone",
                "name": "RakaZone Gaming",
                "youtube_url": "https://www.youtube.com/@RakaZoneGaming",
                "kick_url": "https://kick.com/rakazone",
                "creator_tag": "@RakaZoneGaming",
                "watermark_text": "Credit: @RakaZoneGaming",
                "channel_watermark": "@RAKAGTAVI",
                "required_hashtags": "#RakaZoneGaming #GTA6 #GTA #Gaming #Shorts",
                "default_clips": 6,
                "target_max_duration": 45,
                "campaign_id": "rakazone-gta6",
                "youtube_channel_id": "",
                "last_scanned_at": "",
                "items": [],
                "manual_clipped_urls": [],
            }
        ]
    }


def load_monitor_state() -> dict[str, Any]:
    if MONITOR_PATH.exists():
        try:
            data = json.loads(MONITOR_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("creators"), list):
                changed = False
                for c in data["creators"]:
                    if c.get("creator_id") == "rakazone" and not str(c.get("channel_watermark") or "").strip():
                        c["channel_watermark"] = "@RAKAGTAVI"
                        changed = True
                if changed:
                    save_monitor_state(data)
                return data
        except Exception:
            pass
    default = _default_monitor_state()
    save_monitor_state(default)
    return default


def save_monitor_state(state: dict[str, Any]) -> None:
    MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _extract_yt_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else ""


def _normalize_kick_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def _collect_job_index() -> dict[str, dict[str, Any]]:
    """Map normalized URLs and YouTube IDs -> latest matching job record."""
    idx: dict[str, dict[str, Any]] = {}
    for fp in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            j = json.loads(fp.read_text(encoding="utf-8"))
            sv = str(j.get("source_value") or "").strip()
            if not sv:
                continue
            info = {
                "job_id": j["job_id"],
                "status": j.get("status", ""),
                "num_clips": len(j.get("clips") or []),
                "created_at": j.get("created_at", ""),
            }
            idx[_normalize_kick_url(sv)] = info
            yid = _extract_yt_id(sv)
            if yid:
                idx[f"yt:{yid}"] = info
        except Exception:
            continue
    return idx


def _attach_clip_status(creator: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    job_idx = _collect_job_index()
    manual = {_normalize_kick_url(u) for u in creator.get("manual_clipped_urls") or []}
    out: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        item["copyright_risk"] = is_copyright_risk_title(str(item.get("title") or ""))
        clean_url = _normalize_kick_url(str(item.get("url") or ""))
        yid = str(item.get("video_id") or "")
        match = job_idx.get(clean_url) or (job_idx.get(f"yt:{yid}") if item.get("platform") == "YouTube" and yid else None)
        if match and match["status"] in {"running", "queued"}:
            item["clip_status"] = "running"
            item["job_id"] = match["job_id"]
            item["clipped_count"] = match["num_clips"]
        elif match and match["status"] == "completed" and match["num_clips"] > 0:
            item["clip_status"] = "clipped"
            item["job_id"] = match["job_id"]
            item["clipped_count"] = match["num_clips"]
        elif clean_url in manual:
            item["clip_status"] = "clipped"
            item["job_id"] = ""
            item["clipped_count"] = 0
            item["manual_marked"] = True
        else:
            item["clip_status"] = "pending"
            item["job_id"] = ""
            item["clipped_count"] = 0
        out.append(item)
    return out


def _fetch_yt_rss_dates(channel_id: str) -> dict[str, str]:
    if not channel_id:
        return {}
    try:
        resp = httpx.get(
            f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}",
            timeout=10.0,
        )
        if resp.status_code != 200:
            return {}
        root = ET.fromstring(resp.text)
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
        }
        dates: dict[str, str] = {}
        for entry in root.findall("a:entry", ns):
            vid = entry.findtext("yt:videoId", default="", namespaces=ns).strip()
            pub = entry.findtext("a:published", default="", namespaces=ns).strip()
            if vid and pub:
                dates[vid] = pub[:10]
        return dates
    except Exception:
        return {}


def _scan_youtube_tab(tab_url: str, cached_cid: str = "", limit: int = 8) -> tuple[list[dict[str, Any]], str]:
    out = run_cmd([
        "yt-dlp",
        "--no-warnings",
        "--flat-playlist",
        "--playlist-end",
        str(limit),
        "-J",
        tab_url,
    ])
    data = json.loads(out)
    cid = str(data.get("channel_id") or data.get("uploader_id") or cached_cid or "").strip()
    rss_dates = _fetch_yt_rss_dates(cid)
    items: list[dict[str, Any]] = []
    for e in data.get("entries") or []:
        vid = str(e.get("id") or "").strip()
        if not vid:
            continue
        live_status = str(e.get("live_status") or "")
        is_live = live_status == "is_live" or bool(e.get("is_live"))
        pub_date = rss_dates.get(vid, "")
        if not pub_date and e.get("timestamp"):
            pub_date = datetime.fromtimestamp(float(e["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d")
        if is_live and not pub_date:
            pub_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        items.append({
            "item_id": f"yt_{vid}",
            "platform": "YouTube",
            "video_id": vid,
            "title": str(e.get("title") or "YouTube Stream").strip(),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration_sec": int(e.get("duration") or 0),
            "is_live": is_live,
            "date": pub_date,
            "sort_ts": float(e.get("timestamp") or (datetime.fromisoformat(pub_date).timestamp() if pub_date else 0.0)),
            "view_count": int(e.get("view_count") or 0),
            "kind": "live" if is_live else ("vod" if live_status == "was_live" else "video"),
        })
    return items, cid


def _scan_kick_channel(kick_url: str, limit: int = 8) -> list[dict[str, Any]]:
    base = kick_url.rstrip("/")
    videos_url = base if base.endswith("/videos") else f"{base}/videos"
    items: list[dict[str, Any]] = []

    # Check Kick VOD archive via yt-dlp flat-playlist
    try:
        out = run_cmd([
            "yt-dlp",
            "--no-warnings",
            "--flat-playlist",
            "--playlist-end",
            str(limit),
            "-J",
            videos_url,
        ])
        data = json.loads(out)
        for e in data.get("entries") or []:
            raw_url = _normalize_kick_url(str(e.get("url") or ""))
            vid = str(e.get("id") or raw_url)
            ts = float(e.get("timestamp") or 0.0)
            pub_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts > 0 else ""
            is_live = bool(e.get("is_live"))
            items.append({
                "item_id": f"kick_{vid}",
                "platform": "Kick",
                "video_id": vid,
                "title": str(e.get("title") or "Kick VOD").strip(),
                "url": raw_url or f"{base}/videos/{vid}",
                "duration_sec": int(e.get("duration") or 0),
                "is_live": is_live,
                "date": pub_date,
                "sort_ts": ts,
                "view_count": int(e.get("view_count") or 0),
                "kind": "live" if is_live else "vod",
            })
    except Exception:
        pass

    # Check if Kick channel is currently live
    try:
        live_out = run_cmd([
            "yt-dlp",
            "--no-warnings",
            "--dump-single-json",
            "--skip-download",
            base,
        ])
        live_data = json.loads(live_out)
        if live_data.get("is_live"):
            now_ts = datetime.now(timezone.utc).timestamp()
            items.insert(0, {
                "item_id": f"kick_live_{live_data.get('id') or 'now'}",
                "platform": "Kick",
                "video_id": str(live_data.get("id") or "live"),
                "title": str(live_data.get("title") or "Kick Live Stream").strip(),
                "url": base,
                "duration_sec": 0,
                "is_live": True,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "sort_ts": now_ts + 1000,
                "view_count": int(live_data.get("concurrent_view_count") or 0),
                "kind": "live",
            })
    except Exception:
        pass

    return items


def get_monitor_snapshot() -> dict[str, Any]:
    state = load_monitor_state()
    ap = dict(_default_autopilot())
    ap.update(state.get("autopilot") or {})
    creators_out: list[dict[str, Any]] = []
    for c in state.get("creators") or []:
        cc = dict(c)
        cc["items"] = _attach_clip_status(cc, cc.get("items") or [])
        creators_out.append(cc)
    return {"creators": creators_out, "autopilot": ap}


def get_autopilot_config() -> dict[str, Any]:
    state = load_monitor_state()
    ap = dict(_default_autopilot())
    ap.update(state.get("autopilot") or {})
    return ap


def update_autopilot_config(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_monitor_state()
    ap = dict(_default_autopilot())
    ap.update(state.get("autopilot") or {})
    for k in ("enabled", "auto_upload_youtube", "only_finished_vods"):
        if k in payload:
            ap[k] = bool(payload[k])
    if "interval_minutes" in payload:
        ap["interval_minutes"] = max(10, min(720, int(payload["interval_minutes"])))
    if "max_jobs_per_day" in payload:
        ap["max_jobs_per_day"] = max(1, min(6, int(payload["max_jobs_per_day"])))
    if "min_duration_sec" in payload:
        ap["min_duration_sec"] = max(0, int(payload["min_duration_sec"]))
    if "youtube_privacy" in payload:
        priv = str(payload["youtube_privacy"]).strip().lower()
        if priv in {"private", "unlisted", "public"}:
            ap["youtube_privacy"] = priv
    if "last_run_at" in payload:
        ap["last_run_at"] = str(payload["last_run_at"])
    if "last_action" in payload:
        ap["last_action"] = str(payload["last_action"])[:160]
    state["autopilot"] = ap
    save_monitor_state(state)
    return ap


def count_jobs_started_today() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cnt = 0
    for fp in JOBS_DIR.glob("*.json"):
        try:
            j = json.loads(fp.read_text(encoding="utf-8"))
            if str(j.get("created_at") or "").startswith(today):
                cnt += 1
        except Exception:
            continue
    return cnt


def upsert_creator(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_monitor_state()
    creators = state.get("creators") or []
    cid = str(payload.get("creator_id") or "").strip() or uuid.uuid4().hex[:8]
    existing = next((c for c in creators if c["creator_id"] == cid), None)
    record = {
        "creator_id": cid,
        "name": str(payload.get("name") or "Creator").strip()[:60],
        "youtube_url": str(payload.get("youtube_url") or "").strip(),
        "kick_url": str(payload.get("kick_url") or "").strip(),
        "creator_tag": str(payload.get("creator_tag") or "").strip()[:48],
        "watermark_text": str(payload.get("watermark_text") or "").strip()[:48],
        "channel_watermark": str(
            payload.get("channel_watermark")
            or (existing.get("channel_watermark", "") if existing else ("@RAKAGTAVI" if cid == "rakazone" else ""))
        ).strip()[:48],
        "required_hashtags": str(payload.get("required_hashtags") or "").strip()[:140],
        "default_clips": max(1, min(10, int(payload.get("default_clips") or 6))),
        "target_max_duration": max(15, min(60, int(payload.get("target_max_duration") or 45))),
        "campaign_id": existing.get("campaign_id", "") if existing else str(payload.get("campaign_id") or ""),
        "youtube_channel_id": existing.get("youtube_channel_id", "") if existing else "",
        "last_scanned_at": existing.get("last_scanned_at", "") if existing else "",
        "items": existing.get("items", []) if existing else [],
        "manual_clipped_urls": existing.get("manual_clipped_urls", []) if existing else [],
    }
    creators = [c for c in creators if c["creator_id"] != cid]
    creators.insert(0, record)
    state["creators"] = creators
    save_monitor_state(state)
    return get_monitor_snapshot()


def scan_creator(creator_id: str) -> dict[str, Any]:
    state = load_monitor_state()
    target = next((c for c in state.get("creators") or [] if c["creator_id"] == creator_id), None)
    if not target:
        raise ValueError(f"Creator not found: {creator_id}")

    merged: dict[str, dict[str, Any]] = {}
    yt_url = str(target.get("youtube_url") or "").strip().rstrip("/")
    if yt_url:
        cid = str(target.get("youtube_channel_id") or "")
        for suffix in ("/streams", "/videos"):
            tab = yt_url if yt_url.endswith(("/streams", "/videos")) else f"{yt_url}{suffix}"
            try:
                rows, found_cid = _scan_youtube_tab(tab, cached_cid=cid, limit=8)
                if found_cid:
                    cid = found_cid
                for r in rows:
                    merged.setdefault(r["item_id"], r)
            except Exception:
                continue
        target["youtube_channel_id"] = cid

    kick_url = str(target.get("kick_url") or "").strip()
    if kick_url:
        for r in _scan_kick_channel(kick_url, limit=8):
            merged.setdefault(r["item_id"], r)

    sorted_items = sorted(
        merged.values(),
        key=lambda x: (1 if x.get("is_live") else 0, x.get("date") or "", float(x.get("sort_ts") or 0.0)),
        reverse=True,
    )
    target["items"] = sorted_items
    target["last_scanned_at"] = datetime.now(timezone.utc).isoformat()
    save_monitor_state(state)
    return get_monitor_snapshot()


def toggle_manual_clipped(creator_id: str, url: str, clipped: bool) -> dict[str, Any]:
    state = load_monitor_state()
    target = next((c for c in state.get("creators") or [] if c["creator_id"] == creator_id), None)
    if not target:
        raise ValueError(f"Creator not found: {creator_id}")
    norm = _normalize_kick_url(url)
    cur = [_normalize_kick_url(u) for u in target.get("manual_clipped_urls") or []]
    if clipped and norm not in cur:
        cur.append(norm)
    elif not clipped:
        cur = [u for u in cur if u != norm]
    target["manual_clipped_urls"] = cur
    save_monitor_state(state)
    return get_monitor_snapshot()
