#!/usr/bin/env python3
"""Headless VOD-only Auto-Pilot runner for GitHub Actions & scheduled cron jobs.

Skips live streams so GitHub Actions minutes are spent only on finished VODs
using the ultra-light audio_first pipeline + 60fps CRF 17 rendering.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import COOKIES_STORE_PATH, DATA_DIR, JOBS_DIR  # noqa: E402
from app.monitor import load_monitor_state, scan_creator  # noqa: E402


def bootstrap_ci_secrets() -> None:
    """Materialize YouTube OAuth & cookies from CI env secrets if provided."""
    oauth_raw = os.getenv("YOUTUBE_OAUTH_JSON", "").strip()
    if oauth_raw:
        parsed = json.loads(oauth_raw)
        oauth_path = DATA_DIR / "youtube_oauth.json"
        oauth_path.parent.mkdir(parents=True, exist_ok=True)
        oauth_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        try:
            oauth_path.chmod(0o600)
        except OSError:
            pass

    cookies_raw = os.getenv("YOUTUBE_COOKIES_TXT", "").strip()
    if cookies_raw:
        COOKIES_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIES_STORE_PATH.write_text(cookies_raw + "\n", encoding="utf-8")
        try:
            COOKIES_STORE_PATH.chmod(0o600)
        except OSError:
            pass


def _is_recent_enough(item: dict[str, Any], max_age_days: int) -> bool:
    if max_age_days <= 0:
        return True
    date_str = str(item.get("date") or "").strip()
    if not date_str:
        return True
    try:
        item_dt = datetime.fromisoformat(date_str[:10]).replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return item_dt >= cutoff
    except Exception:
        return True


def select_vod_candidates(
    items: list[dict[str, Any]],
    min_duration_sec: int = 900,
    max_age_days: int = 7,
) -> list[dict[str, Any]]:
    """Return newest eligible finished VODs (strictly skipping active live streams)."""
    eligible: list[dict[str, Any]] = []
    for it in items:
        if it.get("clip_status") != "pending":
            continue
        if it.get("is_live"):
            continue
        if int(it.get("duration_sec") or 0) < min_duration_sec:
            continue
        if not _is_recent_enough(it, max_age_days):
            continue
        eligible.append(it)
    return eligible


def _already_compiled_recently(days: int = 5) -> bool:
    comp_dir = DATA_DIR / "compilations"
    if not comp_dir.exists():
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for fp in comp_dir.glob("*.json"):
        try:
            c = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if c.get("status") != "completed":
            continue
        created_s = str(c.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(created_s.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
        if dt >= cutoff:
            return True
    return False


def _should_run_weekly_compile(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    # auto: Sunday UTC and not already compiled in the last 5 days
    return datetime.now(timezone.utc).weekday() == 6 and not _already_compiled_recently(days=5)


def _recent_completed_job_ids(days: int = 7, limit: int = 12) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    matched: list[tuple[float, str]] = []
    for fp in JOBS_DIR.glob("*.json"):
        try:
            j = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("status") != "completed" or not (j.get("clips") or []):
            continue
        created_s = str(j.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(created_s.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
        if dt >= cutoff:
            matched.append((dt.timestamp(), j["job_id"]))
    matched.sort(reverse=True)
    return [jid for _, jid in matched[:limit]]


def _write_github_output(has_work: bool) -> None:
    out_file = os.getenv("GITHUB_OUTPUT", "").strip()
    if out_file:
        with open(out_file, "a", encoding="utf-8") as fh:
            fh.write(f"has_work={'true' if has_work else 'false'}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VOD-only RakaZone Auto-Pilot sweep.")
    parser.add_argument("--creator", default="rakazone", help="Creator ID or 'all'.")
    parser.add_argument("--max-jobs", type=int, default=1, help="Max finished VODs to clip per run.")
    parser.add_argument("--min-duration-sec", type=int, default=900, help="Skip videos shorter than N sec.")
    parser.add_argument("--max-age-days", type=int, default=7, help="Only consider VODs from last N days.")
    parser.add_argument("--auto-upload", action="store_true", help="Auto-upload rendered Shorts to YouTube.")
    parser.add_argument("--privacy", default="public", choices=["private", "unlisted", "public"])
    parser.add_argument("--weekly-compile", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--check-only", action="store_true", help="Only scan feeds and report whether VOD work exists.")
    args = parser.parse_args()

    bootstrap_ci_secrets()
    state = load_monitor_state()
    creators = state.get("creators") or []
    if args.creator != "all":
        creators = [c for c in creators if c.get("creator_id") == args.creator]

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for creator in creators:
        cid = creator["creator_id"]
        print(f"Scanning creator feeds for {cid} ({creator.get('name', cid)})...")
        snap = scan_creator(cid)
        updated_c = next((c for c in snap.get("creators") or [] if c["creator_id"] == cid), creator)
        for item in select_vod_candidates(
            updated_c.get("items") or [],
            min_duration_sec=args.min_duration_sec,
            max_age_days=args.max_age_days,
        ):
            candidates.append((updated_c, item))

    weekly_ready = _should_run_weekly_compile(args.weekly_compile) and len(_recent_completed_job_ids()) >= 2
    has_work = bool(candidates) or weekly_ready
    print(f"Eligible finished VODs: {len(candidates)} | Weekly compilation due: {weekly_ready}")

    if args.check_only:
        for _, it in candidates[:5]:
            print(f"  - [{it.get('platform')}] {it.get('date')} | {it.get('duration_sec')}s | {it.get('title')} ({it.get('url')})")
        _write_github_output(has_work)
        return 0

    if not has_work:
        print("Nothing to process right now (all recent finished VODs already clipped).")
        return 0

    # Lazy import heavy pipeline deps only when actual rendering/uploading runs
    from app.main import (
        _compilation_worker,
        _load_comp,
        _load_job,
        _process_job,
        _save_comp,
        _save_job,
    )
    from app.uploader import get_youtube_status, upload_short_to_youtube

    yt_stat = get_youtube_status()
    can_upload = bool(args.auto_upload and yt_stat.get("connected"))
    if args.auto_upload and not can_upload:
        print("Warning: --auto-upload requested, but YouTube OAuth is not connected (check YOUTUBE_OAUTH_JSON).")

    ran_jobs = 0
    for creator, item in candidates[: max(0, args.max_jobs)]:
        job_id = uuid.uuid4().hex[:10]
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_mode": "youtube",
            "youtube_mode": "audio_first",
            "source_value": str(item["url"]).strip(),
            "source_label": str(item.get("title") or item["url"]).strip(),
            "is_uploaded_copy": False,
            "num_clips": max(1, min(10, int(creator.get("default_clips") or 6))),
            "rsi_iterations": 3,
            "layout_preference": "auto",
            "llm_provider": "auto_free",
            "whisper_model": "base",
            "burn_captions": True,
            "subtitle_language": "auto",
            "subtitle_script": "roman",
            "watermark_text": str(creator.get("watermark_text") or "").strip()[:48],
            "channel_watermark": str(
                creator.get("channel_watermark")
                or ("@RAKAGTAVI" if creator.get("creator_id") == "rakazone" else "")
            ).strip()[:48],
            "creator_tag": str(creator.get("creator_tag") or "").strip()[:48],
            "required_hashtags": str(creator.get("required_hashtags") or "").strip()[:140],
            "target_max_duration": max(15, min(60, int(creator.get("target_max_duration") or 45))),
            "campaign_id": str(creator.get("campaign_id") or "").strip(),
            "auto_upload_youtube": can_upload,
            "youtube_privacy": args.privacy,
            "logs": [],
            "clips": [],
            "rsi_traces": [],
        }
        _save_job(job)
        print(f"Processing finished VOD job {job_id}: {job['source_label']} ({job['source_value']})")
        _process_job(job_id)
        done = _load_job(job_id)
        print(f"Job {job_id} finished with status={done.get('status')}, clips={len(done.get('clips') or [])}")
        for line in (done.get("logs") or [])[-12:]:
            print(f"  {line}")
        if done.get("status") != "completed":
            print(done.get("error") or "Job failed")
            return 1
        ran_jobs += 1

    if weekly_ready:
        recent_jids = _recent_completed_job_ids(days=7, limit=12)
        if len(recent_jids) >= 2:
            c0 = creators[0] if creators else {}
            comp_id = f"comp_{uuid.uuid4().hex[:8]}"
            today_label = datetime.now(timezone.utc).strftime("%b %d")
            comp = {
                "compilation_id": comp_id,
                "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "title": f"RakaZone GTA 6 Funniest & Craziest Moments This Week ({today_label})",
                "job_ids": recent_jids,
                "max_clips": 10,
                "watermark_text": str(c0.get("watermark_text") or "Credit: @RakaZoneGaming"),
                "channel_watermark": str(c0.get("channel_watermark") or "@RAKAGTAVI"),
                "creator_tag": str(c0.get("creator_tag") or "@RakaZoneGaming"),
                "required_hashtags": str(c0.get("required_hashtags") or "#RakaZoneGaming #GTA6 #Gaming"),
                "vfx_enabled": True,
                "transition_style": "slide_wipe",
                "intro_audio_path": "",
                "logs": [],
            }
            _save_comp(comp)
            print(f"Building weekly compilation {comp_id} from {len(recent_jids)} completed VOD jobs...")
            _compilation_worker(comp_id)
            done_c = _load_comp(comp_id)
            print(f"Compilation {comp_id} status={done_c.get('status')}")
            if done_c.get("status") == "completed" and can_upload:
                out_p = Path(str(done_c.get("output_path") or ""))
                if out_p.exists():
                    up = upload_short_to_youtube(
                        video_path=out_p,
                        title=done_c["title"],
                        description=done_c.get("youtube_description") or done_c["title"],
                        privacy_status=args.privacy,
                        is_short=False,
                    )
                    done_c["youtube_upload"] = up
                    _save_comp(done_c)
                    print(f"Uploaded weekly compilation -> {up.get('url')}")

    print(f"CI Auto-Pilot completed cleanly (VOD jobs run: {ran_jobs}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
