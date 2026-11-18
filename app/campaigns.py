from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import DATA_DIR

CAMPAIGNS_PATH = DATA_DIR / "campaigns.json"


def _default_campaigns() -> list[dict[str, Any]]:
    return [
        {
            "campaign_id": "sample-kick-whop",
            "name": "Sample: Kick / Whop Gaming CPM",
            "platform": "Whop Content Rewards",
            "cpm_usd": 1.5,
            "max_clip_payout_usd": 150.0,
            "vod_url": "",
            "creator_tag": "@CreatorHandle",
            "watermark_text": "kick.com/creator",
            "required_hashtags": "#gaming #kick #clips #shorts",
            "target_max_duration": 55,
            "notes": "Sample template — edit with your real Whop/Kick brief details or add a new campaign.",
            "submissions": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    ]


def load_campaigns() -> list[dict[str, Any]]:
    if CAMPAIGNS_PATH.exists():
        try:
            data = json.loads(CAMPAIGNS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [_with_totals(c) for c in data]
        except Exception:
            pass
    defaults = _default_campaigns()
    save_campaigns(defaults)
    return [_with_totals(c) for c in defaults]


def save_campaigns(campaigns: list[dict[str, Any]]) -> None:
    CAMPAIGNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGNS_PATH.write_text(json.dumps(campaigns, indent=2), encoding="utf-8")


def _calc_sub_payout(views: int, cpm_usd: float, cap_usd: float) -> float:
    raw = (max(0, int(views)) / 1000.0) * max(0.0, float(cpm_usd))
    if cap_usd > 0:
        raw = min(raw, float(cap_usd))
    return round(raw, 2)


def _with_totals(c: dict[str, Any]) -> dict[str, Any]:
    out = dict(c)
    cpm = float(out.get("cpm_usd") or 0.0)
    cap = float(out.get("max_clip_payout_usd") or 0.0)
    subs = []
    tot_views = 0
    tot_usd = 0.0
    for s in out.get("submissions") or []:
        sc = dict(s)
        v = int(sc.get("views") or 0)
        est = _calc_sub_payout(v, cpm, cap)
        sc["est_payout_usd"] = est
        tot_views += v
        tot_usd += est
        subs.append(sc)
    out["submissions"] = subs
    out["total_views"] = tot_views
    out["total_est_usd"] = round(tot_usd, 2)
    return out


def upsert_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    campaigns = load_campaigns()
    cid = str(payload.get("campaign_id") or "").strip() or uuid.uuid4().hex[:8]
    existing = next((c for c in campaigns if c["campaign_id"] == cid), None)
    record = {
        "campaign_id": cid,
        "name": str(payload.get("name") or "Untitled Campaign").strip()[:80],
        "platform": str(payload.get("platform") or "Whop Content Rewards").strip()[:50],
        "cpm_usd": round(max(0.0, float(payload.get("cpm_usd") or 0.0)), 2),
        "max_clip_payout_usd": round(max(0.0, float(payload.get("max_clip_payout_usd") or 0.0)), 2),
        "vod_url": str(payload.get("vod_url") or "").strip(),
        "creator_tag": str(payload.get("creator_tag") or "").strip()[:48],
        "watermark_text": str(payload.get("watermark_text") or "").strip()[:48],
        "required_hashtags": str(payload.get("required_hashtags") or "").strip()[:140],
        "target_max_duration": max(15, min(60, int(payload.get("target_max_duration") or 55))),
        "notes": str(payload.get("notes") or "").strip()[:300],
        "submissions": existing.get("submissions", []) if existing else [],
        "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
    }
    campaigns = [c for c in campaigns if c["campaign_id"] != cid]
    campaigns.insert(0, record)
    save_campaigns(campaigns)
    return _with_totals(record)


def delete_campaign(campaign_id: str) -> list[dict[str, Any]]:
    campaigns = [c for c in load_campaigns() if c["campaign_id"] != campaign_id]
    save_campaigns(campaigns)
    return [_with_totals(c) for c in campaigns]


def add_campaign_submission(
    campaign_id: str,
    platform: str,
    post_url: str,
    clip_label: str = "",
    views: int = 0,
) -> dict[str, Any] | None:
    campaigns = load_campaigns()
    target = next((c for c in campaigns if c["campaign_id"] == campaign_id), None)
    if not target:
        return None
    sub = {
        "sub_id": uuid.uuid4().hex[:8],
        "platform": platform.strip()[:30] or "YouTube Shorts",
        "post_url": post_url.strip(),
        "clip_label": clip_label.strip()[:60] or "Short Clip",
        "views": max(0, int(views)),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    target.setdefault("submissions", []).insert(0, sub)
    save_campaigns(campaigns)
    return _with_totals(target)


def update_submission_views(
    campaign_id: str, sub_id: str, views: int
) -> dict[str, Any] | None:
    campaigns = load_campaigns()
    target = next((c for c in campaigns if c["campaign_id"] == campaign_id), None)
    if not target:
        return None
    for s in target.get("submissions", []):
        if s.get("sub_id") == sub_id:
            s["views"] = max(0, int(views))
            break
    save_campaigns(campaigns)
    return _with_totals(target)


def generate_bounty_post_pack(
    clip: dict[str, Any],
    creator_tag: str = "",
    required_hashtags: str = "",
    watermark_text: str = "",
    burn_captions: bool = True,
) -> dict[str, Any]:
    """Generate ready-to-paste Shorts/Reels/TikTok title, caption, hashtags & brief compliance checks."""
    hook = str(clip.get("hook_title") or "CRAZY STREAM MOMENT").strip()
    dur = round(float(clip.get("end", 0)) - float(clip.get("start", 0)), 1)
    preview = str((clip.get("evaluation") or {}).get("text_preview") or "").strip()
    snippet = preview[:110].rstrip(". ") + ("..." if len(preview) > 110 else "")

    clean_tag = creator_tag.strip()
    if clean_tag and not clean_tag.startswith("@") and "/" not in clean_tag:
        clean_tag = "@" + clean_tag

    tags = required_hashtags.strip() or "#gaming #clips #shorts #viral"
    title_parts = [hook.title()]
    if clean_tag:
        title_parts.append(f"({clean_tag})")
    title = " ".join(title_parts)[:95]

    caption_lines = [hook.title()]
    if snippet:
        caption_lines.append(f'"{snippet}"')
    if clean_tag:
        caption_lines.append(f"Credit: {clean_tag}")
    caption_lines.append(tags)
    caption = "\n".join(caption_lines)

    return {
        "title": title,
        "caption": caption,
        "hashtags": tags,
        "compliance": {
            "vertical_9_16": True,
            "duration_sec": dur,
            "duration_ok": 12.0 <= dur <= 59.0,
            "captions_burned": bool(burn_captions),
            "watermark_burned": bool(watermark_text.strip()),
            "watermark_text": watermark_text.strip(),
        },
    }
