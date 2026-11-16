from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import DATA_DIR

YOUTUBE_OAUTH_PATH = DATA_DIR / "youtube_oauth.json"
YOUTUBE_UPLOAD_SCOPE = (
    "https://www.googleapis.com/auth/youtube.upload "
    "https://www.googleapis.com/auth/youtube.readonly"
)
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_RESUMABLE_INIT_URL = (
    "https://www.googleapis.com/upload/youtube/v3/videos"
    "?uploadType=resumable&part=snippet,status"
)
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true"


def load_youtube_oauth() -> dict[str, Any]:
    if YOUTUBE_OAUTH_PATH.exists():
        try:
            data = json.loads(YOUTUBE_OAUTH_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_youtube_oauth(updates: dict[str, Any], replace: bool = False) -> dict[str, Any]:
    data = {} if replace else load_youtube_oauth()
    data.update(updates)
    YOUTUBE_OAUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    YOUTUBE_OAUTH_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        YOUTUBE_OAUTH_PATH.chmod(0o600)
    except OSError:
        pass
    return data


def clear_youtube_tokens() -> dict[str, Any]:
    cur = load_youtube_oauth()
    kept = {
        k: cur[k]
        for k in ("client_id", "client_secret")
        if cur.get(k)
    }
    return save_youtube_oauth(kept, replace=True)


def save_youtube_client_credentials(
    client_secret_json_text: str = "",
    client_id: str = "",
    client_secret: str = "",
) -> dict[str, Any]:
    """Store Google OAuth Desktop/Web client credentials from pasted JSON or raw fields."""
    cid = client_id.strip()
    csec = client_secret.strip()
    if client_secret_json_text.strip():
        raw = json.loads(client_secret_json_text)
        block = raw.get("web") or raw.get("installed") or raw
        cid = str(block.get("client_id") or cid).strip()
        csec = str(block.get("client_secret") or csec).strip()
    if not cid or not csec:
        raise ValueError("Provide a valid Google OAuth client_id & client_secret (or client_secret.json).")
    return save_youtube_oauth({"client_id": cid, "client_secret": csec})


def get_youtube_status() -> dict[str, Any]:
    d = load_youtube_oauth()
    has_client = bool(d.get("client_id") and d.get("client_secret"))
    connected = bool(has_client and d.get("refresh_token"))
    cid = str(d.get("client_id") or "")
    masked_id = f"{cid[:12]}...{cid[-16:]}" if len(cid) > 30 else ("set" if cid else "")
    return {
        "has_client": has_client,
        "client_id_masked": masked_id,
        "connected": connected,
        "channel_id": d.get("channel_id", ""),
        "channel_title": d.get("channel_title", ""),
        "connected_at": d.get("connected_at", ""),
    }


def build_youtube_auth_url(redirect_uri: str) -> str:
    d = load_youtube_oauth()
    if not d.get("client_id") or not d.get("client_secret"):
        raise ValueError("Save your Google Cloud OAuth Client credentials first.")
    state = secrets.token_urlsafe(24)
    save_youtube_oauth({"oauth_state": state, "last_redirect_uri": redirect_uri})
    params = {
        "client_id": d["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": YOUTUBE_UPLOAD_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_youtube_code(code: str, state: str = "", redirect_uri: str = "") -> dict[str, Any]:
    d = load_youtube_oauth()
    if state and d.get("oauth_state") and state != d["oauth_state"]:
        raise ValueError("Invalid OAuth state; restart Connect YouTube.")
    redir = redirect_uri or d.get("last_redirect_uri") or "http://127.0.0.1:8765/api/youtube/callback"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code.strip(),
                "client_id": d["client_id"],
                "client_secret": d["client_secret"],
                "redirect_uri": redir,
                "grant_type": "authorization_code",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Google token exchange failed ({resp.status_code}): {resp.text[:300]}")
        tok = resp.json()

        access_token = str(tok.get("access_token") or "")
        refresh_token = str(tok.get("refresh_token") or d.get("refresh_token") or "")
        if not refresh_token:
            raise RuntimeError(
                "Google did not return a refresh_token. Revoke previous access or re-consent with prompt=consent."
            )
        expires_in = int(tok.get("expires_in") or 3600)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=max(60, expires_in - 60))).isoformat()

        channel_id = ""
        channel_title = ""
        try:
            ch = client.get(
                YOUTUBE_CHANNELS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if ch.status_code == 200:
                items = ch.json().get("items") or []
                if items:
                    channel_id = str(items[0].get("id") or "")
                    channel_title = str((items[0].get("snippet") or {}).get("title") or "")
        except Exception:
            pass

    save_youtube_oauth(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "channel_id": channel_id,
            "channel_title": channel_title or "Connected YouTube Channel",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "oauth_state": "",
        }
    )
    return get_youtube_status()


def _get_valid_access_token(client: httpx.Client) -> str:
    d = load_youtube_oauth()
    if not d.get("client_id") or not d.get("client_secret") or not d.get("refresh_token"):
        raise RuntimeError("YouTube channel is not connected yet. Connect it in Cloud, Keys & Storage.")

    exp_str = str(d.get("expires_at") or "")
    if d.get("access_token") and exp_str:
        try:
            if datetime.fromisoformat(exp_str) > datetime.now(timezone.utc):
                return str(d["access_token"])
        except Exception:
            pass

    resp = client.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": d["client_id"],
            "client_secret": d["client_secret"],
            "refresh_token": d["refresh_token"],
            "grant_type": "refresh_token",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"YouTube OAuth refresh failed ({resp.status_code}): {resp.text[:300]}")
    tok = resp.json()
    access_token = str(tok["access_token"])
    expires_in = int(tok.get("expires_in") or 3600)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=max(60, expires_in - 60))).isoformat()
    save_youtube_oauth({"access_token": access_token, "expires_at": expires_at})
    return access_token


def _extract_clean_tags(description: str) -> list[str]:
    found = [t.lstrip("#").strip() for t in re.findall(r"#([A-Za-z0-9_]{2,30})", description)]
    if "shorts" not in [t.lower() for t in found]:
        found.append("shorts")
    return found[:15]


def upload_short_to_youtube(
    video_path: Path,
    title: str,
    description: str,
    privacy_status: str = "private",
    publish_at_iso: str = "",
    category_id: str = "20",
) -> dict[str, Any]:
    """
    Upload a vertical MP4 clip to the connected YouTube channel using
    Google YouTube Data API v3 chunked resumable upload (8 MiB chunks).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Clip file not found: {video_path}")

    priv = privacy_status.strip().lower()
    if priv not in {"public", "unlisted", "private"}:
        priv = "private"

    clean_title = (title or "Stream Highlight").strip()
    if "#shorts" not in clean_title.lower() and len(clean_title) <= 91:
        clean_title = f"{clean_title} #shorts"
    clean_title = clean_title[:100]

    clean_desc = (description or "").strip()
    if "#shorts" not in clean_desc.lower():
        clean_desc = (clean_desc + "\n#shorts").strip()

    status_block: dict[str, Any] = {
        "privacyStatus": "private" if publish_at_iso.strip() else priv,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at_iso.strip():
        status_block["publishAt"] = publish_at_iso.strip()

    body = {
        "snippet": {
            "title": clean_title,
            "description": clean_desc[:4900],
            "tags": _extract_clean_tags(clean_desc),
            "categoryId": category_id,
        },
        "status": status_block,
    }

    total_bytes = video_path.stat().st_size
    chunk_size = 8 * 1024 * 1024  # 8 MiB aligned per YouTube resumable spec (multiples of 256 KiB)

    with httpx.Client(timeout=180.0) as client:
        access_token = _get_valid_access_token(client)
        init_resp = client.post(
            YOUTUBE_RESUMABLE_INIT_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(total_bytes),
                "X-Upload-Content-Type": "video/mp4",
            },
            json=body,
        )
        if init_resp.status_code not in {200, 201}:
            raise RuntimeError(
                f"YouTube upload init failed ({init_resp.status_code}): {init_resp.text[:350]}"
            )
        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            raise RuntimeError("YouTube resumable init response missing Location header.")

        final_resp: httpx.Response | None = None
        with video_path.open("rb") as fp:
            offset = 0
            while offset < total_bytes:
                chunk = fp.read(chunk_size)
                if not chunk:
                    break
                end = offset + len(chunk) - 1
                put_resp = client.put(
                    upload_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{total_bytes}",
                        "Content-Type": "video/mp4",
                    },
                    content=chunk,
                )
                if put_resp.status_code == 308:
                    offset = end + 1
                    continue
                if put_resp.status_code in {200, 201}:
                    final_resp = put_resp
                    break
                raise RuntimeError(
                    f"YouTube chunk upload failed ({put_resp.status_code}): {put_resp.text[:350]}"
                )

    if final_resp is None:
        raise RuntimeError("YouTube upload finished without final video response.")

    data = final_resp.json()
    vid = str(data.get("id") or "")
    if not vid:
        raise RuntimeError(f"YouTube upload returned no video id: {data}")
    return {
        "video_id": vid,
        "url": f"https://youtube.com/shorts/{vid}",
        "privacy_status": status_block["privacyStatus"],
        "publish_at": status_block.get("publishAt", ""),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
