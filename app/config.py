from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

FFMPEG_FULL_BIN = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
FFPROBE_FULL_BIN = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")
FFMPEG_BIN = (
    str(FFMPEG_FULL_BIN)
    if FFMPEG_FULL_BIN.exists()
    else (shutil.which("ffmpeg") or "ffmpeg")
)
FFPROBE_BIN = (
    str(FFPROBE_FULL_BIN)
    if FFPROBE_FULL_BIN.exists()
    else (shutil.which("ffprobe") or "ffprobe")
)

DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
STATIC_DIR = BASE_DIR / "static"
RSI_MEMORY_PATH = DATA_DIR / "rsi_memory.json"
SETTINGS_PATH = DATA_DIR / "settings.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COOKIES_STORE_PATH = DATA_DIR / "youtube_cookies.txt"


def load_settings() -> dict[str, Any]:
    default_storage = os.getenv("STORAGE_DIR", "").strip() or str(BASE_DIR)
    defaults = {
        "storage_dir": default_storage,
        "auto_cleanup_work": True,
        "yt_dlp_cookies_file": os.getenv("YT_DLP_COOKIES_FILE", "").strip(),
        "groq_api_key": GROQ_API_KEY,
    }
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in saved.items() if v is not None})
        except Exception:
            pass
    # Ensure configured storage_dir is usable, fallback to BASE_DIR if unmounted
    st = Path(defaults["storage_dir"]).expanduser()
    if not st.exists():
        try:
            st.mkdir(parents=True, exist_ok=True)
        except Exception:
            defaults["storage_dir"] = str(BASE_DIR)
    return defaults


def save_settings(new_settings: dict[str, Any]) -> dict[str, Any]:
    curr = load_settings()
    curr.update(new_settings)
    st = Path(str(curr["storage_dir"])).expanduser()
    st.mkdir(parents=True, exist_ok=True)
    curr["storage_dir"] = str(st.resolve())
    SETTINGS_PATH.write_text(json.dumps(curr, indent=2), encoding="utf-8")
    for fn in (get_upload_dir, get_work_dir, get_output_dir):
        fn()
    return curr


def get_storage_root() -> Path:
    root = Path(load_settings()["storage_dir"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_upload_dir() -> Path:
    d = get_storage_root() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_work_dir() -> Path:
    d = get_storage_root() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_dir() -> Path:
    d = get_storage_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Keep module-level constants initialized for compatibility
UPLOAD_DIR = get_upload_dir()
WORK_DIR = get_work_dir()
OUTPUT_DIR = get_output_dir()


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def list_storage_volumes() -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path]] = [("Workspace (Internal)", BASE_DIR)]
    onedrive = Path.home() / "OneDrive"
    if onedrive.exists():
        candidates.append(("OneDrive Folder", onedrive / "RSI_Shorts"))
    vols = Path("/Volumes")
    if vols.exists():
        for entry in sorted(vols.iterdir()):
            if entry.name in {"Macintosh HD", ".timemachine"}:
                continue
            if "Installer" in entry.name:
                continue
            # Check writable
            try:
                usage = shutil.disk_usage(entry)
                if usage.total < 500 * 1024 * 1024:
                    continue
                if os.access(entry, os.W_OK):
                    candidates.append((f"Volume: {entry.name}", entry / "RSI_Shorts"))
            except OSError:
                continue

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, path in candidates:
        probe = path if path.exists() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "label": label,
                "path": key,
                "free_gb": round(usage.free / (1024**3), 1),
                "total_gb": round(usage.total / (1024**3), 1),
            }
        )
    return results


def get_yt_dlp_cookies_file() -> str:
    val = str(load_settings().get("yt_dlp_cookies_file") or "").strip()
    if val and Path(val).expanduser().is_file():
        return str(Path(val).expanduser())
    if COOKIES_STORE_PATH.is_file() and COOKIES_STORE_PATH.stat().st_size > 0:
        return str(COOKIES_STORE_PATH)
    return ""


def get_groq_api_key() -> str:
    val = str(load_settings().get("groq_api_key") or "").strip()
    return val or GROQ_API_KEY


def save_youtube_cookies_text(cookies_text: str) -> None:
    cleaned = cookies_text.strip()
    if not cleaned:
        if COOKIES_STORE_PATH.exists():
            COOKIES_STORE_PATH.unlink()
        return
    COOKIES_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIES_STORE_PATH.write_text(cleaned + "\n", encoding="utf-8")


def get_storage_status() -> dict[str, Any]:
    settings = load_settings()
    root = get_storage_root()
    usage = shutil.disk_usage(root)
    groq_key = get_groq_api_key()
    return {
        "storage_dir": str(root),
        "auto_cleanup_work": bool(settings.get("auto_cleanup_work", True)),
        "yt_dlp_cookies_file": str(settings.get("yt_dlp_cookies_file") or ""),
        "cookies_configured": bool(get_yt_dlp_cookies_file()),
        "groq_configured": bool(groq_key),
        "groq_key_hint": f"...{groq_key[-4:]}" if len(groq_key) >= 4 else "",
        "free_gb": round(usage.free / (1024**3), 1),
        "total_gb": round(usage.total / (1024**3), 1),
        "work_mb": round(_dir_size_bytes(get_work_dir()) / (1024**2), 1),
        "uploads_mb": round(_dir_size_bytes(get_upload_dir()) / (1024**2), 1),
        "outputs_mb": round(_dir_size_bytes(get_output_dir()) / (1024**2), 1),
        "volumes": list_storage_volumes(),
    }
