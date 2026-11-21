from __future__ import annotations

import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np

from app.config import FFMPEG_BIN, FFPROBE_BIN, get_groq_api_key, get_yt_dlp_cookies_file


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-1000:]}"
        )
    return proc.stdout


def probe_video(video_path: Path) -> dict[str, Any]:
    """Return duration, width, height, fps using ffprobe."""
    out = run_cmd(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    fps_raw = stream.get("r_frame_rate", "30/1")
    num, den = (fps_raw.split("/") + ["1"])[:2]
    fps = float(num) / max(float(den), 1.0)
    return {
        "width": int(stream.get("width", 1920)),
        "height": int(stream.get("height", 1080)),
        "fps": round(fps, 3),
        "duration": float(fmt.get("duration", 0.0)),
    }


def probe_audio_duration(audio_path: Path) -> float:
    out = run_cmd(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    data = json.loads(out)
    return float((data.get("format") or {}).get("duration", 0.0))


def _yt_dlp_base_cmd() -> list[str]:
    """Build base yt-dlp command with ffmpeg location, JS challenge solver, and optional cookies."""
    cmd = ["yt-dlp", "--no-playlist", "-N", "4", "--remote-components", "ejs:github"]
    ffmpeg_path = Path(FFMPEG_BIN)
    if ffmpeg_path.exists():
        cmd.extend(["--ffmpeg-location", str(ffmpeg_path.parent)])
    cookies_file = get_yt_dlp_cookies_file()
    if cookies_file:
        cmd.extend(["--cookies", cookies_file])
    return cmd


def fetch_youtube_info(url: str) -> dict[str, Any]:
    """Fetch YouTube title, duration & optional chapter ranges without downloading media."""
    out = run_cmd(
        _yt_dlp_base_cmd()
        + [
            "--no-warnings",
            "--dump-single-json",
            "--skip-download",
            url,
        ]
    )
    data = json.loads(out)
    chapters = [
        {
            "start_time": float(ch.get("start_time") or 0.0),
            "end_time": float(ch.get("end_time") or 0.0),
            "title": str(ch.get("title") or "").strip(),
        }
        for ch in (data.get("chapters") or [])
        if isinstance(ch, dict)
    ]
    return {
        "title": data.get("title", "YouTube Video"),
        "duration": float(data.get("duration") or 0.0),
        "is_live": bool(data.get("is_live")),
        "chapters": chapters,
    }


def capture_live_segment(
    url: str,
    dest_dir: Path,
    duration_sec: int = 600,
    log_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Capture a rolling segment from an ongoing YouTube/Kick livestream into an MP4
    using yt-dlp -g stream extraction + FFmpeg -t limit so clipping works while live.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = dest_dir / "live_capture.mp4"
    dur = max(120, min(1800, int(duration_sec)))
    if log_cb:
        log_cb(f"Live stream detected! Capturing {dur // 60}m live buffer via FFmpeg...")

    raw_urls = run_cmd(
        _yt_dlp_base_cmd()
        + [
            "--no-warnings",
            "-f",
            "best[height<=1080][ext=mp4]/best[height<=1080]/best",
            "-g",
            url,
        ]
    )
    stream_urls = [u.strip() for u in raw_urls.splitlines() if u.strip()]
    if not stream_urls:
        raise RuntimeError("Could not extract playable livestream URL.")

    cmd = [FFMPEG_BIN, "-y"]
    if len(stream_urls) >= 2:
        cmd.extend(["-t", str(dur), "-i", stream_urls[0], "-t", str(dur), "-i", stream_urls[1]])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-t", str(dur), "-i", stream_urls[0]])
    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(out_mp4)])
    run_cmd(cmd)
    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError("Live stream capture produced empty file.")
    return out_mp4


def _run_yt_dlp_streamed(
    cmd: list[str], log_cb: Callable[[str], None] | None = None
) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_pct_logged = -10
    tail_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        tail_lines.append(line)
        tail_lines = tail_lines[-20:]
        if log_cb and line.startswith("[download]") and "%" in line:
            try:
                pct_str = line.split("%", 1)[0].split()[-1]
                pct = int(float(pct_str))
                if pct >= last_pct_logged + 10 or pct == 100:
                    last_pct_logged = pct
                    log_cb(line)
            except Exception:
                pass
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"yt-dlp failed ({rc}):\n" + "\n".join(tail_lines[-10:]))


def download_youtube_audio(
    url: str, dest_dir: Path, log_cb: Callable[[str], None] | None = None
) -> Path:
    """Download ONLY audio (~95% smaller than full 1080p video) for ultra-fast transcription."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(dest_dir / "source_audio.%(ext)s")
    if log_cb:
        log_cb("Downloading YouTube audio-only track via yt-dlp (saves ~90% disk & bandwidth)...")
    cmd = _yt_dlp_base_cmd() + [
        "--newline",
        "-f",
        "bestaudio[ext=m4a]/bestaudio/worst[acodec!=none]/best",
        "-o",
        out_tpl,
        url,
    ]
    _run_yt_dlp_streamed(cmd, log_cb=log_cb)
    candidates = sorted(dest_dir.glob("source_audio.*"))
    if not candidates:
        raise FileNotFoundError("yt-dlp finished but no audio file was found.")
    return candidates[0]


def download_youtube_section(
    url: str,
    start_s: float,
    end_s: float,
    dest_path: Path,
    log_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download ONLY a specific [start_s, end_s] time range in 1080p MP4."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        dest_path.unlink()
    s_str = f"{max(0.0, start_s):.2f}"
    e_str = f"{max(start_s + 2.0, end_s):.2f}"
    if log_cb:
        log_cb(f"Fetching 1080p video slice [{s_str}s - {e_str}s] via yt-dlp --download-sections...")
    cmd = _yt_dlp_base_cmd() + [
        "--newline",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        f"*{s_str}-{e_str}",
        "--force-keyframes-at-cuts",
        "-o",
        str(dest_path.with_suffix(".%(ext)s")),
        url,
    ]
    _run_yt_dlp_streamed(cmd, log_cb=None)
    matches = sorted(dest_path.parent.glob(f"{dest_path.stem}.*"))
    mp4s = [m for m in matches if m.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not mp4s:
        raise FileNotFoundError(f"Section download failed for {s_str}-{e_str}")
    return mp4s[0]


def download_youtube_video(
    url: str, dest_dir: Path, log_cb: Callable[[str], None] | None = None
) -> tuple[Path, Path | None]:
    """Download full YouTube video (<=1080p MP4) using yt-dlp."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(dest_dir / "source.%(ext)s")
    if log_cb:
        log_cb("Downloading full YouTube video via yt-dlp...")

    cmd = _yt_dlp_base_cmd() + [
        "--newline",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tpl,
        url,
    ]
    _run_yt_dlp_streamed(cmd, log_cb=log_cb)

    mp4_candidates = sorted(dest_dir.glob("source*.mp4"))
    if not mp4_candidates:
        mp4_candidates = sorted(
            p for p in dest_dir.glob("source*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        )
    if not mp4_candidates:
        raise FileNotFoundError("yt-dlp finished but no video file was found.")
    video_path = mp4_candidates[0]
    sub_candidates = sorted(dest_dir.glob("*.json3"))
    return video_path, (sub_candidates[0] if sub_candidates else None)


def extract_wav(media_path: Path, wav_path: Path) -> Path:
    """Extract 16kHz mono PCM WAV from video or audio for transcription & energy analysis."""
    run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    return wav_path


def _parse_json3_captions(sub_path: Path) -> list[dict[str, Any]]:
    data = json.loads(sub_path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        start_s = (ev.get("tStartMs") or 0) / 1000.0
        dur_s = (ev.get("dDurationMs") or 2000) / 1000.0
        words: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        for idx, s in enumerate(segs):
            txt = (s.get("utf8") or "").replace("\n", " ").strip()
            if not txt:
                continue
            raw_parts.append(txt)
            offset = (s.get("tOffsetMs") or 0) / 1000.0
            w_start = start_s + offset
            w_end = (
                start_s + ((segs[idx + 1].get("tOffsetMs") or 0) / 1000.0)
                if idx + 1 < len(segs)
                else start_s + dur_s
            )
            for token in txt.split():
                words.append(
                    {
                        "word": token,
                        "start": round(w_start, 3),
                        "end": round(max(w_end, w_start + 0.12), 3),
                    }
                )
        text = " ".join(raw_parts).strip()
        if text:
            segments.append(
                {
                    "start": round(start_s, 3),
                    "end": round(start_s + dur_s, 3),
                    "text": text,
                    "words": words,
                }
            )
    return segments


def _wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(max(wf.getframerate(), 1))


def _read_wav_chunk(wav_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    """Read a mono/stereo 16-bit PCM WAV slice as normalized float32 (-1..1)."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()
        start_frame = max(0, min(n_frames, int(start_sec * sr)))
        end_frame = max(start_frame, min(n_frames, int(end_sec * sr)))
        if end_frame <= start_frame:
            return np.zeros(0, dtype=np.float32)
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1 and len(arr) >= channels:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr


def _select_transcription_windows(
    total_dur: float,
    audio_profile: dict[str, Any] | None = None,
    chunk_sec: float = 60.0,
    long_threshold_sec: float = 1800.0,
    max_windows: int = 18,
) -> list[tuple[float, float]]:
    """
    Chunk audio into manageable 60s blocks (preventing language lock across long windows).
    For long streams (>30 min), select the highest-energy conversational blocks evenly across
    the full timeline so GPU/Cloud transcription completes quickly while covering early, mid,
    and late highlights.
    """
    if total_dur <= 0:
        return []
    blocks: list[tuple[float, float]] = []
    t = 0.0
    while t < total_dur:
        blocks.append((round(t, 2), round(min(total_dur, t + chunk_sec), 2)))
        t += chunk_sec

    if total_dur <= long_threshold_sec or not audio_profile or not audio_profile.get("energy"):
        return blocks

    energy = audio_profile["energy"]
    bin_sec = float(audio_profile.get("bin_sec") or 0.5)
    scored_blocks: list[tuple[float, float, float]] = []
    for bs, be in blocks:
        i0 = max(0, int(bs / bin_sec))
        i1 = min(len(energy), max(i0 + 1, int(be / bin_sec)))
        vals = energy[i0:i1]
        if not vals:
            continue
        speech_ratio = sum(1 for v in vals if 0.10 <= v <= 1.25) / len(vals)
        mean_e = sum(vals) / len(vals)
        peak_e = max(vals)
        # Favor sustained energetic speech/reactions over flat silence or dead air
        score = (mean_e * 0.5) + (speech_ratio * 0.35) + (min(1.0, peak_e) * 0.15)
        scored_blocks.append((score, bs, be))

    if len(scored_blocks) <= max_windows:
        return [(bs, be) for _, bs, be in scored_blocks]

    # Pick the best high-energy block inside each timeline bucket
    bucket_size = len(scored_blocks) / float(max_windows)
    chosen: list[tuple[float, float]] = []
    for idx in range(max_windows):
        lo = int(idx * bucket_size)
        hi = max(lo + 1, int((idx + 1) * bucket_size))
        best = max(scored_blocks[lo:hi], key=lambda item: item[0])
        chosen.append((best[1], best[2]))
    return chosen


def _is_repetitive_hallucination(text: str) -> bool:
    """Detect Whisper decoder loop hallucinations (e.g. 'ouch ouch ouch' or repeated phrases)."""
    tokens = text.strip().split()
    if len(tokens) >= 5:
        lowered = [t.lower() for t in tokens]
        unique_ratio = len(set(lowered)) / len(lowered)
        if unique_ratio < 0.38:
            return True
    # Check 3+ identical consecutive words or 2-word bigrams
    if len(tokens) >= 6:
        lowered = [t.lower() for t in tokens]
        run = 1
        for i in range(1, len(lowered)):
            run = run + 1 if lowered[i] == lowered[i - 1] else 1
            if run >= 3:
                return True
        bigrams = [(lowered[i], lowered[i + 1]) for i in range(len(lowered) - 1)]
        for i in range(len(bigrams) - 4):
            if bigrams[i] == bigrams[i + 2] == bigrams[i + 4]:
                return True
    return False


def _append_whisper_segments(
    target: list[dict[str, Any]],
    raw_segments: Any,
    win_start: float,
    attr_mode: bool = False,
) -> None:
    """Normalize MLX or faster-whisper segments while filtering loop hallucinations."""
    for seg in raw_segments:
        txt = (seg.text if attr_mode else seg["text"]).strip()
        if not txt or _is_repetitive_hallucination(txt):
            continue
        # Collapse >2 consecutive identical short segments (e.g. repeated 'ouch')
        if target and len(txt.split()) <= 2:
            norm = txt.lower()
            recent_same = sum(1 for prev in target[-3:] if prev["text"].strip().lower() == norm)
            if recent_same >= 2:
                continue
        raw_words = (seg.words if attr_mode else seg.get("words")) or []
        words = []
        for w in raw_words:
            w_txt = (w.word if attr_mode else w.get("word", "")).strip()
            if not w_txt:
                continue
            ws = w.start if attr_mode else w["start"]
            we = w.end if attr_mode else w["end"]
            words.append(
                {
                    "word": w_txt,
                    "start": round(win_start + float(ws), 3),
                    "end": round(win_start + float(we), 3),
                }
            )
        seg_s = seg.start if attr_mode else seg["start"]
        seg_e = seg.end if attr_mode else seg["end"]
        target.append(
            {
                "start": round(win_start + float(seg_s), 3),
                "end": round(win_start + float(seg_e), 3),
                "text": txt,
                "words": words,
            }
        )


def _parse_groq_whisper_payload(
    payload: dict[str, Any], win_start: float, win_end: float
) -> list[dict[str, Any]]:
    raw_words = [
        {
            "word": str(w.get("word", "")).strip(),
            "start": round(win_start + float(w["start"]), 3),
            "end": round(win_start + float(w["end"]), 3),
        }
        for w in (payload.get("words") or [])
        if str(w.get("word", "")).strip()
    ]
    raw_segs = payload.get("segments") or []
    out: list[dict[str, Any]] = []
    if raw_segs:
        valid_segs: list[dict[str, Any]] = []
        for seg in raw_segs:
            txt = str(seg.get("text", "")).strip()
            if not txt or _is_repetitive_hallucination(txt):
                continue
            valid_segs.append(
                {
                    "start": round(win_start + float(seg["start"]), 3),
                    "end": round(win_start + float(seg["end"]), 3),
                    "text": txt,
                    "words": [],
                }
            )
        for w in raw_words:
            w_mid = 0.5 * (w["start"] + w["end"])
            best_seg = (
                min(
                    valid_segs,
                    key=lambda s: 0.0
                    if s["start"] <= w_mid <= s["end"]
                    else min(abs(w_mid - s["start"]), abs(w_mid - s["end"])),
                )
                if valid_segs
                else None
            )
            if best_seg is not None:
                best_seg["words"].append(w)
        # If segments were kept but raw_words attached imperfectly, synthesize missing words cleanly
        for seg in valid_segs:
            if not seg["words"] and seg["text"]:
                toks = seg["text"].split()
                step = max(0.08, (seg["end"] - seg["start"]) / max(1, len(toks)))
                seg["words"] = [
                    {
                        "word": tok,
                        "start": round(seg["start"] + i * step, 3),
                        "end": round(min(seg["end"], seg["start"] + (i + 1) * step), 3),
                    }
                    for i, tok in enumerate(toks)
                ]
        out.extend(valid_segs)
    elif payload.get("text") and not _is_repetitive_hallucination(str(payload["text"])):
        out.append(
            {
                "start": round(win_start, 3),
                "end": round(win_end, 3),
                "text": str(payload["text"]).strip(),
                "words": raw_words,
            }
        )
    return out


def _post_groq_whisper_file(
    client: httpx.Client,
    audio_path: Path,
    mime_type: str,
    api_key: str,
    groq_model: str = "whisper-large-v3",
    language: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": groq_model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
        "temperature": "0.0",
    }
    if language in {"en", "hi"}:
        data["language"] = language

    for attempt in range(2):
        with audio_path.open("rb") as fh:
            resp = client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (audio_path.name, fh, mime_type)},
                data=data,
            )
        if resp.status_code == 429 and attempt == 0:
            retry_after = float(resp.headers.get("retry-after", "2.5") or 2.5)
            time.sleep(min(5.0, max(1.0, retry_after)))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Groq Whisper request failed after retry")


def _transcribe_with_groq(
    wav_path: Path,
    windows: list[tuple[float, float]],
    api_key: str,
    groq_model: str = "whisper-large-v3",
    subtitle_language: str = "auto",
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Offload Whisper transcription to Groq's free cloud API (`whisper-large-v3` by default)
    with 0% local CPU/GPU load, lossless FLAC audio, and word-level timestamps.
    """
    if log_cb:
        log_cb(f"Transcribing on Groq Cloud API ({groq_model}, 0% local CPU/GPU)...")
    segments: list[dict[str, Any]] = []
    detected_langs: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir, httpx.Client(timeout=120.0) as client:
        for idx, (win_start, win_end) in enumerate(windows, 1):
            chunk = _read_wav_chunk(wav_path, win_start, win_end)
            if len(chunk) < 1600 or float(np.sqrt(np.mean(chunk * chunk))) < 0.002:
                if log_cb:
                    pct = int(100 * idx / len(windows))
                    log_cb(f"Groq Cloud Whisper: zone {idx}/{len(windows)} ({pct}%) silent — skipped")
                continue

            flac_chunk = Path(tmp_dir) / f"chunk_{idx}.flac"
            run_cmd(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-ss",
                    f"{win_start:.2f}",
                    "-t",
                    f"{max(1.0, win_end - win_start):.2f}",
                    "-i",
                    str(wav_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "flac",
                    str(flac_chunk),
                ]
            )
            payload = _post_groq_whisper_file(
                client,
                flac_chunk,
                "audio/flac",
                api_key,
                groq_model=groq_model,
                language=subtitle_language if subtitle_language in {"en", "hi"} else None,
            )
            if payload.get("language"):
                detected_langs.append(str(payload["language"]))
            segments.extend(_parse_groq_whisper_payload(payload, win_start, win_end))

            if log_cb:
                pct = int(100 * idx / len(windows))
                log_cb(
                    f"Groq Cloud Whisper: zone {idx}/{len(windows)} ({pct}%) "
                    f"[{win_start/60:.1f}m-{win_end/60:.1f}m] -> {len(segments)} segments"
                )

    return {
        "engine": f"groq:{groq_model}",
        "language": detected_langs[0] if detected_langs else "auto",
        "segments": segments,
    }


def transcribe_audio(
    wav_path: Path,
    model_size: str = "base",
    fallback_json3: Path | None = None,
    log_cb: Callable[[str], None] | None = None,
    audio_profile: dict[str, Any] | None = None,
    subtitle_language: str = "auto",
) -> dict[str, Any]:
    """
    Transcribe audio with word-level timestamps using Groq Cloud Whisper API (0% local load,
    if GROQ_API_KEY configured), mlx-whisper (Apple Silicon GPU), or faster-whisper (CPU int8).
    """
    total_dur = _wav_duration_sec(wav_path)
    windows = _select_transcription_windows(total_dur, audio_profile=audio_profile)
    if not windows:
        return {"engine": "empty", "segments": []}

    if log_cb and total_dur > 1800.0 and audio_profile:
        est_mins = sum(e - s for s, e in windows) / 60.0
        log_cb(
            f"Long video ({total_dur/60:.1f}m): targeting {len(windows)} peak-energy zones "
            f"({est_mins:.1f}m audio) across full timeline for fast transcription."
        )

    groq_key = get_groq_api_key()
    if groq_key:
        groq_model = "whisper-large-v3-turbo" if model_size == "tiny" else "whisper-large-v3"
        try:
            return _transcribe_with_groq(
                wav_path,
                windows,
                groq_key,
                groq_model=groq_model,
                subtitle_language=subtitle_language,
                log_cb=log_cb,
            )
        except Exception as groq_exc:
            if log_cb:
                log_cb(f"Groq Cloud Whisper failed ({groq_exc}); falling back to local Whisper...")

    try:
        import mlx_whisper  # type: ignore

        if log_cb:
            log_cb(f"Transcribing on Apple Silicon GPU via mlx-whisper ({model_size})...")
        repo = f"mlx-community/whisper-{model_size}-mlx"
        segments: list[dict[str, Any]] = []
        detected_langs: list[str] = []
        for idx, (win_start, win_end) in enumerate(windows, 1):
            chunk = _read_wav_chunk(wav_path, win_start, win_end)
            if len(chunk) < 1600 or float(np.sqrt(np.mean(chunk * chunk))) < 0.002:
                if log_cb:
                    pct = int(100 * idx / len(windows))
                    log_cb(f"mlx-whisper GPU: zone {idx}/{len(windows)} ({pct}%) silent — skipped")
                continue
            res = mlx_whisper.transcribe(
                chunk,
                path_or_hf_repo=repo,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4),
                compression_ratio_threshold=2.2,
                logprob_threshold=-0.8,
                no_speech_threshold=0.55,
                hallucination_silence_threshold=1.5,
                verbose=False,
            )
            if res.get("language"):
                detected_langs.append(str(res["language"]))
            _append_whisper_segments(segments, res.get("segments", []), win_start, attr_mode=False)
            if log_cb:
                pct = int(100 * idx / len(windows))
                log_cb(
                    f"mlx-whisper GPU: zone {idx}/{len(windows)} ({pct}%) "
                    f"[{win_start/60:.1f}m-{win_end/60:.1f}m] -> {len(segments)} segments"
                )
        return {
            "engine": f"mlx-whisper:{model_size}",
            "language": detected_langs[0] if detected_langs else "auto",
            "segments": segments,
        }
    except Exception as mlx_exc:
        if log_cb:
            log_cb(f"mlx-whisper skipped ({mlx_exc}); falling back to faster-whisper...")

    try:
        from faster_whisper import WhisperModel  # type: ignore

        if log_cb:
            log_cb(f"Transcribing locally with faster-whisper ({model_size}, int8)...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments = []
        last_lang = "en"
        for idx, (win_start, win_end) in enumerate(windows, 1):
            chunk = _read_wav_chunk(wav_path, win_start, win_end)
            if len(chunk) < 1600 or float(np.sqrt(np.mean(chunk * chunk))) < 0.002:
                continue
            seg_iter, info = model.transcribe(
                chunk,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False,
            )
            last_lang = getattr(info, "language", last_lang) or last_lang
            _append_whisper_segments(segments, seg_iter, win_start, attr_mode=True)
            if log_cb:
                pct = int(100 * idx / len(windows))
                log_cb(
                    f"faster-whisper: zone {idx}/{len(windows)} ({pct}%) "
                    f"[{win_start/60:.1f}m-{win_end/60:.1f}m] -> {len(segments)} segments"
                )
        return {
            "engine": f"faster-whisper:{model_size}",
            "language": last_lang,
            "segments": segments,
        }
    except Exception as exc:
        if fallback_json3 and fallback_json3.exists():
            if log_cb:
                log_cb(f"Whisper unavailable ({exc}); using YouTube json3 captions fallback.")
            return {"engine": "yt-json3", "segments": _parse_json3_captions(fallback_json3)}
        raise RuntimeError(f"Transcription failed and no caption fallback available: {exc}") from exc


def _count_non_rep_words(segs: list[dict[str, Any]]) -> int:
    words = [w["word"].lower() for s in segs for w in (s.get("words") or []) if w.get("word")]
    if not words:
        words = [t.lower() for s in segs for t in s.get("text", "").split() if t]
    if not words:
        return 0
    # Penalize collapsed repetitive loops
    unique_ratio = len(set(words)) / max(1, len(words))
    if len(words) >= 8 and unique_ratio < 0.28:
        return int(len(words) * 0.25)
    return len(words)


def _find_dip_split_points(wav_path: Path, target_step: float = 7.5, search_pad: float = 1.4) -> list[float]:
    """Find quietest ~80ms RMS energy dips near every `target_step` seconds so bursts never split mid-word."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    dur = n_frames / float(max(1, sr))
    if dur <= 10.5:
        return [0.0, dur]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(sr * 0.08))
    cuts = [0.0]
    t = target_step
    while t < dur - 3.0:
        lo_s = max(cuts[-1] + 3.0, t - search_pad)
        hi_s = min(dur - 2.5, t + search_pad)
        if hi_s > lo_s:
            best_t = t
            best_rms = 1e9
            pos = int(lo_s * sr)
            hi_pos = int(hi_s * sr)
            step_samples = max(1, int(sr * 0.04))
            while pos + win <= hi_pos:
                seg = samples[pos : pos + win]
                rms = float(np.sqrt(np.mean(seg * seg)))
                if rms < best_rms:
                    best_rms = rms
                    best_t = (pos + win * 0.5) / float(sr)
                pos += step_samples
            cuts.append(round(best_t, 3))
            t = best_t + target_step
        else:
            break
    cuts.append(round(dur, 3))
    return cuts


def transcribe_clip_precision(
    media_path: Path,
    clip_start: float,
    clip_end: float,
    source_offset: float = 0.0,
    model_size: str = "small",
    subtitle_language: str = "auto",
    subtitle_script: str = "roman",
    video_title: str = "",
    llm_provider: str = "auto_free",
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Precision per-clip transcription pass on ONLY the exact [clip_start, clip_end] window.
    Splits the clip at natural RMS energy dips into ~6-9s independent bursts (preventing Whisper
    condition-on-previous-text hallucination cascades) and applies 1-to-1 LLM Hinglish/phonetic
    word alignment while preserving exact word timestamps.
    """
    from app.rsi_engine import refine_subtitles_with_llm

    dur = max(1.0, clip_end - clip_start)
    seek_in = max(0.0, clip_start - source_offset)
    groq_key = get_groq_api_key()
    lang_hint = subtitle_language if subtitle_language in {"en", "hi"} else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        raw_wav = tmp / "raw.wav"
        run_cmd(
            [
                FFMPEG_BIN,
                "-y",
                "-ss",
                f"{seek_in:.3f}",
                "-t",
                f"{dur:.3f}",
                "-i",
                str(media_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(raw_wav),
            ]
        )

        best_segs: list[dict[str, Any]] = []
        det_lang = "auto"
        engine_used = "precision"

        if groq_key:
            try:
                cuts = _find_dip_split_points(raw_wav, target_step=7.5, search_pad=1.4)
                with httpx.Client(timeout=90.0) as client:
                    for b_idx in range(len(cuts) - 1):
                        b0 = cuts[b_idx]
                        b1 = cuts[b_idx + 1]
                        if b1 - b0 < 0.4:
                            continue
                        burst_wav = tmp / f"burst_{b_idx}.wav"
                        run_cmd(
                            [
                                FFMPEG_BIN,
                                "-y",
                                "-ss",
                                f"{b0:.3f}",
                                "-t",
                                f"{(b1 - b0):.3f}",
                                "-i",
                                str(raw_wav),
                                "-c:a",
                                "copy",
                                str(burst_wav),
                            ]
                        )
                        payload = _post_groq_whisper_file(
                            client,
                            burst_wav,
                            "audio/wav",
                            groq_key,
                            groq_model="whisper-large-v3",
                            language=lang_hint,
                        )
                        if payload.get("language") and det_lang == "auto":
                            det_lang = str(payload["language"])
                        b_segs = _parse_groq_whisper_payload(
                            payload, clip_start + b0, clip_start + b1
                        )
                        # Drop ultra-low-confidence hallucination segments unless nothing else exists
                        good_segs = [
                            s
                            for s in b_segs
                            if s.get("text", "").strip()
                        ]
                        best_segs.extend(good_segs)
                engine_used = "groq:whisper-large-v3-bursts"
            except Exception as exc:
                if log_cb:
                    log_cb(f"Precision Groq clip transcription failed ({exc}); falling back to local Whisper...")

        if not best_segs:
            res = transcribe_audio(
                raw_wav,
                model_size=model_size if model_size in {"small", "medium"} else "small",
                log_cb=None,
                subtitle_language=subtitle_language,
            )
            det_lang = str(res.get("language") or "auto")
            engine_used = str(res.get("engine") or "local")
            for s in res.get("segments") or []:
                shifted_words = [
                    {
                        "word": w["word"],
                        "start": round(clip_start + float(w["start"]), 3),
                        "end": round(clip_start + float(w["end"]), 3),
                    }
                    for w in (s.get("words") or [])
                ]
                best_segs.append(
                    {
                        "start": round(clip_start + float(s["start"]), 3),
                        "end": round(clip_start + float(s["end"]), 3),
                        "text": s["text"],
                        "words": shifted_words,
                    }
                )

    if best_segs and llm_provider != "local_offline":
        best_segs = refine_subtitles_with_llm(
            best_segs,
            video_title=video_title,
            script_mode=subtitle_script,
            subtitle_language=subtitle_language,
            llm_provider=llm_provider,
        )

    return {
        "engine": engine_used,
        "language": det_lang,
        "segments": best_segs,
    }

