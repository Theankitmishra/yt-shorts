from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import FFMPEG_BIN


@lru_cache(maxsize=1)
def _pick_caption_font(needs_devanagari: bool = False) -> str:
    if needs_devanagari:
        try:
            proc = subprocess.run(["fc-list", ":family"], capture_output=True, text=True, timeout=3)
            fams = proc.stdout
            for cand in ("Kohinoor Devanagari", "Noto Sans Devanagari", "Devanagari Sangam MN", "Arial Unicode MS"):
                if cand.lower() in fams.lower():
                    return cand
        except Exception:
            pass
    return "Arial"


@lru_cache(maxsize=1)
def _ffmpeg_has_ass_filter() -> bool:
    proc = subprocess.run([FFMPEG_BIN, "-filters"], capture_output=True, text=True)
    return " ass " in proc.stdout or "\n .. ass " in proc.stdout or "libass" in proc.stdout


def _format_ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(txt: str) -> str:
    return txt.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _escape_ffmpeg_filter_path(path: Path) -> str:
    s = str(path.resolve())
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def generate_ass_subtitles(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    hook_title: str,
    ass_path: Path,
    burn_hook_banner: bool = True,
    watermark_text: str = "",
    burn_captions: bool = True,
) -> Path:
    """Create a 1080x1920 ASS subtitle file with karaoke word highlighting, hook banner & optional bounty watermark."""
    all_text = hook_title + " " + watermark_text + " " + " ".join(str(s.get("text", "")) for s in segments)
    needs_devanagari = any("\u0900" <= ch <= "\u097f" for ch in all_text)
    font_name = _pick_caption_font(needs_devanagari=needs_devanagari)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},68,&H00FFFFFF,&H0000FFFF,&H00111111,&H90000000,-1,0,0,0,100,100,0,0,1,4,2,2,80,80,310,1
Style: HookBanner,{font_name},54,&H00111111,&H000000FF,&H0000E5FF,&H0000E5FF,-1,0,0,0,100,100,0,0,3,18,0,8,90,90,170,1
Style: Watermark,{font_name},30,&H44FFFFFF,&H000000FF,&H77111111,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,9,45,45,62,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    if watermark_text.strip():
        clip_dur = max(1.0, clip_end - clip_start)
        events.append(
            f"Dialogue: 2,{_format_ass_time(0.0)},{_format_ass_time(clip_dur)},Watermark,,0,0,0,,{_escape_ass_text(watermark_text.strip())}"
        )

    if burn_hook_banner and hook_title.strip():
        banner_end = min(3.8, max(2.0, (clip_end - clip_start) * 0.25))
        events.append(
            f"Dialogue: 1,{_format_ass_time(0.0)},{_format_ass_time(banner_end)},HookBanner,,0,0,0,,{_escape_ass_text(hook_title.strip())}"
        )

    if not burn_captions:
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
        return ass_path

    # Collect relative words inside [clip_start, clip_end]
    rel_words: list[dict[str, Any]] = []
    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                if w["end"] >= clip_start and w["start"] <= clip_end:
                    ws = max(0.0, float(w["start"]) - clip_start)
                    we = max(ws + 0.08, min(clip_end - clip_start, float(w["end"]) - clip_start))
                    wt = w["word"].strip()
                    if wt:
                        rel_words.append({"word": wt, "start": ws, "end": we})
        else:
            tokens = seg["text"].split()
            if not tokens:
                continue
            s0 = max(clip_start, float(seg["start"]))
            s1 = min(clip_end, float(seg["end"]))
            step = max(0.12, (s1 - s0) / len(tokens))
            for idx, tok in enumerate(tokens):
                ws = max(0.0, s0 + idx * step - clip_start)
                we = min(clip_end - clip_start, ws + step)
                rel_words.append({"word": tok, "start": ws, "end": we})

    # Enforce monotonically non-overlapping word timing so karaoke chunks never stack vertically
    for idx in range(1, len(rel_words)):
        if rel_words[idx]["start"] < rel_words[idx - 1]["start"] + 0.04:
            rel_words[idx]["start"] = round(rel_words[idx - 1]["start"] + 0.04, 3)
        if rel_words[idx - 1]["end"] > rel_words[idx]["start"]:
            rel_words[idx - 1]["end"] = round(
                max(rel_words[idx - 1]["start"] + 0.04, rel_words[idx]["start"] - 0.01), 3
            )

    # Split into natural phrase chunks on pauses (>0.35s), punctuation, max 4 words or 26 chars
    chunks: list[list[dict[str, Any]]] = []
    cur_chunk: list[dict[str, Any]] = []
    for idx, wobj in enumerate(rel_words):
        cur_chunk.append(wobj)
        total_chars = sum(len(x["word"]) for x in cur_chunk) + max(0, len(cur_chunk) - 1)
        gap_next = (
            rel_words[idx + 1]["start"] - wobj["end"] if idx + 1 < len(rel_words) else 99.0
        )
        ends_sentence = bool(re.search(r"[.?!।]$", wobj["word"]))
        if len(cur_chunk) >= 4 or total_chars >= 26 or gap_next > 0.35 or ends_sentence:
            chunks.append(cur_chunk)
            cur_chunk = []
    if cur_chunk:
        chunks.append(cur_chunk)

    for c_idx, chunk in enumerate(chunks):
        next_chunk_start = chunks[c_idx + 1][0]["start"] if c_idx + 1 < len(chunks) else None
        for active_idx, target in enumerate(chunk):
            sub_start = target["start"]
            if active_idx + 1 < len(chunk):
                sub_end = chunk[active_idx + 1]["start"]
            else:
                sub_end = chunk[-1]["end"] + 0.14
                if next_chunk_start is not None:
                    sub_end = min(sub_end, next_chunk_start - 0.01)
            if sub_end <= sub_start:
                sub_end = sub_start + 0.08
            parts: list[str] = []
            for j, wobj in enumerate(chunk):
                wt = _escape_ass_text(wobj["word"])
                if j == active_idx:
                    parts.append(r"{\c&H0000E5FF&\fscx108\fscy108}" + wt + r"{\c&H00FFFFFF&\fscx100\fscy100}")
                else:
                    parts.append(wt)
            events.append(
                f"Dialogue: 0,{_format_ass_time(sub_start)},{_format_ass_time(sub_end)},Caption,,0,0,0,,{' '.join(parts)}"
            )

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return ass_path


def _compute_smooth_crop_x(
    visual: dict[str, Any],
    clip_start: float,
    clip_end: float,
    src_w: int,
    src_h: int,
) -> tuple[int, int]:
    """Compute smoothed start/end X pixel offsets for a 9:16 crop window."""
    crop_w = min(src_w, int(round(src_h * 9.0 / 16.0)))
    max_x = max(0, src_w - crop_w)
    if max_x == 0:
        return 0, 0

    frames = [
        f
        for f in visual.get("frames", [])
        if clip_start <= f["t"] <= clip_end and f.get("primary_cx") is not None
    ]
    if not frames:
        center = max_x // 2
        return center, center

    mid = (clip_start + clip_end) / 2.0
    first_half = [f["primary_cx"] for f in frames if f["t"] <= mid] or [f["primary_cx"] for f in frames]
    second_half = [f["primary_cx"] for f in frames if f["t"] > mid] or [f["primary_cx"] for f in frames]

    cx1 = sum(first_half) / len(first_half)
    cx2 = sum(second_half) / len(second_half)

    x1 = int(max(0, min(max_x, cx1 * src_w - crop_w / 2.0)))
    x2 = int(max(0, min(max_x, cx2 * src_w - crop_w / 2.0)))
    return x1, x2


def _resolve_ffprobe_bin() -> str:
    sibling = Path(FFMPEG_BIN).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


def _probe_source_fps(video_path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                _resolve_ffprobe_bin(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
            timeout=5,
        ).strip()
        if "/" in out:
            num, den = out.split("/", 1)
            return float(num) / max(1.0, float(den))
        return float(out or 30.0)
    except Exception:
        return 30.0


def render_short_clip(
    source_video: Path,
    output_path: Path,
    segments: list[dict[str, Any]],
    visual: dict[str, Any],
    clip: dict[str, Any],
    src_w: int,
    src_h: int,
    work_dir: Path,
    burn_captions: bool = True,
    source_offset: float = 0.0,
    has_audio: bool = True,
) -> Path:
    """Render a 1080x1920 vertical short clip with chosen framing and burned-in ASS captions."""
    start = float(clip["start"])
    end = float(clip["end"])
    duration = max(1.0, end - start)
    seek_in_file = max(0.0, start - source_offset)
    layout = clip.get("layout", "blurred_bg_fit")

    watermark_text = str(clip.get("watermark_text") or "").strip()
    use_ass = (burn_captions or bool(watermark_text)) and _ffmpeg_has_ass_filter()
    ass_path = work_dir / f"{clip['clip_id']}.ass"
    if use_ass:
        generate_ass_subtitles(
            segments=segments,
            clip_start=start,
            clip_end=end,
            hook_title=clip.get("hook_title", ""),
            ass_path=ass_path,
            burn_hook_banner=burn_captions,
            watermark_text=watermark_text,
            burn_captions=burn_captions,
        )

    ass_filter = f",ass='{_escape_ffmpeg_filter_path(ass_path)}'" if use_ass else ""
    src_fps = _probe_source_fps(source_video)
    out_fps = 60 if src_fps >= 48.0 else 30

    if layout == "blurred_bg_fit":
        fc = (
            f"[0:v]fps={out_fps},scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,gblur=sigma=8,scale=1080:1920,eq=brightness=-0.08[bg];"
            f"[0:v]fps={out_fps},scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{ass_filter}[vout]"
        )
    elif layout == "split_screen_focus":
        # Top: full-width/slightly zoomed screen or context (1080x1120); Bottom: face/speaker region (1080x800)
        x1, _ = _compute_smooth_crop_x(visual, start, end, src_w, src_h)
        crop_w = min(src_w, int(round(src_h * 9.0 / 16.0)))
        fc = (
            f"[0:v]fps={out_fps},scale=1080:1120:force_original_aspect_ratio=increase,crop=1080:1120[top];"
            f"[0:v]fps={out_fps},crop={crop_w}:{src_h}:{x1}:0,scale=1080:800:force_original_aspect_ratio=increase,crop=1080:800[bot];"
            f"[top][bot]vstack=inputs=2{ass_filter}[vout]"
        )
    elif layout == "face_track_9_16":
        crop_w = min(src_w, int(round(src_h * 9.0 / 16.0)))
        x1, x2 = _compute_smooth_crop_x(visual, start, end, src_w, src_h)
        if abs(x2 - x1) > 20:
            x_expr = f"'{x1}+({x2}-{x1})*min(1,t/{duration:.2f})'"
        else:
            x_expr = str(x1)
        fc = (
            f"[0:v]fps={out_fps},crop={crop_w}:{src_h}:{x_expr}:0,scale=1080:1920,setsar=1{ass_filter}[vout]"
        )
    else:  # center_9_16
        crop_w = min(src_w, int(round(src_h * 9.0 / 16.0)))
        x_mid = max(0, (src_w - crop_w) // 2)
        fc = f"[0:v]fps={out_fps},crop={crop_w}:{src_h}:{x_mid}:0,scale=1080:1920,setsar=1{ass_filter}[vout]"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{seek_in_file:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
    ]
    if has_audio:
        cmd.extend(["-map", "0:a?", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed for {clip['clip_id']}:\n{proc.stderr[-1200:]}")
    return output_path
