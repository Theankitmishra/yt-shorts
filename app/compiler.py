from __future__ import annotations

import json
import math
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.config import BASE_DIR, FFMPEG_BIN, get_output_dir
from app.renderer import (
    _escape_ass_text,
    _escape_ffmpeg_filter_path,
    _ffmpeg_has_ass_filter,
    _format_ass_time,
    _pick_caption_font,
)
from app.transcriber import download_youtube_section, probe_video, run_cmd


def _fmt_chapter_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _write_stereo_wav(path: Path, samples_mono: np.ndarray, sr: int = 44100) -> None:
    clean = np.nan_to_num(samples_mono, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.clip(clean * 32767.0, -32767, 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())


def _generate_sfx_bank(work_dir: Path) -> dict[str, Path]:
    """Synthesize clean copyright-free transition SFX (whoosh, pop, vinyl cut)."""
    sr = 44100
    rng = np.random.default_rng(42)

    # 1. Whoosh (0.45s swept noise + low thump)
    d_w = 0.45
    t_w = np.linspace(0, d_w, int(sr * d_w), dtype=np.float32)
    noise = rng.normal(0, 1, len(t_w)).astype(np.float32)
    # Smooth moving-average low-pass sweep proxy + band envelope
    kernel = np.ones(18, dtype=np.float32) / 18.0
    smooth_n = np.convolve(noise, kernel, mode="same")
    env_w = np.maximum(0.0, np.sin(math.pi * (t_w / d_w))) ** 1.4
    thump = np.sin(2.0 * math.pi * (95.0 - 55.0 * (t_w / d_w)) * t_w) * np.exp(-t_w * 9.0) * 0.45
    whoosh_p = work_dir / "sfx_whoosh.wav"
    _write_stereo_wav(whoosh_p, (smooth_n * env_w * 0.55 + thump) * 0.5, sr)

    # 2. Pop (0.14s crisp UI pop for chapter numbers)
    d_p = 0.14
    t_p = np.linspace(0, d_p, int(sr * d_p), dtype=np.float32)
    freq_p = 420.0 + 420.0 * np.exp(-t_p * 32.0)
    pop_sig = np.sin(2.0 * math.pi * freq_p * t_p) * np.exp(-t_p * 28.0) * 0.45
    pop_p = work_dir / "sfx_pop.wav"
    _write_stereo_wav(pop_p, pop_sig, sr)

    # 3. Vinyl Cut / Rewind Stop (0.35s downward pitch sweep for Cold Open stop)
    d_v = 0.35
    t_v = np.linspace(0, d_v, int(sr * d_v), dtype=np.float32)
    freq_v = 650.0 * np.exp(-t_v * 7.5) + 45.0
    v_sig = (
        np.sin(2.0 * math.pi * freq_v * t_v) * 0.35
        + np.sin(4.0 * math.pi * freq_v * t_v) * 0.15
    ) * np.clip(1.0 - t_v / d_v, 0.0, 1.0)
    vinyl_p = work_dir / "sfx_vinyl.wav"
    _write_stereo_wav(vinyl_p, v_sig, sr)

    return {"whoosh": whoosh_p, "pop": pop_p, "vinyl": vinyl_p}


def _make_studio_card_bg(path: Path, width: int = 1920, height: int = 1080, accent_rgb: tuple[int, int, int] = (108, 140, 255)) -> Path:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = 20.0
    img[:, :, 1] = 14.0
    img[:, :, 2] = 11.0
    g1 = np.exp(-((x - 0.22) ** 2 / 0.16 + (y - 0.2) ** 2 / 0.2))
    g2 = np.exp(-((x - 0.8) ** 2 / 0.18 + (y - 0.82) ** 2 / 0.2))
    img[:, :, 0] += g1 * accent_rgb[2] * 0.22 + g2 * 38.0
    img[:, :, 1] += g1 * accent_rgb[1] * 0.20 + g2 * 38.0
    img[:, :, 2] += g1 * accent_rgb[0] * 0.20 + g2 * 12.0
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8))
    return path


def _write_simple_ass(ass_path: Path, dialogues: list[str], font_name: str = "Arial") -> Path:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title, {font_name}, 74, &H00F9F2EE, &H00FFFFFF, &H00140E0B, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 3, 2, 5, 110, 110, 40, 1
Style: SubTitle, {font_name}, 36, &H00BFD42D, &H00FFFFFF, &H00140E0B, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 2, 1, 5, 110, 110, 40, 1
Style: RankNum, {font_name}, 118, &H0000D7FF, &H00FFFFFF, &H00140E0B, &H90000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 4, 2, 5, 100, 100, 40, 1
Style: Caption16x9, {font_name}, 54, &H00FFFFFF, &H0000D7FF, &H000A0A0A, &H85000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 4, 2, 2, 110, 110, 62, 1
Style: CornerBadge, {font_name}, 30, &H00FFFFFF, &H00FFFFFF, &H00140E0B, &HA0000000, 1, 0, 0, 0, 100, 100, 0, 0, 3, 10, 0, 9, 40, 42, 34, 1
Style: BrandCorner16x9, {font_name}, 28, &H2200D7FF, &H00FFFFFF, &H00140E0B, &HA0000000, 1, 0, 0, 0, 100, 100, 0, 0, 3, 10, 0, 1, 42, 42, 30, 1
Style: ColdBadge, {font_name}, 38, &H000A0E14, &H000A0E14, &H0000D7FF, &H0000D7FF, 1, 0, 0, 0, 100, 100, 0, 0, 3, 14, 0, 7, 48, 48, 42, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_path.write_text(header + "\n".join(dialogues) + "\n", encoding="utf-8")
    return ass_path


def _render_card_segment(
    bg_png: Path,
    out_mp4: Path,
    duration_s: float,
    ass_dialogues: list[str],
    sfx_wav: Path,
    work_dir: Path,
    name_tag: str,
    voice_audio_path: Path | None = None,
) -> float:
    dur = max(1.2, float(duration_s))
    ass_fp = _write_simple_ass(work_dir / f"{name_tag}.ass", ass_dialogues)
    use_ass = _ffmpeg_has_ass_filter()
    ass_part = f",ass='{_escape_ffmpeg_filter_path(ass_fp)}'" if use_ass else ""

    cmd = [
        FFMPEG_BIN, "-y",
        "-loop", "1", "-t", f"{dur:.2f}", "-i", str(bg_png),
        "-i", str(sfx_wav),
    ]
    if voice_audio_path and voice_audio_path.exists():
        cmd.extend(["-i", str(voice_audio_path)])
        afilter = (
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.5[sfx];"
            "[2:a]aformat=sample_rates=44100:channel_layouts=stereo,loudnorm=I=-14:LRA=7:TP=-1.5[vox];"
            f"[sfx][vox]amix=inputs=2:duration=longest,apad=whole_dur={dur:.2f},atrim=0:{dur:.2f}[aout]"
        )
    else:
        afilter = (
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.75,apad=whole_dur={dur:.2f},atrim=0:{dur:.2f}[aout]"
        )

    cmd.extend([
        "-filter_complex",
        f"[0:v]fps=30,scale=1920:1080{ass_part},fade=t=in:st=0:d=0.18,fade=t=out:st={max(0.2, dur - 0.18):.2f}:d=0.18[vout];{afilter}",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(out_mp4),
    ])
    run_cmd(cmd)
    return float(probe_video(out_mp4)["duration"])


def collect_compilation_candidates(
    jobs: list[dict[str, Any]], max_clips: int = 8
) -> list[dict[str, Any]]:
    """Collect and rank the strongest clips across multiple jobs, ordered as a countdown (#N -> #1)."""
    raw: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for job in jobs:
        if job.get("status") != "completed":
            continue
        job_id = job["job_id"]
        src_mode = job.get("source_mode", "youtube")
        src_val = str(job.get("source_value") or "")
        src_video_path = str(job.get("video_path") or "")
        for c in job.get("clips") or []:
            start = float(c.get("start") or 0.0)
            end = float(c.get("end") or 0.0)
            if end - start < 4.0:
                continue
            key = f"{src_val}:{int(start // 5)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rendered_mp4 = None
            for cand in (
                Path(job.get("output_dir") or (get_output_dir() / job_id)) / f"{c['clip_id']}.mp4",
                BASE_DIR / "outputs" / job_id / f"{c['clip_id']}.mp4",
            ):
                if cand.exists() and cand.is_file():
                    rendered_mp4 = str(cand)
                    break

            ev = c.get("evaluation") or {}
            raw.append({
                "job_id": job_id,
                "clip_id": c["clip_id"],
                "source_mode": src_mode,
                "source_value": src_val,
                "source_video_path": src_video_path,
                "rendered_short_mp4": rendered_mp4,
                "start": start,
                "end": end,
                "score": float(c.get("score") or ev.get("overall_score") or 75.0),
                "hook_title": str(c.get("hook_title") or "HIGHLIGHT MOMENT").strip(),
                "text_preview": str(ev.get("text_preview") or "").strip(),
            })

    raw.sort(key=lambda x: x["score"], reverse=True)
    picked = raw[: max(2, min(15, int(max_clips)))]
    # Reverse into Countdown order (#N ... #2 -> #1 Best Moment at the climax!)
    countdown = list(reversed(picked))
    total = len(countdown)
    for idx, item in enumerate(countdown):
        item["rank"] = total - idx
    return countdown


def _resolve_landscape_source(item: dict[str, Any], work_dir: Path, idx: int, log_cb: Callable[[str], None] | None = None) -> tuple[Path, float, bool]:
    """
    Get a crisp 16:9 landscape clip file whenever possible (local file or partial yt-dlp 1080p slice).
    Returns (media_path, seek_offset, is_portrait_fallback).
    """
    vp = item.get("source_video_path")
    if vp and Path(vp).exists():
        return Path(vp), float(item["start"]), False

    if item.get("source_mode") == "youtube" and str(item.get("source_value") or "").startswith("http"):
        try:
            slice_p = work_dir / f"comp_src_{idx:02d}.mp4"
            got = download_youtube_section(
                item["source_value"],
                max(0.0, float(item["start"]) - 0.5),
                float(item["end"]) + 0.5,
                slice_p,
                log_cb=log_cb,
            )
            return got, 0.5 if float(item["start"]) >= 0.5 else 0.0, False
        except Exception:
            pass

    short_mp4 = item.get("rendered_short_mp4")
    if short_mp4 and Path(short_mp4).exists():
        return Path(short_mp4), 0.0, True

    raise FileNotFoundError(f"No media found for {item['job_id']}/{item['clip_id']}")


def _build_16x9_clip_ass(
    ass_path: Path,
    duration_s: float,
    text_preview: str,
    watermark_text: str,
    rank_label: str = "",
    channel_watermark: str = "",
) -> Path:
    font = _pick_caption_font(needs_devanagari=False)
    lines: list[str] = []
    if channel_watermark.strip():
        lines.append(
            f"Dialogue: 2,0:00:00.00,{_format_ass_time(duration_s)},BrandCorner16x9,,0,0,0,,{_escape_ass_text(channel_watermark.strip())}"
        )
    if watermark_text.strip():
        lines.append(
            f"Dialogue: 2,0:00:00.00,{_format_ass_time(duration_s)},CornerBadge,,0,0,0,,{_escape_ass_text(watermark_text.strip())}"
        )
    if rank_label:
        lines.append(
            f"Dialogue: 2,0:00:00.00,{_format_ass_time(min(duration_s, 4.0))},ColdBadge,,0,0,0,,{{\\fad(120,250)}}{_escape_ass_text(rank_label)}"
        )

    words = [w for w in text_preview.split() if w]
    if words and duration_s > 2.0:
        # Spread preview words evenly into snappy 6-word subtitle chunks across start of clip
        chunks = [" ".join(words[i : i + 6]) for i in range(0, min(len(words), 36), 6)]
        span = min(duration_s - 0.4, len(chunks) * 2.4)
        step = span / max(1, len(chunks))
        for i, ch in enumerate(chunks):
            t0 = 0.2 + i * step
            t1 = min(duration_s - 0.15, t0 + step)
            lines.append(
                f"Dialogue: 1,{_format_ass_time(t0)},{_format_ass_time(t1)},Caption16x9,,0,0,0,,{_escape_ass_text(ch)}"
            )
    return _write_simple_ass(ass_path, lines, font_name=font)


def _render_clip_segment_16x9(
    src_path: Path,
    seek_s: float,
    dur_s: float,
    out_mp4: Path,
    is_portrait_fallback: bool,
    sfx_whoosh: Path,
    watermark_text: str,
    text_preview: str,
    vfx_enabled: bool,
    transition_style: str,
    work_dir: Path,
    tag: str,
    rank_badge: str = "",
    channel_watermark: str = "",
) -> float:
    dur = max(2.0, float(dur_s))
    ass_fp = _build_16x9_clip_ass(
        work_dir / f"{tag}.ass",
        dur,
        text_preview,
        watermark_text,
        rank_label=rank_badge,
        channel_watermark=channel_watermark,
    )
    ass_part = f",ass='{_escape_ffmpeg_filter_path(ass_fp)}'" if _ffmpeg_has_ass_filter() else ""
    meta = probe_video(src_path)
    has_audio = bool(meta.get("has_audio", True))

    if is_portrait_fallback:
        base_v = (
            "[0:v]fps=30,split[bg][fg];"
            "[bg]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=25:5,eq=brightness=-0.18[bgb];"
            "[fg]scale=-2:1080[fgs];"
            "[bgb][fgs]overlay=(W-w)/2:0"
        )
    else:
        if vfx_enabled:
            # Dynamic punch-in motion + midpoint energy contrast boost
            mid = max(1.0, dur * 0.48)
            base_v = (
                "[0:v]fps=30,"
                "scale=w='trunc(1920*(1+0.055*min(1,t/6))/2)*2':h='trunc(1080*(1+0.055*min(1,t/6))/2)*2':eval=frame,"
                "crop=1920:1080:(iw-1920)/2:(ih-1080)/2,"
                f"eq=contrast=1.10:saturation=1.16:enable='between(t,{mid:.2f},{mid+0.8:.2f})'"
            )
        else:
            base_v = "[0:v]fps=30,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"

    fade_out_color = "fade=t=out:st={:.2f}:d=0.16{}".format(
        max(0.2, dur - 0.16),
        ":c=white" if transition_style == "flash_cut" else "",
    )
    vfilter = f"{base_v}{ass_part},fade=t=in:st=0:d=0.16,{fade_out_color}[vout]"

    cmd = [FFMPEG_BIN, "-y", "-ss", f"{max(0.0, seek_s):.2f}", "-t", f"{dur:.2f}", "-i", str(src_path)]
    if has_audio:
        cmd.extend(["-i", str(sfx_whoosh)])
        afilter = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,loudnorm=I=-14:LRA=7:TP=-1.5[main_a];"
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.45[sfx_a];"
            f"[main_a][sfx_a]amix=inputs=2:duration=first:normalize=0,apad=whole_dur={dur:.2f},atrim=0:{dur:.2f}[aout]"
        )
    else:
        cmd.extend(["-i", str(sfx_whoosh)])
        afilter = f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.45,apad=whole_dur={dur:.2f},atrim=0:{dur:.2f}[aout]"

    cmd.extend([
        "-filter_complex", f"{vfilter};{afilter}",
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(out_mp4),
    ])
    run_cmd(cmd)
    return float(probe_video(out_mp4)["duration"])


def build_weekly_compilation(
    comp_id: str,
    title: str,
    jobs: list[dict[str, Any]],
    out_dir: Path,
    work_dir: Path,
    max_clips: int = 8,
    watermark_text: str = "Credit: @RakaZoneGaming",
    channel_watermark: str = "@RAKAGTAVI",
    creator_tag: str = "@RakaZoneGaming",
    required_hashtags: str = "#RakaZoneGaming #GTA6 #Gaming",
    vfx_enabled: bool = True,
    transition_style: str = "slide_wipe",
    intro_audio_path: Path | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build a retention-edited 1920x1080 weekly compilation (Cold Open -> Countdown Cards -> 16:9 VFX Clips -> Chapters)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_compilation_candidates(jobs, max_clips=max_clips)
    if not candidates:
        raise ValueError("No completed clips found across the selected jobs.")

    if log_cb:
        log_cb(f"Selected top {len(candidates)} moments across {len(jobs)} streams (Countdown #{len(candidates)} -> #1).")

    sfx = _generate_sfx_bank(work_dir)
    bg_png = _make_studio_card_bg(work_dir / "card_bg.png")

    seg_files: list[Path] = []
    chapters: list[dict[str, Any]] = []
    cur_t = 0.0

    # 1. Cold Open (4.5s peek of #1 highest-scoring clip!)
    best_item = candidates[-1]
    if log_cb:
        log_cb(f"Rendering Cold Open teaser from #1 moment ({best_item['hook_title']})...")
    best_src, best_seek, best_port = _resolve_landscape_source(best_item, work_dir, 99, log_cb=log_cb)
    cold_dur = min(4.5, max(2.5, (best_item["end"] - best_item["start"]) * 0.5))
    cold_mp4 = work_dir / "seg_00_cold.mp4"
    actual_cold = _render_clip_segment_16x9(
        src_path=best_src,
        seek_s=best_seek + 1.0 if (best_item["end"] - best_item["start"]) > 6.5 else best_seek,
        dur_s=cold_dur,
        out_mp4=cold_mp4,
        is_portrait_fallback=best_port,
        sfx_whoosh=sfx["vinyl"],
        watermark_text=watermark_text,
        text_preview="",
        vfx_enabled=True,
        transition_style="flash_cut",
        work_dir=work_dir,
        tag="cold",
        rank_badge="COMING UP...",
        channel_watermark=channel_watermark,
    )
    seg_files.append(cold_mp4)
    chapters.append({"time": _fmt_chapter_ts(cur_t), "label": "Cold Open"})
    cur_t += actual_cold

    # 2. Intro Title Card (2.4s, or longer if user recorded intro voiceover)
    intro_dur = 2.4
    if intro_audio_path and intro_audio_path.exists():
        try:
            ia_dur = float(probe_video(intro_audio_path)["duration"])
            if ia_dur > 2.0:
                intro_dur = min(30.0, ia_dur + 0.4)
        except Exception:
            pass

    intro_mp4 = work_dir / "seg_01_intro.mp4"
    credit_sub = f"Credit: {creator_tag}" if creator_tag else (watermark_text or "Weekly Highlights")
    actual_intro = _render_card_segment(
        bg_png=bg_png,
        out_mp4=intro_mp4,
        duration_s=intro_dur,
        ass_dialogues=[
            rf"Dialogue: 1,0:00:00.00,{_format_ass_time(intro_dur)},Title,,0,0,0,,{{\pos(960,475)\fad(180,180)}}{_escape_ass_text(title)}",
            rf"Dialogue: 1,0:00:00.12,{_format_ass_time(intro_dur)},SubTitle,,0,0,0,,{{\pos(960,590)\fad(180,180)}}{_escape_ass_text(credit_sub)}",
        ],
        sfx_wav=sfx["whoosh"],
        work_dir=work_dir,
        name_tag="intro",
        voice_audio_path=intro_audio_path,
    )
    seg_files.append(intro_mp4)
    chapters.append({"time": _fmt_chapter_ts(cur_t), "label": "Intro"})
    cur_t += actual_intro

    # 3. Countdown Segments (#N -> #1)
    for idx, item in enumerate(candidates, 1):
        rank = item["rank"]
        hook = item["hook_title"]
        if log_cb:
            log_cb(f"Rendering highlight #{rank} ({idx}/{len(candidates)}): {hook}...")

        # Brief 1.4s chapter card
        card_mp4 = work_dir / f"seg_{idx:02d}_card.mp4"
        card_dur = _render_card_segment(
            bg_png=bg_png,
            out_mp4=card_mp4,
            duration_s=1.45,
            ass_dialogues=[
                rf"Dialogue: 1,0:00:00.00,0:00:01.45,RankNum,,0,0,0,,{{\pos(960,435)\fad(100,120)}}#{rank}",
                rf"Dialogue: 1,0:00:00.08,0:00:01.45,SubTitle,,0,0,0,,{{\pos(960,595)\fad(120,120)}}{_escape_ass_text(hook[:56])}",
            ],
            sfx_wav=sfx["pop"],
            work_dir=work_dir,
            name_tag=f"card_{idx:02d}",
        )
        chapters.append({"time": _fmt_chapter_ts(cur_t), "label": f"#{rank} - {hook[:48]}"})
        seg_files.append(card_mp4)
        cur_t += card_dur

        # Main 16:9 Clip
        src_p, seek_s, is_port = (
            (best_src, best_seek, best_port)
            if item is best_item
            else _resolve_landscape_source(item, work_dir, idx, log_cb=log_cb)
        )
        clip_dur = max(4.0, float(item["end"]) - float(item["start"]))
        clip_mp4 = work_dir / f"seg_{idx:02d}_clip.mp4"
        actual_clip = _render_clip_segment_16x9(
            src_path=src_p,
            seek_s=seek_s,
            dur_s=clip_dur,
            out_mp4=clip_mp4,
            is_portrait_fallback=is_port,
            sfx_whoosh=sfx["whoosh"],
            watermark_text=watermark_text,
            text_preview=item["text_preview"],
            vfx_enabled=vfx_enabled,
            transition_style=transition_style,
            work_dir=work_dir,
            tag=f"clip_{idx:02d}",
            rank_badge=f"#{rank}" if rank <= 3 else "",
            channel_watermark=channel_watermark,
        )
        seg_files.append(clip_mp4)
        cur_t += actual_clip

    # 4. Fast lossless concat of normalized segments
    concat_txt = work_dir / "concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in seg_files) + "\n",
        encoding="utf-8",
    )
    out_mp4 = out_dir / "compilation_1080p.mp4"
    if log_cb:
        log_cb("Stitching final 1920x1080 compilation MP4...")
    run_cmd([
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c", "copy", "-movflags", "+faststart", str(out_mp4),
    ])

    chap_lines = "\n".join(f"{c['time']} {c['label']}" for c in chapters)
    clean_tag = creator_tag.strip()
    if clean_tag and not clean_tag.startswith("@"):
        clean_tag = "@" + clean_tag
    desc_lines = [
        f"{title}",
        "",
        f"Credit: {clean_tag}" if clean_tag else "",
        "",
        "Chapters:",
        chap_lines,
        "",
        required_hashtags.strip(),
    ]
    description = "\n".join(l for l in desc_lines if l is not None).strip()

    return {
        "compilation_id": comp_id,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(candidates),
        "duration_sec": round(cur_t, 1),
        "file_size_mb": round(out_mp4.stat().st_size / (1024 * 1024), 2),
        "output_url": f"/outputs/compilations/{comp_id}/{out_mp4.name}",
        "output_path": str(out_mp4),
        "chapters": chapters,
        "youtube_description": description,
        "creator_tag": clean_tag,
        "watermark_text": watermark_text,
    }
