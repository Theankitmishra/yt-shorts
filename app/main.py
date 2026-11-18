from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.analyzer import analyze_visual_stream, compute_audio_profile
from app.campaigns import (
    add_campaign_submission,
    delete_campaign,
    generate_bounty_post_pack,
    load_campaigns,
    update_submission_views,
    upsert_campaign,
)
from app.uploader import (
    build_youtube_auth_url,
    clear_youtube_tokens,
    exchange_youtube_code,
    get_youtube_status,
    save_youtube_client_credentials,
    upload_short_to_youtube,
)
from app.compiler import build_weekly_compilation
from app.monitor import (
    count_jobs_started_today,
    get_autopilot_config,
    get_monitor_snapshot,
    load_monitor_state,
    scan_creator,
    toggle_manual_clipped,
    update_autopilot_config,
    upsert_creator,
)
from app.config import (
    BASE_DIR,
    DATA_DIR,
    FFMPEG_BIN,
    JOBS_DIR,
    STATIC_DIR,
    get_groq_api_key,
    get_output_dir,
    get_storage_status,
    get_upload_dir,
    get_work_dir,
    load_settings,
    save_settings,
    save_youtube_cookies_text,
)
from app.renderer import _ffmpeg_has_ass_filter, render_short_clip
from app.rsi_engine import (
    detect_available_llm,
    learn_from_human_feedback,
    load_rsi_memory,
    reset_rsi_memory,
    run_rsi_pipeline,
)
from app.transcriber import (
    capture_live_segment,
    download_youtube_audio,
    download_youtube_section,
    download_youtube_video,
    extract_wav,
    fetch_youtube_info,
    probe_audio_duration,
    probe_video,
    transcribe_audio,
    transcribe_clip_precision,
)

app = FastAPI(title="RSI Shorts Studio", version="1.1.0")
executor = ThreadPoolExecutor(max_workers=2)


def _job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _load_job(job_id: str) -> dict[str, Any]:
    fp = _job_file(job_id)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(fp.read_text(encoding="utf-8"))


def _save_job(job: dict[str, Any]) -> None:
    _job_file(job["job_id"]).write_text(json.dumps(job, indent=2), encoding="utf-8")


def _append_log(job: dict[str, Any], msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    job["logs"].append(f"[{ts}] {msg}")
    job["logs"] = job["logs"][-150:]
    _save_job(job)


def _cleanup_job_temp_media(work_dir: Path, upload_path: Path | None = None) -> float:
    """Remove bulky intermediate audio/video files while keeping JSON metadata."""
    freed = 0
    for p in list(work_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".wav", ".opus", ".part", ".ytdl"}:
            try:
                freed += p.stat().st_size
                p.unlink()
            except OSError:
                pass
    if upload_path and upload_path.exists():
        try:
            freed += upload_path.stat().st_size
            upload_path.unlink()
        except OSError:
            pass
    return round(freed / (1024 * 1024), 1)


def _process_job(job_id: str) -> None:
    job = _load_job(job_id)
    work_dir = get_work_dir() / job_id
    out_dir = get_output_dir() / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        _append_log(job, msg)

    try:
        job["status"] = "running"
        _save_job(job)

        source_mode = job["source_mode"]
        youtube_mode = job.get("youtube_mode", "audio_first")
        num_clips = int(job.get("num_clips", 3))
        rsi_iterations = int(job.get("rsi_iterations", 3))
        layout_pref = job.get("layout_preference", "auto")
        llm_pref = job.get("llm_provider", "auto_free")
        burn_captions = bool(job.get("burn_captions", True))
        sub_lang = job.get("subtitle_language", "auto")
        sub_script = job.get("subtitle_script", "roman")
        video_title = str(job.get("source_label") or "")

        yt_info: dict[str, Any] = {}
        is_live_stream = False
        if source_mode == "youtube":
            log("Checking stream / video metadata...")
            yt_info = fetch_youtube_info(job["source_value"])
            is_live_stream = bool(yt_info.get("is_live"))
            if yt_info.get("title"):
                job["source_label"] = yt_info["title"]
                _save_job(job)

        if source_mode == "youtube" and youtube_mode == "audio_first" and not is_live_stream:
            # --- ULTRA-LIGHT AUDIO-FIRST + PARTIAL 1080p SLICE PIPELINE ---
            url = job["source_value"]
            duration = float(yt_info.get("duration") or 0.0)

            audio_path = download_youtube_audio(url, work_dir, log_cb=log)
            if duration <= 0:
                duration = probe_audio_duration(audio_path)

            wav_path = work_dir / "audio_16k.wav"
            log("Extracting 16kHz mono audio & computing RMS energy profile...")
            extract_wav(audio_path, wav_path)
            audio_profile = compute_audio_profile(wav_path)
            log(f"Audio profile ready ({len(audio_profile['silences'])} silence gaps, duration={duration:.0f}s).")

            transcript = transcribe_audio(
                wav_path,
                model_size=job.get("whisper_model", "base"),
                fallback_json3=None,
                log_cb=log,
                audio_profile=audio_profile,
                subtitle_language=sub_lang,
            )
            segments = transcript["segments"]
            job["transcript_engine"] = transcript["engine"]
            job["segment_count"] = len(segments)
            (work_dir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
            log(f"Transcribed {len(segments)} segments via {transcript['engine']}.")

            # Stage 1 RSI: pick top candidate windows from transcript + audio energy
            log("RSI Stage 1: Selecting top candidate clip windows from transcript & audio energy...")
            stage1 = run_rsi_pipeline(
                segments=segments,
                audio=audio_profile,
                visual={"frames": [], "scene_cuts": [], "dominant_style": "unknown"},
                video_duration=duration,
                num_clips=num_clips,
                iterations=1,
                layout_preference=layout_pref,
                llm_provider=llm_pref,
                log_cb=log,
            )
            seed_clips = stage1["clips"]

            # Fetch ONLY those short 1080p slices (+6s before, +12s after for RSI refinement)
            slice_map: dict[str, dict[str, Any]] = {}
            merged_frames: list[dict[str, Any]] = []
            merged_cuts: list[float] = []
            meta = {"width": 1920, "height": 1080, "fps": 30.0, "duration": duration}

            for c in seed_clips:
                cid = c["clip_id"]
                s_start = max(0.0, float(c["start"]) - 6.0)
                s_end = min(duration, float(c["end"]) + 12.0)
                slice_dest = work_dir / f"slice_{cid}.mp4"
                slice_path = download_youtube_section(url, s_start, s_end, slice_dest, log_cb=log)
                s_meta = probe_video(slice_path)
                meta["width"] = s_meta["width"]
                meta["height"] = s_meta["height"]
                meta["fps"] = s_meta["fps"]

                log(f"Scanning OpenCV visual framing on {cid} slice...")
                v_slice = analyze_visual_stream(slice_path, sample_fps=1.0, time_offset=s_start)
                merged_frames.extend(v_slice.get("frames", []))
                merged_cuts.extend(v_slice.get("scene_cuts", []))
                slice_map[cid] = {
                    "path": slice_path,
                    "offset": s_start,
                    "slice_end": s_start + s_meta["duration"],
                    "meta": s_meta,
                }

            merged_frames.sort(key=lambda f: f["t"])
            visual = {
                "frames": merged_frames,
                "scene_cuts": sorted(merged_cuts),
                "dominant_style": "mixed_visual",
            }
            (work_dir / "visual.json").write_text(json.dumps(visual, indent=2), encoding="utf-8")
            job["video_meta"] = meta

            # Stage 2 RSI: Full multi-round self-improvement with visual framing + boundary refinement
            rsi_res = run_rsi_pipeline(
                segments=segments,
                audio=audio_profile,
                visual=visual,
                video_duration=duration,
                num_clips=num_clips,
                iterations=rsi_iterations,
                layout_preference=layout_pref,
                llm_provider=llm_pref,
                log_cb=log,
                initial_clips=seed_clips,
            )
            job["llm_provider_used"] = rsi_res["llm_provider"]
            job["memory_generation"] = rsi_res["memory_generation"]
            job["rsi_traces"] = rsi_res["rsi_traces"]

            rendered_clips: list[dict[str, Any]] = []
            watermark_text = str(job.get("watermark_text") or "").strip()
            channel_watermark = str(job.get("channel_watermark") or "").strip()
            creator_tag = str(job.get("creator_tag") or "").strip()
            req_tags = str(job.get("required_hashtags") or "").strip()
            max_dur = int(job.get("target_max_duration") or 55)
            for idx, clip in enumerate(rsi_res["clips"], 1):
                s_info = slice_map[clip["clip_id"]]
                # Keep refined boundaries safely inside downloaded slice & optional campaign duration cap
                clip["start"] = round(max(s_info["offset"], float(clip["start"])), 2)
                clip["end"] = round(min(s_info["slice_end"], clip["start"] + max_dur, float(clip["end"])), 2)
                if watermark_text:
                    clip["watermark_text"] = watermark_text
                if channel_watermark:
                    clip["channel_watermark"] = channel_watermark

                clip_segments = segments
                if burn_captions:
                    log(f"Precision subtitle pass on {clip['clip_id']} ({sub_lang}/{sub_script})...")
                    prec = transcribe_clip_precision(
                        media_path=s_info["path"],
                        clip_start=float(clip["start"]),
                        clip_end=float(clip["end"]),
                        source_offset=s_info["offset"],
                        model_size=job.get("whisper_model", "small"),
                        subtitle_language=sub_lang,
                        subtitle_script=sub_script,
                        video_title=str(job.get("source_label") or video_title),
                        llm_provider=llm_pref,
                        log_cb=log,
                    )
                    if prec.get("segments"):
                        clip_segments = prec["segments"]
                        new_txt = " ".join(s.get("text", "") for s in clip_segments).strip()
                        if new_txt and clip.get("evaluation"):
                            clip["evaluation"]["text_preview"] = (
                                new_txt[:280] + ("..." if len(new_txt) > 280 else "")
                            )
                        sug_hook = clip_segments[0].pop("_suggested_hook", None)
                        if sug_hook:
                            clip["hook_title"] = sug_hook

                log(
                    f"Rendering {clip['clip_id']} ({idx}/{len(rsi_res['clips'])}) "
                    f"[{clip['start']:.1f}s - {clip['end']:.1f}s] layout={clip['layout']}..."
                )
                out_file = out_dir / f"{clip['clip_id']}.mp4"
                render_short_clip(
                    source_video=s_info["path"],
                    output_path=out_file,
                    segments=clip_segments,
                    visual=visual,
                    clip=clip,
                    src_w=s_info["meta"]["width"],
                    src_h=s_info["meta"]["height"],
                    work_dir=work_dir,
                    burn_captions=burn_captions,
                    source_offset=s_info["offset"],
                )
                clip["output_url"] = f"/outputs/{job_id}/{out_file.name}"
                clip["file_size_mb"] = round(out_file.stat().st_size / (1024 * 1024), 2)
                clip["post_pack"] = generate_bounty_post_pack(
                    clip,
                    creator_tag=creator_tag,
                    required_hashtags=req_tags,
                    watermark_text=watermark_text,
                    burn_captions=burn_captions,
                )
                rendered_clips.append(clip)

        else:
            # --- FULL VIDEO / LIVE BUFFER PIPELINE (Local File, Upload, Full VOD, or Active Live) ---
            fallback_json3: Path | None = None
            if source_mode == "youtube":
                if is_live_stream:
                    video_path = capture_live_segment(
                        job["source_value"], work_dir, duration_sec=600, log_cb=log
                    )
                else:
                    video_path, fallback_json3 = download_youtube_video(job["source_value"], work_dir, log_cb=log)
            else:
                video_path = Path(job["source_value"])
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

            job["video_path"] = str(video_path)
            meta = probe_video(video_path)
            job["video_meta"] = meta
            log(f"Video ready: {meta['width']}x{meta['height']} @ {meta['fps']}fps, {meta['duration']:.1f}s")

            wav_path = work_dir / "audio_16k.wav"
            log("Extracting 16kHz mono audio & computing RMS energy profile...")
            extract_wav(video_path, wav_path)
            audio_profile = compute_audio_profile(wav_path)
            log(f"Audio profile ready ({len(audio_profile['silences'])} silence gaps detected).")

            transcript = transcribe_audio(
                wav_path,
                model_size=job.get("whisper_model", "base"),
                fallback_json3=fallback_json3,
                log_cb=log,
                audio_profile=audio_profile,
                subtitle_language=sub_lang,
            )
            segments = transcript["segments"]
            job["transcript_engine"] = transcript["engine"]
            job["segment_count"] = len(segments)
            (work_dir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
            log(f"Transcribed {len(segments)} segments via {transcript['engine']}.")

            log("Scanning video frames (face tracking, screen-share UI density, scene cuts)...")
            visual = analyze_visual_stream(video_path, sample_fps=1.0, log_cb=log)
            (work_dir / "visual.json").write_text(json.dumps(visual, indent=2), encoding="utf-8")
            log(
                f"Visual style detected: {visual.get('dominant_style')} "
                f"({len(visual.get('scene_cuts', []))} scene cuts)."
            )

            rsi_res = run_rsi_pipeline(
                segments=segments,
                audio=audio_profile,
                visual=visual,
                video_duration=meta["duration"],
                num_clips=num_clips,
                iterations=rsi_iterations,
                layout_preference=layout_pref,
                llm_provider=llm_pref,
                log_cb=log,
            )
            job["llm_provider_used"] = rsi_res["llm_provider"]
            job["memory_generation"] = rsi_res["memory_generation"]
            job["rsi_traces"] = rsi_res["rsi_traces"]

            rendered_clips = []
            watermark_text = str(job.get("watermark_text") or "").strip()
            channel_watermark = str(job.get("channel_watermark") or "").strip()
            creator_tag = str(job.get("creator_tag") or "").strip()
            req_tags = str(job.get("required_hashtags") or "").strip()
            max_dur = int(job.get("target_max_duration") or 55)
            for idx, clip in enumerate(rsi_res["clips"], 1):
                clip["end"] = round(min(float(clip["end"]), float(clip["start"]) + max_dur), 2)
                if watermark_text:
                    clip["watermark_text"] = watermark_text
                if channel_watermark:
                    clip["channel_watermark"] = channel_watermark
                clip_segments = segments
                if burn_captions:
                    log(f"Precision subtitle pass on {clip['clip_id']} ({sub_lang}/{sub_script})...")
                    prec = transcribe_clip_precision(
                        media_path=video_path,
                        clip_start=float(clip["start"]),
                        clip_end=float(clip["end"]),
                        source_offset=0.0,
                        model_size=job.get("whisper_model", "small"),
                        subtitle_language=sub_lang,
                        subtitle_script=sub_script,
                        video_title=str(job.get("source_label") or video_title),
                        llm_provider=llm_pref,
                        log_cb=log,
                    )
                    if prec.get("segments"):
                        clip_segments = prec["segments"]
                        new_txt = " ".join(s.get("text", "") for s in clip_segments).strip()
                        if new_txt and clip.get("evaluation"):
                            clip["evaluation"]["text_preview"] = (
                                new_txt[:280] + ("..." if len(new_txt) > 280 else "")
                            )
                        sug_hook = clip_segments[0].pop("_suggested_hook", None)
                        if sug_hook:
                            clip["hook_title"] = sug_hook

                log(
                    f"Rendering {clip['clip_id']} ({idx}/{len(rsi_res['clips'])}) "
                    f"[{clip['start']:.1f}s - {clip['end']:.1f}s] layout={clip['layout']}..."
                )
                out_file = out_dir / f"{clip['clip_id']}.mp4"
                render_short_clip(
                    source_video=video_path,
                    output_path=out_file,
                    segments=clip_segments,
                    visual=visual,
                    clip=clip,
                    src_w=meta["width"],
                    src_h=meta["height"],
                    work_dir=work_dir,
                    burn_captions=burn_captions,
                )
                clip["output_url"] = f"/outputs/{job_id}/{out_file.name}"
                clip["file_size_mb"] = round(out_file.stat().st_size / (1024 * 1024), 2)
                clip["post_pack"] = generate_bounty_post_pack(
                    clip,
                    creator_tag=creator_tag,
                    required_hashtags=req_tags,
                    watermark_text=watermark_text,
                    burn_captions=burn_captions,
                )
                rendered_clips.append(clip)

        job["clips"] = rendered_clips
        job["output_dir"] = str(out_dir)

        if load_settings().get("auto_cleanup_work", True):
            up_path = Path(job["source_value"]) if job.get("is_uploaded_copy") else None
            freed_mb = _cleanup_job_temp_media(work_dir, upload_path=up_path)
            if freed_mb > 0:
                log(f"Auto-cleaned {freed_mb} MB of temporary work media to save disk space.")

        if job.get("auto_upload_youtube"):
            yt_stat = get_youtube_status()
            if not yt_stat.get("connected"):
                log("Auto-upload skipped: YouTube channel not connected in Cloud, Keys & Storage.")
            else:
                yt_priv = str(job.get("youtube_privacy") or "private")
                for clip in rendered_clips:
                    try:
                        log(f"Auto-uploading {clip['clip_id']} to YouTube Shorts ({yt_priv})...")
                        res = _upload_clip_to_youtube_record(job, clip, privacy_status=yt_priv)
                        log(f"Uploaded {clip['clip_id']} -> {res['url']} ({res['privacy_status']})")
                    except Exception as up_exc:
                        clip["youtube_upload_error"] = str(up_exc)
                        log(f"Auto-upload error on {clip['clip_id']}: {up_exc}")

        job["status"] = "completed"
        log("All clips rendered and ready for review!")
        _save_job(job)

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{exc}\n{traceback.format_exc()}"
        log(f"ERROR: {exc}")
        _save_job(job)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "monitor.html").read_text(encoding="utf-8"))


@app.get("/outputs/compilations/{comp_id}/{filename}")
def serve_compilation_output(comp_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    candidates = [
        get_output_dir() / "compilations" / comp_id / safe_name,
        BASE_DIR / "outputs" / "compilations" / comp_id / safe_name,
    ]
    for fp in candidates:
        if fp.exists() and fp.is_file():
            return FileResponse(fp, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Compilation video not found")


@app.get("/outputs/{job_id}/{filename}")
def serve_output(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    candidates = [
        get_output_dir() / job_id / safe_name,
        BASE_DIR / "outputs" / job_id / safe_name,
    ]
    for fp in candidates:
        if fp.exists() and fp.is_file():
            return FileResponse(fp, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Output clip not found")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ffmpeg": shutil.which(FFMPEG_BIN) is not None or Path(FFMPEG_BIN).exists(),
        "ffmpeg_ass": _ffmpeg_has_ass_filter(),
        "yt_dlp": shutil.which("yt-dlp") is not None,
        "detected_llm": detect_available_llm("auto_free"),
        "transcription_mode": "groq_cloud" if get_groq_api_key() else "local",
        "youtube": get_youtube_status(),
        "memory": load_rsi_memory(),
        "storage": get_storage_status(),
    }


@app.get("/api/storage")
def api_get_storage() -> dict[str, Any]:
    return get_storage_status()


class StorageBody(BaseModel):
    storage_dir: str
    auto_cleanup_work: bool = True
    yt_dlp_cookies_file: str = ""
    yt_dlp_cookies_text: str | None = None
    groq_api_key: str | None = None


@app.post("/api/storage")
def api_update_storage(body: StorageBody) -> dict[str, Any]:
    try:
        updates: dict[str, Any] = {
            "storage_dir": body.storage_dir.strip(),
            "auto_cleanup_work": body.auto_cleanup_work,
            "yt_dlp_cookies_file": body.yt_dlp_cookies_file.strip(),
        }
        if body.groq_api_key is not None and body.groq_api_key.strip() != "__KEEP__":
            updates["groq_api_key"] = body.groq_api_key.strip()
        if body.yt_dlp_cookies_text is not None:
            save_youtube_cookies_text(body.yt_dlp_cookies_text)
        save_settings(updates)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot update storage/cloud settings: {exc}") from exc
    return get_storage_status()


@app.post("/api/storage/cleanup")
def api_cleanup_storage() -> dict[str, Any]:
    freed_bytes = 0
    for d in (get_work_dir(), get_upload_dir(), BASE_DIR / "work", BASE_DIR / "uploads"):
        if not d.exists():
            continue
        for item in list(d.iterdir()):
            try:
                if item.is_dir():
                    # Don't touch active running jobs
                    job_fp = JOBS_DIR / f"{item.name}.json"
                    if job_fp.exists():
                        st = json.loads(job_fp.read_text(encoding="utf-8")).get("status")
                        if st == "running":
                            continue
                    freed_bytes += sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item, ignore_errors=True)
                elif item.is_file():
                    freed_bytes += item.stat().st_size
                    item.unlink()
            except OSError:
                pass
    status = get_storage_status()
    status["freed_mb"] = round(freed_bytes / (1024 * 1024), 1)
    return status


@app.get("/api/memory")
def get_memory() -> dict[str, Any]:
    return load_rsi_memory()


@app.post("/api/memory/reset")
def api_reset_memory() -> dict[str, Any]:
    return reset_rsi_memory()


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for fp in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            j = json.loads(fp.read_text(encoding="utf-8"))
            jobs.append(
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "created_at": j["created_at"],
                    "source_label": j.get("source_label", ""),
                    "num_clips": len(j.get("clips", [])),
                }
            )
        except Exception:
            continue
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _load_job(job_id)


@app.post("/api/jobs")
async def create_job(
    video_file: UploadFile | None = File(default=None),
    youtube_url: str = Form(default=""),
    local_path: str = Form(default=""),
    num_clips: int = Form(default=3),
    rsi_iterations: int = Form(default=3),
    layout_preference: str = Form(default="auto"),
    llm_provider: str = Form(default="auto_free"),
    whisper_model: str = Form(default="base"),
    youtube_mode: str = Form(default="audio_first"),
    burn_captions: bool = Form(default=True),
    subtitle_language: str = Form(default="auto"),
    subtitle_script: str = Form(default="roman"),
    watermark_text: str = Form(default=""),
    channel_watermark: str = Form(default=""),
    creator_tag: str = Form(default=""),
    required_hashtags: str = Form(default=""),
    target_max_duration: int = Form(default=55),
    campaign_id: str = Form(default=""),
    auto_upload_youtube: bool = Form(default=False),
    youtube_privacy: str = Form(default="private"),
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    is_uploaded_copy = False
    if video_file and video_file.filename:
        dest = get_upload_dir() / f"{job_id}_{Path(video_file.filename).name}"
        with dest.open("wb") as f:
            while chunk := await video_file.read(1024 * 1024):
                f.write(chunk)
        source_mode = "file"
        source_value = str(dest)
        source_label = video_file.filename
        is_uploaded_copy = True
    elif youtube_url.strip():
        source_mode = "youtube"
        source_value = youtube_url.strip()
        source_label = youtube_url.strip()
    elif local_path.strip():
        p = Path(local_path.strip()).expanduser()
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Local file does not exist: {p}")
        source_mode = "file"
        source_value = str(p)
        source_label = p.name
    else:
        raise HTTPException(status_code=400, detail="Provide a video file, YouTube URL, or local path.")

    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "youtube_mode": youtube_mode,
        "source_value": source_value,
        "source_label": source_label,
        "is_uploaded_copy": is_uploaded_copy,
        "num_clips": max(1, min(10, num_clips)),
        "rsi_iterations": max(1, min(6, rsi_iterations)),
        "layout_preference": layout_preference,
        "llm_provider": llm_provider,
        "whisper_model": whisper_model,
        "burn_captions": burn_captions,
        "subtitle_language": subtitle_language,
        "subtitle_script": subtitle_script,
        "watermark_text": watermark_text.strip()[:48],
        "channel_watermark": channel_watermark.strip()[:48],
        "creator_tag": creator_tag.strip()[:48],
        "required_hashtags": required_hashtags.strip()[:140],
        "target_max_duration": max(15, min(60, int(target_max_duration))),
        "campaign_id": campaign_id.strip(),
        "auto_upload_youtube": bool(auto_upload_youtube),
        "youtube_privacy": youtube_privacy.strip().lower() or "private",
        "logs": [],
        "clips": [],
        "rsi_traces": [],
    }
    _save_job(job)
    executor.submit(_process_job, job_id)
    return job


class FeedbackBody(BaseModel):
    rating: str
    note: str = ""


@app.post("/api/clips/{job_id}/{clip_id}/feedback")
def submit_feedback(job_id: str, clip_id: str, body: FeedbackBody) -> dict[str, Any]:
    job = _load_job(job_id)
    clip = next((c for c in job.get("clips", []) if c["clip_id"] == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip["human_feedback"] = {"rating": body.rating, "note": body.note}
    _save_job(job)
    return learn_from_human_feedback(clip, body.rating, body.note)


class RerenderBody(BaseModel):
    start: float
    end: float
    layout: str
    hook_title: str


@app.post("/api/clips/{job_id}/{clip_id}/rerender")
def rerender_clip(job_id: str, clip_id: str, body: RerenderBody) -> dict[str, Any]:
    job = _load_job(job_id)
    clip = next((c for c in job.get("clips", []) if c["clip_id"] == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    work_dir = get_work_dir() / job_id
    out_dir = get_output_dir() / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate transcript & visual metadata
    t_path = work_dir / "transcript.json"
    v_path = work_dir / "visual.json"
    if not t_path.exists():
        t_path = BASE_DIR / "work" / job_id / "transcript.json"
    if not v_path.exists():
        v_path = BASE_DIR / "work" / job_id / "visual.json"

    transcript = json.loads(t_path.read_text(encoding="utf-8"))
    visual = json.loads(v_path.read_text(encoding="utf-8"))
    meta = job["video_meta"]

    clip["start"] = round(max(0.0, body.start), 2)
    clip["end"] = round(min(meta["duration"], max(clip["start"] + 3.0, body.end)), 2)
    clip["layout"] = body.layout
    clip["hook_title"] = body.hook_title.strip()

    source_offset = 0.0
    src_path_str = job.get("video_path")
    src_video = Path(src_path_str) if src_path_str else None
    temp_slice: Path | None = None

    if not src_video or not src_video.exists():
        if job["source_mode"] == "youtube":
            s0 = max(0.0, clip["start"] - 2.0)
            s1 = min(meta["duration"], clip["end"] + 2.0)
            temp_slice = download_youtube_section(
                job["source_value"], s0, s1, work_dir / f"rerender_{clip_id}.mp4"
            )
            src_video = temp_slice
            source_offset = s0
        else:
            raise HTTPException(status_code=400, detail="Source video was cleaned up and cannot be re-fetched.")

    out_file = out_dir / f"{clip_id}.mp4"
    render_short_clip(
        source_video=src_video,
        output_path=out_file,
        segments=transcript["segments"],
        visual=visual,
        clip=clip,
        src_w=meta["width"],
        src_h=meta["height"],
        work_dir=work_dir,
        burn_captions=bool(job.get("burn_captions", True)),
        source_offset=source_offset,
    )
    if temp_slice and temp_slice.exists():
        temp_slice.unlink()

    clip["file_size_mb"] = round(out_file.stat().st_size / (1024 * 1024), 2)
    _save_job(job)
    return clip


class RetranscribeBody(BaseModel):
    subtitle_language: str = "auto"
    subtitle_script: str = "roman"


def _retranscribe_job_worker(job_id: str, sub_lang: str, sub_script: str) -> None:
    job = _load_job(job_id)
    work_dir = get_work_dir() / job_id
    out_dir = get_output_dir() / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        _append_log(job, msg)

    try:
        job["status"] = "running"
        job["subtitle_language"] = sub_lang
        job["subtitle_script"] = sub_script
        _save_job(job)

        v_path = work_dir / "visual.json"
        if not v_path.exists():
            v_path = BASE_DIR / "work" / job_id / "visual.json"
        visual = (
            json.loads(v_path.read_text(encoding="utf-8"))
            if v_path.exists()
            else {"frames": [], "scene_cuts": [], "dominant_style": "screen_share"}
        )
        meta = job.get("video_meta") or {"width": 1920, "height": 1080, "fps": 30.0, "duration": 3600.0}
        llm_pref = job.get("llm_provider", "auto_free")
        video_title = str(job.get("source_label") or "")

        clips = job.get("clips", [])
        log(f"Re-transcribing {len(clips)} clips with Precision Whisper v3 ({sub_lang}/{sub_script})...")
        for idx, clip in enumerate(clips, 1):
            cid = clip["clip_id"]
            c_start = float(clip["start"])
            c_end = float(clip["end"])
            source_offset = 0.0
            src_video: Path | None = None
            temp_slice: Path | None = None

            src_path_str = job.get("video_path")
            if src_path_str and Path(src_path_str).exists():
                src_video = Path(src_path_str)
            elif job.get("source_mode") == "youtube":
                s0 = max(0.0, c_start - 2.0)
                s1 = min(float(meta.get("duration") or (c_end + 10.0)), c_end + 2.0)
                temp_slice = download_youtube_section(
                    job["source_value"], s0, s1, work_dir / f"resub_{cid}.mp4", log_cb=log
                )
                src_video = temp_slice
                source_offset = s0
            else:
                raise FileNotFoundError(f"Source media unavailable for {cid}")

            prec = transcribe_clip_precision(
                media_path=src_video,
                clip_start=c_start,
                clip_end=c_end,
                source_offset=source_offset,
                model_size=job.get("whisper_model", "small"),
                subtitle_language=sub_lang,
                subtitle_script=sub_script,
                video_title=video_title,
                llm_provider=llm_pref,
                log_cb=log,
            )
            prec_segs = prec.get("segments") or []
            if prec_segs:
                new_txt = " ".join(s.get("text", "") for s in prec_segs).strip()
                if new_txt and clip.get("evaluation"):
                    clip["evaluation"]["text_preview"] = (
                        new_txt[:280] + ("..." if len(new_txt) > 280 else "")
                    )
                sug_hook = prec_segs[0].pop("_suggested_hook", None)
                if sug_hook:
                    clip["hook_title"] = sug_hook

            log(f"Re-rendering {cid} ({idx}/{len(clips)}) with precision subtitles...")
            out_file = out_dir / f"{cid}.mp4"
            render_short_clip(
                source_video=src_video,
                output_path=out_file,
                segments=prec_segs,
                visual=visual,
                clip=clip,
                src_w=int(meta.get("width", 1920)),
                src_h=int(meta.get("height", 1080)),
                work_dir=work_dir,
                burn_captions=True,
                source_offset=source_offset,
            )
            if temp_slice and temp_slice.exists():
                temp_slice.unlink()
            clip["file_size_mb"] = round(out_file.stat().st_size / (1024 * 1024), 2)
            clip["post_pack"] = generate_bounty_post_pack(
                clip,
                creator_tag=str(job.get("creator_tag") or ""),
                required_hashtags=str(job.get("required_hashtags") or ""),
                watermark_text=str(job.get("watermark_text") or clip.get("watermark_text") or ""),
                burn_captions=True,
            )
            _save_job(job)

        if load_settings().get("auto_cleanup_work", True):
            freed_mb = _cleanup_job_temp_media(work_dir)
            if freed_mb > 0:
                log(f"Auto-cleaned {freed_mb} MB of temporary slice media.")
        job["status"] = "completed"
        log("Precision subtitle re-transcription complete!")
        _save_job(job)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{exc}\n{traceback.format_exc()}"
        log(f"ERROR during subtitle fix: {exc}")
        _save_job(job)


@app.post("/api/jobs/{job_id}/retranscribe")
def retranscribe_job(job_id: str, body: RetranscribeBody) -> dict[str, Any]:
    job = _load_job(job_id)
    if not job.get("clips"):
        raise HTTPException(status_code=400, detail="Job has no rendered clips to re-transcribe.")
    if job.get("status") == "running":
        raise HTTPException(status_code=409, detail="Job is currently running.")
    job["status"] = "queued"
    _save_job(job)
    executor.submit(_retranscribe_job_worker, job_id, body.subtitle_language, body.subtitle_script)
    return job


class CampaignBody(BaseModel):
    campaign_id: str = ""
    name: str
    platform: str = "Whop Content Rewards"
    cpm_usd: float = 1.5
    max_clip_payout_usd: float = 150.0
    vod_url: str = ""
    creator_tag: str = ""
    watermark_text: str = ""
    required_hashtags: str = "#gaming #kick #clips #shorts"
    target_max_duration: int = 55
    notes: str = ""


class SubmissionBody(BaseModel):
    platform: str = "YouTube Shorts"
    post_url: str
    clip_label: str = ""
    views: int = 0


class SubmissionViewsBody(BaseModel):
    views: int


@app.get("/api/campaigns")
def api_list_campaigns() -> dict[str, Any]:
    items = load_campaigns()
    return {
        "campaigns": items,
        "total_views": sum(c.get("total_views", 0) for c in items),
        "total_est_usd": round(sum(c.get("total_est_usd", 0.0) for c in items), 2),
    }


@app.post("/api/campaigns")
def api_upsert_campaign(body: CampaignBody) -> dict[str, Any]:
    saved = upsert_campaign(body.model_dump())
    return {"campaign": saved, **api_list_campaigns()}


@app.delete("/api/campaigns/{campaign_id}")
def api_delete_campaign(campaign_id: str) -> dict[str, Any]:
    delete_campaign(campaign_id)
    return api_list_campaigns()


@app.post("/api/campaigns/{campaign_id}/submissions")
def api_add_submission(campaign_id: str, body: SubmissionBody) -> dict[str, Any]:
    updated = add_campaign_submission(
        campaign_id, body.platform, body.post_url, body.clip_label, body.views
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"campaign": updated, **api_list_campaigns()}


@app.patch("/api/campaigns/{campaign_id}/submissions/{sub_id}")
def api_update_submission(
    campaign_id: str, sub_id: str, body: SubmissionViewsBody
) -> dict[str, Any]:
    updated = update_submission_views(campaign_id, sub_id, body.views)
    if not updated:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"campaign": updated, **api_list_campaigns()}


def _resolve_clip_mp4(job: dict[str, Any], clip_id: str) -> Path:
    job_id = job["job_id"]
    candidates = [
        Path(job.get("output_dir") or (get_output_dir() / job_id)) / f"{clip_id}.mp4",
        get_output_dir() / job_id / f"{clip_id}.mp4",
        BASE_DIR / "outputs" / job_id / f"{clip_id}.mp4",
    ]
    for fp in candidates:
        if fp.exists() and fp.is_file():
            return fp
    raise FileNotFoundError(f"Rendered MP4 not found for {clip_id}")


def _upload_clip_to_youtube_record(
    job: dict[str, Any],
    clip: dict[str, Any],
    privacy_status: str = "private",
    publish_at_iso: str = "",
    title_override: str = "",
    description_override: str = "",
) -> dict[str, Any]:
    mp4 = _resolve_clip_mp4(job, clip["clip_id"])
    pp = clip.get("post_pack") or {}
    title = title_override.strip() or str(pp.get("title") or clip.get("hook_title") or "Stream Highlight")
    desc = description_override.strip() or str(pp.get("caption") or f"{title}\n#shorts")
    res = upload_short_to_youtube(
        video_path=mp4,
        title=title,
        description=desc,
        privacy_status=privacy_status,
        publish_at_iso=publish_at_iso,
    )
    clip["youtube_upload"] = res
    clip.pop("youtube_upload_error", None)
    cid = str(job.get("campaign_id") or "").strip()
    if cid:
        add_campaign_submission(
            cid,
            platform="YouTube Shorts",
            post_url=res["url"],
            clip_label=title[:60],
            views=0,
        )
    return res


class YoutubeConfigBody(BaseModel):
    client_secret_json_text: str = ""
    client_id: str = ""
    client_secret: str = ""


class YoutubeCodeBody(BaseModel):
    code: str
    redirect_uri: str = ""


class YoutubeUploadBody(BaseModel):
    privacy_status: str = "private"
    publish_at_iso: str = ""
    title: str = ""
    description: str = ""


@app.get("/api/youtube/status")
def api_youtube_status() -> dict[str, Any]:
    return get_youtube_status()


@app.post("/api/youtube/config")
def api_youtube_config(body: YoutubeConfigBody) -> dict[str, Any]:
    try:
        save_youtube_client_credentials(
            client_secret_json_text=body.client_secret_json_text,
            client_id=body.client_id,
            client_secret=body.client_secret,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_youtube_status()


@app.get("/api/youtube/auth-url")
def api_youtube_auth_url(request: Request) -> dict[str, str]:
    base = str(request.base_url).rstrip("/")
    redirect_uri = f"{base}/api/youtube/callback"
    try:
        url = build_youtube_auth_url(redirect_uri=redirect_uri)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"auth_url": url, "redirect_uri": redirect_uri}


@app.get("/api/youtube/callback", response_class=HTMLResponse)
def api_youtube_callback(request: Request, code: str = "", state: str = "", error: str = "") -> HTMLResponse:
    if error:
        return HTMLResponse(
            f"<h3>YouTube connection canceled/error: {error}</h3><p><a href='/'>Back to Studio</a></p>",
            status_code=400,
        )
    if not code:
        return HTMLResponse("<h3>Missing authorization code.</h3><p><a href='/'>Back to Studio</a></p>", status_code=400)
    redirect_uri = str(request.base_url).rstrip("/") + "/api/youtube/callback"
    try:
        status = exchange_youtube_code(code=code, state=state, redirect_uri=redirect_uri)
    except Exception as exc:
        return HTMLResponse(
            f"<h3>YouTube OAuth failed:</h3><pre>{exc}</pre><p><a href='/'>Back to Studio</a></p>",
            status_code=400,
        )
    ch = status.get("channel_title") or "Your YouTube Channel"
    return HTMLResponse(
        f"""
        <html><body style="background:#0b0e14;color:#e8edf5;font-family:sans-serif;padding:40px;text-align:center;">
          <h2>✅ Connected: {ch}</h2>
          <p>You can close this tab or click below to return to RSI Shorts Studio.</p>
          <p><a href="/?yt_connected=1" style="color:#2dd4bf;">Return to RSI Shorts Studio</a></p>
          <script>setTimeout(() => {{ window.location.href = '/?yt_connected=1'; }}, 1200);</script>
        </body></html>
        """
    )


@app.post("/api/youtube/exchange-code")
def api_youtube_exchange_code(body: YoutubeCodeBody) -> dict[str, Any]:
    try:
        return exchange_youtube_code(code=body.code, redirect_uri=body.redirect_uri)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/youtube/disconnect")
def api_youtube_disconnect() -> dict[str, Any]:
    clear_youtube_tokens()
    return get_youtube_status()


@app.post("/api/clips/{job_id}/{clip_id}/upload-youtube")
def api_upload_clip_youtube(job_id: str, clip_id: str, body: YoutubeUploadBody) -> dict[str, Any]:
    job = _load_job(job_id)
    clip = next((c for c in job.get("clips", []) if c["clip_id"] == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    try:
        res = _upload_clip_to_youtube_record(
            job,
            clip,
            privacy_status=body.privacy_status,
            publish_at_iso=body.publish_at_iso,
            title_override=body.title,
            description_override=body.description,
        )
        _save_job(job)
    except Exception as exc:
        clip["youtube_upload_error"] = str(exc)
        _save_job(job)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"upload": res, "clip": clip}


COMPILATIONS_DIR = DATA_DIR / "compilations"
COMPILATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_comp(comp_id: str) -> dict[str, Any]:
    fp = COMPILATIONS_DIR / f"{comp_id}.json"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Compilation not found")
    return json.loads(fp.read_text(encoding="utf-8"))


def _save_comp(comp: dict[str, Any]) -> None:
    COMPILATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (COMPILATIONS_DIR / f"{comp['compilation_id']}.json").write_text(
        json.dumps(comp, indent=2), encoding="utf-8"
    )


def _compilation_worker(comp_id: str) -> None:
    comp = _load_comp(comp_id)
    work_dir = get_work_dir() / f"comp_{comp_id}"
    out_dir = get_output_dir() / "compilations" / comp_id

    def log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        comp.setdefault("logs", []).append(f"[{ts}] {msg}")
        _save_comp(comp)

    try:
        comp["status"] = "running"
        _save_comp(comp)
        jobs = [_load_job(jid) for jid in comp.get("job_ids", []) if _job_file(jid).exists()]
        intro_audio_str = str(comp.get("intro_audio_path") or "").strip()
        intro_audio_p = Path(intro_audio_str) if intro_audio_str else None
        res = build_weekly_compilation(
            comp_id=comp_id,
            title=comp["title"],
            jobs=jobs,
            out_dir=out_dir,
            work_dir=work_dir,
            max_clips=int(comp.get("max_clips", 8)),
            watermark_text=str(comp.get("watermark_text") or ""),
            channel_watermark=str(comp.get("channel_watermark") or "@RAKAGTAVI"),
            creator_tag=str(comp.get("creator_tag") or ""),
            required_hashtags=str(comp.get("required_hashtags") or ""),
            vfx_enabled=bool(comp.get("vfx_enabled", True)),
            transition_style=str(comp.get("transition_style") or "slide_wipe"),
            intro_audio_path=intro_audio_p,
            log_cb=log,
        )
        comp.update(res)
        if load_settings().get("auto_cleanup_work", True):
            freed_mb = _cleanup_job_temp_media(work_dir, upload_path=intro_audio_p)
            if freed_mb > 0:
                log(f"Auto-cleaned {freed_mb} MB of compilation temp media.")
        comp["status"] = "completed"
        log("1080p compilation video ready!")
        _save_comp(comp)
    except Exception as exc:
        comp["status"] = "failed"
        comp["error"] = f"{exc}\n{traceback.format_exc()}"
        log(f"ERROR: {exc}")
        _save_comp(comp)


@app.get("/api/compilations")
def api_list_compilations() -> list[dict[str, Any]]:
    items = []
    for fp in sorted(COMPILATIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            items.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


@app.get("/api/compilations/{comp_id}")
def api_get_compilation(comp_id: str) -> dict[str, Any]:
    return _load_comp(comp_id)


@app.post("/api/compilations")
async def api_create_compilation(
    title: str = Form(default="Best Stream Moments This Week"),
    job_ids: str = Form(default=""),
    max_clips: int = Form(default=8),
    watermark_text: str = Form(default="Credit: @RakaZoneGaming"),
    channel_watermark: str = Form(default="@RAKAGTAVI"),
    creator_tag: str = Form(default="@RakaZoneGaming"),
    required_hashtags: str = Form(default="#RakaZoneGaming #GTA6 #GTA #Gaming"),
    vfx_enabled: bool = Form(default=True),
    transition_style: str = Form(default="slide_wipe"),
    intro_audio: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    selected_ids = [j.strip() for j in job_ids.split(",") if j.strip()]
    if not selected_ids:
        # Auto-pick up to 7 latest completed jobs if none specified
        for fp in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:7]:
            try:
                j = json.loads(fp.read_text(encoding="utf-8"))
                if j.get("status") == "completed" and j.get("clips"):
                    selected_ids.append(j["job_id"])
            except Exception:
                continue
    if not selected_ids:
        raise HTTPException(status_code=400, detail="Generate at least 1 completed stream job first.")

    comp_id = uuid.uuid4().hex[:8]
    intro_path = ""
    if intro_audio and intro_audio.filename:
        dest = get_upload_dir() / f"comp_intro_{comp_id}_{Path(intro_audio.filename).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            while chunk := await intro_audio.read(1024 * 1024):
                f.write(chunk)
        intro_path = str(dest)

    comp = {
        "compilation_id": comp_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": title.strip()[:95] or "Weekly Stream Compilation",
        "job_ids": selected_ids,
        "max_clips": max(2, min(15, int(max_clips))),
        "watermark_text": watermark_text.strip()[:48],
        "channel_watermark": channel_watermark.strip()[:48],
        "creator_tag": creator_tag.strip()[:48],
        "required_hashtags": required_hashtags.strip()[:160],
        "vfx_enabled": bool(vfx_enabled),
        "transition_style": transition_style,
        "intro_audio_path": intro_path,
        "logs": [],
    }
    _save_comp(comp)
    executor.submit(_compilation_worker, comp_id)
    return comp


@app.post("/api/compilations/{comp_id}/upload-youtube")
def api_upload_compilation_youtube(comp_id: str, body: YoutubeUploadBody) -> dict[str, Any]:
    comp = _load_comp(comp_id)
    out_p = Path(comp.get("output_path") or (get_output_dir() / "compilations" / comp_id / "compilation_1080p.mp4"))
    if not out_p.exists():
        raise HTTPException(status_code=404, detail="Compiled MP4 not found.")
    try:
        res = upload_short_to_youtube(
            video_path=out_p,
            title=body.title.strip() or comp["title"],
            description=body.description.strip() or comp.get("youtube_description") or comp["title"],
            privacy_status=body.privacy_status,
            publish_at_iso=body.publish_at_iso,
            is_short=False,
        )
        comp["youtube_upload"] = res
        comp.pop("youtube_upload_error", None)
        _save_comp(comp)
    except Exception as exc:
        comp["youtube_upload_error"] = str(exc)
        _save_comp(comp)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"upload": res, "compilation": comp}


class MonitorCreatorBody(BaseModel):
    creator_id: str = ""
    name: str
    youtube_url: str = ""
    kick_url: str = ""
    creator_tag: str = ""
    watermark_text: str = ""
    channel_watermark: str = ""
    required_hashtags: str = ""
    default_clips: int = 6
    target_max_duration: int = 45
    campaign_id: str = ""


class MonitorToggleBody(BaseModel):
    url: str
    clipped: bool = True


class MonitorClipNowBody(BaseModel):
    url: str
    title: str = ""
    auto_upload_youtube: bool = False
    youtube_privacy: str = "private"


@app.get("/api/monitor")
def api_get_monitor() -> dict[str, Any]:
    return get_monitor_snapshot()


@app.post("/api/monitor/creators")
def api_upsert_monitor_creator(body: MonitorCreatorBody) -> dict[str, Any]:
    return upsert_creator(body.model_dump())


@app.post("/api/monitor/{creator_id}/scan")
def api_scan_monitor_creator(creator_id: str) -> dict[str, Any]:
    try:
        return scan_creator(creator_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/monitor/{creator_id}/toggle-clipped")
def api_toggle_monitor_clipped(creator_id: str, body: MonitorToggleBody) -> dict[str, Any]:
    try:
        return toggle_manual_clipped(creator_id, body.url, body.clipped)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _submit_monitor_clip_job(
    creator: dict[str, Any],
    url: str,
    title: str = "",
    auto_upload_youtube: bool = False,
    youtube_privacy: str = "private",
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": "youtube",
        "youtube_mode": "audio_first",
        "source_value": url.strip(),
        "source_label": title.strip() or url.strip(),
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
        "auto_upload_youtube": bool(auto_upload_youtube),
        "youtube_privacy": youtube_privacy.strip().lower() or "private",
        "logs": [],
        "clips": [],
        "rsi_traces": [],
    }
    _save_job(job)
    executor.submit(_process_job, job_id)
    return job


@app.post("/api/monitor/{creator_id}/clip-now")
def api_clip_monitor_item(creator_id: str, body: MonitorClipNowBody) -> dict[str, Any]:
    state = load_monitor_state()
    creator = next((c for c in state.get("creators") or [] if c["creator_id"] == creator_id), None)
    if not creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing stream URL")

    job = _submit_monitor_clip_job(
        creator,
        url=url,
        title=body.title,
        auto_upload_youtube=body.auto_upload_youtube,
        youtube_privacy=body.youtube_privacy,
    )
    return {"job": job, **get_monitor_snapshot()}


class AutopilotBody(BaseModel):
    enabled: bool = False
    interval_minutes: int = 60
    auto_upload_youtube: bool = False
    youtube_privacy: str = "private"
    max_jobs_per_day: int = 2
    only_finished_vods: bool = True
    min_duration_sec: int = 900


_autopilot_lock = threading.Lock()


def run_autopilot_sweep() -> dict[str, Any]:
    """Scan creators and auto-clip newest eligible unclipped VOD within daily quota."""
    if not _autopilot_lock.acquire(blocking=False):
        return get_monitor_snapshot()
    try:
        ap = get_autopilot_config()
        now_iso = datetime.now(timezone.utc).isoformat()
        started_today = count_jobs_started_today()
        max_today = int(ap.get("max_jobs_per_day") or 2)
        if started_today >= max_today:
            update_autopilot_config({
                "last_run_at": now_iso,
                "last_action": f"Daily cap reached ({started_today}/{max_today} jobs today)",
            })
            return get_monitor_snapshot()

        only_vods = bool(ap.get("only_finished_vods", True))
        min_dur = int(ap.get("min_duration_sec") or 900)
        state = load_monitor_state()
        triggered_msg = "Checked feeds — no new unclipped streams"

        for c in state.get("creators") or []:
            snap = scan_creator(c["creator_id"])
            updated_c = next((x for x in snap.get("creators") or [] if x["creator_id"] == c["creator_id"]), None)
            if not updated_c:
                continue
            for it in updated_c.get("items") or []:
                if it.get("clip_status") != "pending":
                    continue
                if only_vods and it.get("is_live"):
                    continue
                if not it.get("is_live") and int(it.get("duration_sec") or 0) < min_dur:
                    continue
                job = _submit_monitor_clip_job(
                    updated_c,
                    url=it["url"],
                    title=it["title"],
                    auto_upload_youtube=bool(ap.get("auto_upload_youtube")),
                    youtube_privacy=str(ap.get("youtube_privacy") or "private"),
                )
                triggered_msg = f"Auto-started job {job['job_id']} for {it['title'][:45]}"
                break
            if "Auto-started job" in triggered_msg:
                break

        update_autopilot_config({"last_run_at": now_iso, "last_action": triggered_msg})
        return get_monitor_snapshot()
    finally:
        _autopilot_lock.release()


def _autopilot_loop() -> None:
    while True:
        time.sleep(30)
        try:
            ap = get_autopilot_config()
            if not ap.get("enabled"):
                continue
            interval_s = max(600, int(ap.get("interval_minutes") or 60) * 60)
            last_str = str(ap.get("last_run_at") or "")
            due = True
            if last_str:
                try:
                    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_str)).total_seconds()
                    due = elapsed >= interval_s
                except Exception:
                    due = True
            if due:
                run_autopilot_sweep()
        except Exception:
            pass


@app.on_event("startup")
def _start_autopilot_thread() -> None:
    t = threading.Thread(target=_autopilot_loop, daemon=True, name="autopilot-monitor")
    t.start()


@app.post("/api/monitor/autopilot")
def api_update_autopilot(body: AutopilotBody) -> dict[str, Any]:
    update_autopilot_config(body.model_dump())
    return get_monitor_snapshot()


@app.post("/api/monitor/autopilot/run-now")
def api_run_autopilot_now() -> dict[str, Any]:
    return run_autopilot_sweep()



