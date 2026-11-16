from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np


def compute_audio_profile(wav_path: Path, bin_sec: float = 0.5) -> dict[str, Any]:
    """Compute per-bin RMS loudness (0..1) and silence intervals from 16kHz mono WAV."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) == 0:
        return {"bin_sec": bin_sec, "energy": [], "silences": []}

    step = max(1, int(sr * bin_sec))
    rms_vals: list[float] = []
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        rms_vals.append(rms)

    arr = np.array(rms_vals, dtype=np.float32)
    p95 = float(np.percentile(arr, 95)) if len(arr) else 1.0
    denom = max(p95, 1e-5)
    norm = np.clip(arr / denom, 0.0, 1.5).tolist()

    # Detect silence intervals (energy < 0.08 for >= 0.5s)
    silences: list[dict[str, float]] = []
    in_sil = False
    sil_start = 0.0
    for idx, val in enumerate(norm):
        t = idx * bin_sec
        if val < 0.08 and not in_sil:
            in_sil = True
            sil_start = t
        elif val >= 0.08 and in_sil:
            in_sil = False
            if t - sil_start >= 0.5:
                silences.append({"start": round(sil_start, 2), "end": round(t, 2)})
    if in_sil:
        end_t = len(norm) * bin_sec
        if end_t - sil_start >= 0.5:
            silences.append({"start": round(sil_start, 2), "end": round(end_t, 2)})

    return {
        "bin_sec": bin_sec,
        "energy": [round(float(x), 3) for x in norm],
        "silences": silences,
    }


def analyze_visual_stream(
    video_path: Path,
    sample_fps: float = 1.0,
    log_cb: Callable[[str], None] | None = None,
    time_offset: float = 0.0,
) -> dict[str, Any]:
    """
    Sample video frames via OpenCV to detect faces, screen-share UI density, and scene cuts.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frames": [], "scene_cuts": [], "dominant_style": "unknown"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / max(fps, 1.0)
    effective_fps = sample_fps
    if duration_sec > 3600:
        effective_fps = max(0.15, min(sample_fps, 900.0 / duration_sec))
    elif duration_sec > 900:
        effective_fps = min(sample_fps, 0.5)
    stride = max(1, int(round(fps / max(effective_fps, 0.1))))

    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    face_cascade = None
    if cascade_cls is not None and hasattr(cv2, "data"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cascade_cls(cascade_path)

    frames_info: list[dict[str, Any]] = []
    scene_cuts: list[float] = []
    prev_hist = None

    frame_idx = 0
    sample_count = 0
    while True:
        if total_frames > 0 and frame_idx >= total_frames:
            break
        if stride > 4:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        else:
            ok = False
            frame = None
            ret = cap.grab()
            if not ret:
                break
            ok, frame = cap.retrieve()
            for _ in range(stride - 1):
                if not cap.grab():
                    break
        if not ok or frame is None:
            break

        t = round(time_offset + (frame_idx / fps), 2)
        h, w = frame.shape[:2]
        scale = 480.0 / max(w, 1)
        small = cv2.resize(frame, (480, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = (
            face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24)
            )
            if face_cascade is not None
            else []
        )
        sh, sw = gray.shape[:2]
        face_boxes = []
        for fx, fy, fw, fh in faces:
            face_boxes.append(
                {
                    "cx": round((fx + fw / 2.0) / sw, 3),
                    "cy": round((fy + fh / 2.0) / sh, 3),
                    "area": round((fw * fh) / (sw * sh), 4),
                }
            )
        face_boxes.sort(key=lambda b: b["area"], reverse=True)

        # Edge density helps distinguish crisp UI/screen-shares from natural camera video
        edges = cv2.Canny(gray, 80, 180)
        edge_density = round(float(np.mean(edges > 0)), 4)

        # Histogram scene-cut check
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > 0.38:
                scene_cuts.append(t)
        prev_hist = hist

        frames_info.append(
            {
                "t": t,
                "face_count": len(face_boxes),
                "primary_cx": face_boxes[0]["cx"] if face_boxes else None,
                "primary_cy": face_boxes[0]["cy"] if face_boxes else None,
                "primary_area": face_boxes[0]["area"] if face_boxes else 0.0,
                "edge_density": edge_density,
            }
        )
        sample_count += 1
        frame_idx += stride
        if total_frames > 0 and log_cb and sample_count % 50 == 0:
            pct = min(99, int(100 * frame_idx / total_frames))
            log_cb(f"Visual framing & face scan: {pct}%")

    cap.release()

    if not frames_info:
        return {"frames": [], "scene_cuts": [], "dominant_style": "unknown"}

    face_ratio = sum(1 for f in frames_info if f["face_count"] > 0) / len(frames_info)
    avg_face_area = sum(f["primary_area"] for f in frames_info) / len(frames_info)
    avg_edge = sum(f["edge_density"] for f in frames_info) / len(frames_info)
    multi_face_ratio = sum(1 for f in frames_info if f["face_count"] >= 2) / len(frames_info)

    if avg_edge > 0.11 and avg_face_area < 0.035:
        dominant_style = "screen_share"
    elif multi_face_ratio > 0.25:
        dominant_style = "multi_speaker"
    elif face_ratio > 0.35:
        dominant_style = "talking_head"
    else:
        dominant_style = "mixed_visual"

    return {
        "frames": frames_info,
        "scene_cuts": scene_cuts,
        "dominant_style": dominant_style,
        "stats": {
            "face_ratio": round(face_ratio, 3),
            "avg_face_area": round(avg_face_area, 4),
            "avg_edge_density": round(avg_edge, 4),
            "multi_face_ratio": round(multi_face_ratio, 3),
        },
    }
