from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.analyzer import analyze_visual_stream, compute_audio_profile
from app.config import FFMPEG_BIN
from app.renderer import _ffmpeg_has_ass_filter, generate_ass_subtitles, render_short_clip
from app.rsi_engine import learn_from_human_feedback, reset_rsi_memory, run_rsi_pipeline
from app.transcriber import extract_wav, probe_video


def test_rsi_improves_weak_opener_and_layout() -> None:
    reset_rsi_memory()
    segments = [
        {
            "start": 0.0,
            "end": 3.2,
            "text": "Hey guys, welcome back to the channel.",
            "words": [
                {"word": "Hey", "start": 0.1, "end": 0.4},
                {"word": "guys,", "start": 0.4, "end": 0.9},
                {"word": "welcome", "start": 1.0, "end": 1.6},
                {"word": "back", "start": 1.6, "end": 2.0},
                {"word": "to", "start": 2.0, "end": 2.3},
                {"word": "the", "start": 2.3, "end": 2.6},
                {"word": "channel.", "start": 2.6, "end": 3.1},
            ],
        },
        {
            "start": 3.3,
            "end": 12.0,
            "text": "Why do 90 percent of AI video clippers fail on tutorials?",
            "words": [
                {"word": "Why", "start": 3.3, "end": 3.6},
                {"word": "do", "start": 3.6, "end": 3.9},
                {"word": "90", "start": 3.9, "end": 4.4},
                {"word": "percent", "start": 4.4, "end": 5.0},
                {"word": "of", "start": 5.0, "end": 5.3},
                {"word": "AI", "start": 5.3, "end": 5.8},
                {"word": "video", "start": 5.8, "end": 6.4},
                {"word": "clippers", "start": 6.4, "end": 7.2},
                {"word": "fail", "start": 7.2, "end": 7.8},
                {"word": "on", "start": 7.8, "end": 8.1},
                {"word": "tutorials?", "start": 8.1, "end": 9.2},
            ],
        },
        {
            "start": 9.3,
            "end": 28.0,
            "text": "Because blind 9:16 cropping cuts off the screen share and ruins the lesson completely.",
            "words": [
                {"word": "Because", "start": 9.3, "end": 9.9},
                {"word": "blind", "start": 9.9, "end": 10.6},
                {"word": "9:16", "start": 10.6, "end": 11.4},
                {"word": "cropping", "start": 11.4, "end": 12.4},
                {"word": "cuts", "start": 12.4, "end": 13.0},
                {"word": "off", "start": 13.0, "end": 13.5},
                {"word": "the", "start": 13.5, "end": 13.9},
                {"word": "screen", "start": 13.9, "end": 14.8},
                {"word": "share", "start": 14.8, "end": 15.6},
                {"word": "and", "start": 15.6, "end": 16.0},
                {"word": "ruins", "start": 16.0, "end": 17.0},
                {"word": "the", "start": 17.0, "end": 17.4},
                {"word": "lesson", "start": 17.4, "end": 18.4},
                {"word": "completely.", "start": 18.4, "end": 27.5},
            ],
        },
    ]
    audio = {"bin_sec": 0.5, "energy": [0.2] * 7 + [0.7] * 55, "silences": []}
    visual = {
        "dominant_style": "screen_share",
        "scene_cuts": [],
        "frames": [
            {
                "t": float(t),
                "face_count": 1,
                "primary_cx": 0.85,
                "primary_cy": 0.8,
                "primary_area": 0.01,
                "edge_density": 0.16,
            }
            for t in range(30)
        ],
    }

    res = run_rsi_pipeline(
        segments=segments,
        audio=audio,
        visual=visual,
        video_duration=30.0,
        num_clips=1,
        iterations=3,
        layout_preference="center_9_16",  # intentionally bad initial layout to force RSI fix!
        llm_provider="local_offline",
    )
    traces = res["rsi_traces"]
    assert len(traces) == 3
    first_score = traces[0]["avg_score_before"]
    final_score = traces[-1]["avg_score_after"]
    assert final_score > first_score, f"Expected RSI improvement, got {first_score} -> {final_score}"
    assert res["clips"][0]["layout"] == "blurred_bg_fit"
    assert res["clips"][0]["start"] >= 3.0

    fb = learn_from_human_feedback(res["clips"][0], "down", "hook still a little slow")
    assert fb["generation"] > res["memory_generation"]
    reset_rsi_memory()


def test_end_to_end_synthetic_render(tmp_path: Path) -> None:
    src = tmp_path / "synthetic_src.mp4"
    # Generate a 12s 1280x720 synthetic test video with audio tone via ffmpeg
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    meta = probe_video(src)
    assert meta["width"] == 1280 and meta["height"] == 720
    wav = tmp_path / "a.wav"
    extract_wav(src, wav)
    audio = compute_audio_profile(wav)
    assert len(audio["energy"]) > 10
    visual = analyze_visual_stream(src, sample_fps=1.0)
    assert len(visual["frames"]) >= 8

    segs = [
        {
            "start": 0.5,
            "end": 10.5,
            "text": "Why zero cost AI video clipping works locally.",
            "words": [
                {"word": w, "start": 0.5 + i * 0.8, "end": 1.2 + i * 0.8}
                for i, w in enumerate(["Why", "zero", "cost", "AI", "video", "clipping", "works", "locally."])
            ],
        }
    ]
    clip = {
        "clip_id": "clip_1",
        "start": 0.5,
        "end": 10.5,
        "layout": "blurred_bg_fit",
        "hook_title": "ZERO COST AI CLIPPER",
    }
    ass_file = generate_ass_subtitles(segs, 0.5, 10.5, "ZERO COST AI CLIPPER", tmp_path / "clip_1.ass")
    assert "PlayResX: 1080" in ass_file.read_text(encoding="utf-8")

    out_mp4 = tmp_path / "out_short.mp4"
    render_short_clip(src, out_mp4, segs, visual, clip, meta["width"], meta["height"], tmp_path)
    out_meta = probe_video(out_mp4)
    assert out_meta["width"] == 1080 and out_meta["height"] == 1920
    assert _ffmpeg_has_ass_filter() is True


if __name__ == "__main__":
    import tempfile

    test_rsi_improves_weak_opener_and_layout()
    print("PASS: test_rsi_improves_weak_opener_and_layout")
    with tempfile.TemporaryDirectory() as td:
        test_end_to_end_synthetic_render(Path(td))
    print("PASS: test_end_to_end_synthetic_render")
