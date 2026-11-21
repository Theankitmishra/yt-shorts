from __future__ import annotations

import json
import re
import time
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.config import (
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    RSI_MEMORY_PATH,
    get_groq_api_key,
)

DEFAULT_MEMORY: dict[str, Any] = {
    "generation": 1,
    "strategy_prompt": (
        "Select self-contained 22-55s segments that open mid-insight with a strong curiosity gap, "
        "claim, or question. Never open on greetings or filler conjunctions. Match layout to visual "
        "content: use blurred_bg_fit for screen-shares/tutorials and face_track_9_16 for talking heads."
    ),
    "learned_rules": [
        "Skip introductory filler/greetings ('Hey guys', 'So basically', 'And then') so the first 3 seconds hook immediately.",
        "Snap clip start and end strictly to complete sentence boundaries and natural audio pauses.",
        "For screen-share / high-edge-density segments, prefer blurred_bg_fit or split_screen_focus so UI/text is never cropped.",
        "For off-center single speakers, use face_track_9_16 with smooth tracking.",
        "Target 22-55 seconds with high speech density (130-190 WPM) and minimal dead air.",
    ],
    "bad_opener_patterns": [
        r"^(so|and|but|um|uh|well|yeah|okay|ok|alright|basically|literally|actually)\b",
        r"^(hey guys|welcome back|in this video|today we|before we start|don't forget to subscribe)",
        r"^(it is|this is|that is|they are|he said|she said)\b",
    ],
    "weights": {
        "hook_strength": 0.25,
        "boundary_cleanliness": 0.18,
        "pacing_energy": 0.17,
        "visual_framing": 0.18,
        "standalone_clarity": 0.12,
        "duration_fit": 0.10,
    },
    "preferred_duration": [22, 52],
    "history": [],
}

HOOK_POWER_WORDS = {
    "why",
    "how",
    "secret",
    "mistake",
    "never",
    "stop",
    "actually",
    "truth",
    "nobody",
    "everyone",
    "most",
    "worst",
    "best",
    "free",
    "zero",
    "billion",
    "million",
    "insane",
    "crazy",
    "wrong",
    "hidden",
    "trick",
    "hack",
    "rule",
    "reason",
    "problem",
    "warning",
}

DANGLING_OPENERS = {"it", "this", "that", "they", "them", "these", "those", "he", "she"}


def load_rsi_memory() -> dict[str, Any]:
    if RSI_MEMORY_PATH.exists():
        try:
            data = json.loads(RSI_MEMORY_PATH.read_text(encoding="utf-8"))
            merged = {**DEFAULT_MEMORY, **data}
            return merged
        except Exception:
            pass
    save_rsi_memory(DEFAULT_MEMORY)
    return json.loads(json.dumps(DEFAULT_MEMORY))


def save_rsi_memory(memory: dict[str, Any]) -> None:
    RSI_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RSI_MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def reset_rsi_memory() -> dict[str, Any]:
    fresh = json.loads(json.dumps(DEFAULT_MEMORY))
    save_rsi_memory(fresh)
    return fresh


def detect_available_llm(requested: str = "auto") -> str:
    if requested and requested not in {"auto", "auto_free"}:
        return requested
    # Check Ollama locally (zero cost)
    try:
        base = OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
        r = httpx.get(f"{base}/api/tags", timeout=1.2)
        if r.status_code == 200 and r.json().get("models"):
            return "ollama"
    except Exception:
        pass
    if requested == "auto_free":
        if get_groq_api_key():
            return "groq"
        return "local_offline"
    if get_groq_api_key():
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "local_offline"


_GROQ_CHAT_MODEL_CACHE: str | None = None
_PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]


def _resolve_groq_chat_model(groq_key: str) -> str:
    global _GROQ_CHAT_MODEL_CACHE
    if _GROQ_CHAT_MODEL_CACHE:
        return _GROQ_CHAT_MODEL_CACHE
    try:
        r = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {groq_key}"},
            timeout=10.0,
        )
        if r.status_code == 200:
            ids = {m["id"] for m in (r.json().get("data") or []) if m.get("id")}
            for cand in _PREFERRED_GROQ_MODELS:
                if cand in ids:
                    _GROQ_CHAT_MODEL_CACHE = cand
                    return cand
            chat_candidates = [
                mid
                for mid in sorted(ids)
                if not any(bad in mid.lower() for bad in ("whisper", "guard", "orpheus", "tts"))
            ]
            if chat_candidates:
                _GROQ_CHAT_MODEL_CACHE = chat_candidates[0]
                return _GROQ_CHAT_MODEL_CACHE
    except Exception:
        pass
    return _PREFERRED_GROQ_MODELS[0]


def _call_llm(provider: str, system: str, user: str, json_mode: bool = True) -> str | None:
    try:
        if provider == "groq":
            groq_key = get_groq_api_key()
            if not groq_key:
                return None
            model_id = _resolve_groq_chat_model(groq_key)
            payload: dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.25,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                r = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json=payload,
                    timeout=45.0,
                )
                if r.status_code == 429 and attempt == 0:
                    retry_after = float(r.headers.get("retry-after", "2.0") or 2.0)
                    time.sleep(min(4.5, max(1.0, retry_after)))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]

        if provider == "ollama":
            url = f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions"
            payload: dict[str, Any] = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.4,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            r = httpx.post(url, json=payload, timeout=45.0)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        if provider == "openai" and OPENAI_API_KEY:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.4,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=payload,
                timeout=45.0,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        if provider == "gemini" and GEMINI_API_KEY:
            gen_cfg: dict[str, Any] = {"temperature": 0.4}
            if json_mode:
                gen_cfg["responseMimeType"] = "application/json"
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": gen_cfg,
                },
                timeout=45.0,
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
    return None


def _words_in_range(segments: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    # Find starting segment near `start` (allow segments up to 30s long starting before `start`)
    starts = [float(s["start"]) for s in segments]
    lo = max(0, bisect_left(starts, start - 35.0))
    for seg in segments[lo:]:
        if seg["start"] > end:
            break
        if seg["end"] < start:
            continue
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                if w["end"] >= start - 0.05 and w["start"] <= end + 0.05:
                    words.append(w)
        else:
            # Synthesize word timestamps if only segment text exists
            tokens = seg["text"].split()
            if not tokens:
                continue
            dur = max(0.2, seg["end"] - seg["start"])
            step = dur / len(tokens)
            for idx, tok in enumerate(tokens):
                ws = seg["start"] + idx * step
                we = ws + step
                if we >= start and ws <= end:
                    words.append({"word": tok, "start": round(ws, 3), "end": round(we, 3)})
    return words


def _text_in_range(segments: list[dict[str, Any]], start: float, end: float) -> str:
    words = _words_in_range(segments, start, end)
    if words:
        return " ".join(w["word"] for w in words).strip()
    starts = [float(s["start"]) for s in segments]
    lo = max(0, bisect_left(starts, start - 35.0))
    parts = [s["text"] for s in segments[lo:] if s["start"] <= end and s["end"] >= start]
    return " ".join(parts).strip()


def _visual_stats_for_range(visual: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    frames = [f for f in visual.get("frames", []) if start <= f["t"] <= end]
    if not frames:
        return {
            "face_ratio": 0.0,
            "avg_face_area": 0.0,
            "avg_edge_density": 0.08,
            "cx_offset": 0.0,
            "style": visual.get("dominant_style", "unknown"),
        }
    face_frames = [f for f in frames if f["face_count"] > 0 and f["primary_cx"] is not None]
    face_ratio = len(face_frames) / len(frames)
    avg_area = sum(f["primary_area"] for f in frames) / len(frames)
    avg_edge = sum(f["edge_density"] for f in frames) / len(frames)
    cx_offset = (
        sum(abs(f["primary_cx"] - 0.5) for f in face_frames) / len(face_frames)
        if face_frames
        else 0.0
    )
    if avg_edge > 0.105 and avg_area < 0.035:
        style = "screen_share"
    elif face_ratio > 0.4:
        style = "talking_head"
    else:
        style = "mixed_visual"
    return {
        "face_ratio": round(face_ratio, 3),
        "avg_face_area": round(avg_area, 4),
        "avg_edge_density": round(avg_edge, 4),
        "cx_offset": round(cx_offset, 3),
        "style": style,
    }


def _audio_stats_for_range(audio: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    bin_sec = float(audio.get("bin_sec", 0.5))
    energy = audio.get("energy", [])
    i0 = max(0, int(start / bin_sec))
    i1 = min(len(energy), int(end / bin_sec) + 1)
    slice_e = energy[i0:i1] if i1 > i0 else [0.4]
    avg_e = sum(slice_e) / len(slice_e)
    hook_bins = max(1, int(3.5 / bin_sec))
    hook_e = sum(slice_e[:hook_bins]) / len(slice_e[:hook_bins])

    sil_dur = 0.0
    for s in audio.get("silences", []):
        ov = min(end, s["end"]) - max(start, s["start"])
        if ov > 0:
            sil_dur += ov
    dur = max(1.0, end - start)
    return {
        "avg_energy": round(avg_e, 3),
        "hook_energy": round(hook_e, 3),
        "silence_ratio": round(min(1.0, sil_dur / dur), 3),
    }


def _recommend_layout(vstats: dict[str, Any], preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    style = vstats.get("style", "mixed_visual")
    if style == "screen_share":
        return "blurred_bg_fit"
    if style == "talking_head":
        return "face_track_9_16"
    if vstats.get("face_ratio", 0.0) > 0.25 and vstats.get("avg_edge_density", 0.0) > 0.09:
        return "split_screen_focus"
    return "blurred_bg_fit"


def _make_hook_banner(text: str) -> str:
    """Distill a punchy <= 7-word hook title without repetitive hallucinated tokens."""
    sentences = [s.strip() for s in re.split(r"[.!?।]+", text) if len(s.strip().split()) >= 4]
    candidates = sentences[:8] if sentences else [text]
    best = candidates[0] if candidates else ""
    best_score = -999.0
    for s in candidates:
        toks = [w.strip(",;:-") for w in s.split() if w.strip(",;:-")]
        if not toks:
            continue
        low_toks = [t.lower() for t in toks]
        diversity = len(set(low_toks)) / len(low_toks)
        clean_words = [re.sub(r"[^a-z0-9]", "", w) for w in low_toks]
        sc = (
            diversity * 6.0
            + sum(2.0 for w in clean_words if w in HOOK_POWER_WORDS)
            + (2.0 if any(ch.isdigit() for ch in s) else 0.0)
            + (1.5 if "?" in s or "!" in s else 0.0)
            + (2.0 if 4 <= len(toks) <= 12 else 0.0)
        )
        if sc > best_score:
            best_score = sc
            best = s
    # Strip weak lead-ins and collapse immediate repeated words
    cleaned = re.sub(
        r"^(so|and|but|well|actually|basically|you know|i mean|look|listen)[,\s]+",
        "",
        best.strip(),
        flags=re.IGNORECASE,
    )
    deduped: list[str] = []
    for w in cleaned.split():
        if not deduped or deduped[-1].lower() != w.lower():
            deduped.append(w)
        if len(deduped) >= 7:
            break
    title = " ".join(deduped).strip(" ,;:-")
    if not title:
        title = "HIGHLIGHT MOMENT"
    return title.upper() if len(title) <= 44 else title[:41].rsplit(" ", 1)[0].upper() + "..."


def evaluate_clip(
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
    audio: dict[str, Any],
    visual: dict[str, Any],
    memory: dict[str, Any],
    llm_provider: str = "local_offline",
) -> dict[str, Any]:
    """Score a candidate clip across 6 measurable axes + produce concrete weaknesses & fixes."""
    start, end = float(clip["start"]), float(clip["end"])
    dur = max(0.5, end - start)
    words = _words_in_range(segments, start, end)
    text = " ".join(w["word"] for w in words).strip()
    if not text:
        text = _text_in_range(segments, start, end)

    vstats = _visual_stats_for_range(visual, start, end)
    astats = _audio_stats_for_range(audio, start, end)

    weaknesses: list[str] = []
    suggested_fixes: list[dict[str, Any]] = []

    # 1. Hook strength (first ~3.5s)
    hook_words = [w for w in words if w["start"] <= start + 3.8]
    hook_text = " ".join(w["word"] for w in hook_words).strip()
    hook_lower = hook_text.lower()
    hook_score = 62.0

    bad_opener = False
    for pat in memory.get("bad_opener_patterns", []):
        if re.search(pat, hook_lower):
            bad_opener = True
            hook_score -= 22.0
            weaknesses.append(f"Weak opener starts with filler/greeting: '{' '.join(hook_text.split()[:5])}'")
            suggested_fixes.append({"type": "trim_opener"})
            break

    lower_tokens = [re.sub(r"[^a-z0-9]", "", t) for t in hook_lower.split()]
    power_hits = sum(1 for t in lower_tokens if t in HOOK_POWER_WORDS)
    if power_hits > 0:
        hook_score += min(18.0, power_hits * 7.0)
    if "?" in hook_text:
        hook_score += 9.0
    if any(ch.isdigit() for ch in hook_text):
        hook_score += 8.0
    if astats["hook_energy"] >= 0.45:
        hook_score += 8.0
    elif astats["hook_energy"] < 0.22:
        hook_score -= 10.0
        weaknesses.append("Low vocal energy in first 3.5 seconds.")
    hook_score = max(15.0, min(99.0, hook_score))

    # 2. Boundary cleanliness
    boundary_score = 92.0
    first_seg = next((s for s in segments if s["end"] >= start and s["start"] <= start + 1.5), None)
    last_seg = next((s for s in reversed(segments) if s["start"] <= end and s["end"] >= end - 1.5), None)
    if words:
        first_w_gap = abs(words[0]["start"] - start)
        if first_seg and start - first_seg["start"] > 0.65 and not re.search(r"[.!?]$", words[0]["word"]):
            # Started mid-segment without punctuation boundary
            prev_words = [w for w in (first_seg.get("words") or []) if w["end"] <= start]
            if prev_words and not re.search(r"[.!?]$", prev_words[-1]["word"]):
                boundary_score -= 22.0
                weaknesses.append("Clip starts mid-sentence rather than on a clean clause/sentence boundary.")
                suggested_fixes.append({"type": "snap_start"})
        if first_w_gap > 0.8:
            boundary_score -= 10.0
            weaknesses.append(f"Leading dead air ({first_w_gap:.1f}s) before first word.")
            suggested_fixes.append({"type": "snap_start"})

        if not re.search(r"[.!?\"']$", words[-1]["word"]):
            if last_seg and last_seg["end"] - end > 0.6:
                boundary_score -= 18.0
                weaknesses.append("Clip cuts off before sentence completion.")
                suggested_fixes.append({"type": "snap_end"})
    boundary_score = max(20.0, min(99.0, boundary_score))

    # 3. Pacing & energy
    wpm = (len(words) / dur) * 60.0 if dur > 0 else 0.0
    pacing_score = 78.0
    if 130 <= wpm <= 195:
        pacing_score += 14.0
    elif wpm < 105:
        pacing_score -= 18.0
        weaknesses.append(f"Slow speaking pace ({wpm:.0f} WPM).")
    elif wpm > 225:
        pacing_score -= 8.0
    if astats["silence_ratio"] > 0.18:
        pacing_score -= min(25.0, (astats["silence_ratio"] - 0.15) * 90.0)
        weaknesses.append(f"High dead-air ratio ({int(astats['silence_ratio']*100)}% silence).")
        suggested_fixes.append({"type": "trim_edges"})
    if astats["avg_energy"] >= 0.4:
        pacing_score += 7.0
    pacing_score = max(20.0, min(99.0, pacing_score))

    # 4. Visual framing quality
    layout = clip.get("layout", "auto")
    framing_score = 86.0
    if vstats["style"] == "screen_share" and layout in {"center_9_16", "face_track_9_16"}:
        framing_score = 44.0
        weaknesses.append(
            f"Layout '{layout}' crops screen-share content (edge_density={vstats['avg_edge_density']})."
        )
        suggested_fixes.append({"type": "change_layout", "layout": "blurred_bg_fit"})
    elif vstats["style"] == "talking_head" and layout == "center_9_16" and vstats["cx_offset"] > 0.11:
        framing_score = 55.0
        weaknesses.append("Speaker is off-center; static center_9_16 crop risks cutting off face.")
        suggested_fixes.append({"type": "change_layout", "layout": "face_track_9_16"})
    elif layout == "blurred_bg_fit" and vstats["style"] == "screen_share":
        framing_score = 94.0
    elif layout == "face_track_9_16" and vstats["style"] == "talking_head":
        framing_score = 95.0

    # 5. Standalone clarity & lexical richness
    clarity_score = 84.0
    all_toks = [w["word"].strip(",.!?\"' ").lower() for w in words if w["word"].strip(",.!?\"' ")]
    diversity = (len(set(all_toks)) / len(all_toks)) if len(all_toks) >= 8 else 1.0
    if len(all_toks) >= 8 and diversity < 0.48:
        penalty = min(35.0, (0.50 - diversity) * 110.0)
        clarity_score -= penalty
        hook_score = max(15.0, hook_score - penalty * 0.6)
        weaknesses.append(f"Repetitive speech / low lexical diversity ({diversity:.2f}).")
        suggested_fixes.append({"type": "trim_opener"})

    first_word_clean = re.sub(r"[^a-z]", "", (words[0]["word"].lower() if words else ""))
    if first_word_clean in DANGLING_OPENERS:
        clarity_score -= 18.0
        weaknesses.append(f"Opens with context-dependent pronoun ('{first_word_clean}').")
    if len(words) < 35:
        clarity_score -= 16.0
        weaknesses.append("Too few words to form a complete standalone insight.")
    if len(re.findall(r"[.!?।]", text)) >= 3:
        clarity_score += 8.0
    clarity_score = max(20.0, min(98.0, clarity_score))

    # 6. Duration fit
    min_d, max_d = memory.get("preferred_duration", [22, 52])
    if min_d <= dur <= max_d:
        duration_score = 96.0
    elif dur < min_d:
        duration_score = max(35.0, 96.0 - (min_d - dur) * 3.2)
        weaknesses.append(f"Clip too short ({dur:.1f}s < {min_d}s target).")
        suggested_fixes.append({"type": "expand_end"})
    else:
        duration_score = max(35.0, 96.0 - (dur - max_d) * 2.5)
        weaknesses.append(f"Clip slightly long ({dur:.1f}s > {max_d}s target).")
        suggested_fixes.append({"type": "trim_end"})

    # Optional LLM semantic critique boost
    llm_note = None
    if llm_provider != "local_offline" and len(text) > 40:
        sys_p = (
            "You are a harsh viral Shorts critic. Evaluate the transcript hook, payoff, and clarity. "
            "Return JSON: {\"hook_adjustment\": -10..10, \"clarity_adjustment\": -10..10, "
            "\"better_hook_banner\": \"<=7 WORDS\", \"critique\": \"one concise sentence\"}."
        )
        raw = _call_llm(
            llm_provider,
            sys_p,
            f"Duration: {dur:.1f}s\nCurrent Hook Banner: {clip.get('hook_title','')}\nTranscript:\n{text[:1800]}",
            json_mode=True,
        )
        if raw:
            try:
                j = json.loads(raw)
                hook_score = max(15.0, min(99.0, hook_score + float(j.get("hook_adjustment", 0))))
                clarity_score = max(20.0, min(99.0, clarity_score + float(j.get("clarity_adjustment", 0))))
                if j.get("critique"):
                    llm_note = str(j["critique"])
                if j.get("better_hook_banner"):
                    clip["_llm_suggested_banner"] = str(j["better_hook_banner"]).upper()[:48]
            except Exception:
                pass

    subscores = {
        "hook_strength": round(hook_score, 1),
        "boundary_cleanliness": round(boundary_score, 1),
        "pacing_energy": round(pacing_score, 1),
        "visual_framing": round(framing_score, 1),
        "standalone_clarity": round(clarity_score, 1),
        "duration_fit": round(duration_score, 1),
    }
    weights = memory.get("weights", DEFAULT_MEMORY["weights"])
    total = sum(subscores[k] * float(weights.get(k, 0.15)) for k in subscores)

    if llm_note and llm_note not in weaknesses:
        weaknesses.append(f"LLM Critic: {llm_note}")

    return {
        "overall_score": round(total, 1),
        "subscores": subscores,
        "wpm": round(wpm, 1),
        "visual_style": vstats["style"],
        "weaknesses": weaknesses,
        "suggested_fixes": suggested_fixes,
        "text_preview": text[:280] + ("..." if len(text) > 280 else ""),
    }


def _refine_clip(
    clip: dict[str, Any],
    evaluation: dict[str, Any],
    segments: list[dict[str, Any]],
    video_duration: float,
    memory: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply concrete RSI mutations to fix identified weaknesses."""
    refined = dict(clip)
    actions: list[str] = []
    start = float(refined["start"])
    end = float(refined["end"])
    min_d, max_d = memory.get("preferred_duration", [22, 52])

    fix_types = {f["type"]: f for f in evaluation.get("suggested_fixes", [])}
    words = _words_in_range(segments, max(0.0, start - 3.0), min(video_duration, end + 5.0))

    # 1. Trim weak/filler opener forward to next sentence or punchy word
    if "trim_opener" in fix_types and words:
        in_clip = [w for w in words if start - 0.1 <= w["start"] <= start + 7.5]
        advanced = None
        for idx, w in enumerate(in_clip[:-1]):
            if re.search(r"[.!?]$", w["word"]) and in_clip[idx + 1]["start"] > start + 0.5:
                advanced = in_clip[idx + 1]["start"]
                break
        if advanced is None and len(in_clip) > 4:
            # Skip first filler clause up to 2.5s if next word isn't a filler
            for w in in_clip[1:8]:
                low = re.sub(r"[^a-z]", "", w["word"].lower())
                if low not in {"so", "and", "but", "um", "uh", "well", "yeah", "okay", "ok", "actually", "basically"}:
                    if w["start"] > start + 0.35:
                        advanced = w["start"]
                        break
        if advanced is not None and (end - advanced) >= 16.0:
            actions.append(f"Trimmed weak opener +{advanced - start:.2f}s (start {start:.2f}s -> {advanced:.2f}s)")
            start = max(0.0, advanced - 0.08)

    # 2. Snap start to clean segment/sentence boundary
    if "snap_start" in fix_types and words:
        seg_starts = [s["start"] for s in segments if start - 2.5 <= s["start"] <= start + 3.5]
        if seg_starts:
            best_s = min(seg_starts, key=lambda x: abs(x - start))
            if abs(best_s - start) > 0.15 and (end - best_s) >= 16.0:
                actions.append(f"Snapped start to sentence boundary ({start:.2f}s -> {best_s:.2f}s)")
                start = max(0.0, best_s - 0.05)

    # 3. Snap or trim/expand end to clean sentence ending
    if "snap_end" in fix_types or "trim_end" in fix_types or "expand_end" in fix_types:
        target_end = end
        if "trim_end" in fix_types:
            target_end = min(end, start + max_d - 2.0)
        elif "expand_end" in fix_types:
            target_end = min(video_duration, start + min_d + 4.0)

        sent_ends = [
            w["end"]
            for w in words
            if start + 15.0 <= w["end"] <= min(video_duration, start + max_d + 4.0)
            and re.search(r"[.!?\"']$", w["word"])
        ]
        if sent_ends:
            best_e = min(sent_ends, key=lambda x: abs(x - target_end))
            if abs(best_e - end) > 0.2:
                actions.append(f"Snapped end to sentence payoff ({end:.2f}s -> {best_e + 0.22:.2f}s)")
                end = min(video_duration, best_e + 0.22)

    # 4. Layout self-correction
    if "change_layout" in fix_types:
        new_layout = fix_types["change_layout"]["layout"]
        if new_layout != refined.get("layout"):
            actions.append(f"Switched layout '{refined.get('layout')}' -> '{new_layout}' to fix framing")
            refined["layout"] = new_layout

    # 5. Hook banner upgrade
    new_text = _text_in_range(segments, start, end)
    if refined.get("_llm_suggested_banner"):
        new_banner = refined.pop("_llm_suggested_banner")
    else:
        new_banner = _make_hook_banner(new_text)
    if new_banner and new_banner != refined.get("hook_title"):
        actions.append(f"Sharpened hook banner -> '{new_banner}'")
        refined["hook_title"] = new_banner

    refined["start"] = round(start, 2)
    refined["end"] = round(max(start + 12.0, min(video_duration, end)), 2)
    return refined, actions


def _evolve_memory(
    memory: dict[str, Any],
    evaluations: list[dict[str, Any]],
    llm_provider: str,
) -> list[str]:
    """Meta-optimizer: inspect recurring batch weaknesses and update rsi_memory.json."""
    updates: list[str] = []
    all_weaknesses = " | ".join(w for ev in evaluations for w in ev.get("weaknesses", [])).lower()

    if "filler/greeting" in all_weaknesses:
        rule = "Advance clip start past opening conjunctions/greetings to first concrete claim or question."
        if rule not in memory["learned_rules"]:
            memory["learned_rules"].append(rule)
            updates.append("Added rule: advance start past filler greetings/conjunctions.")

    if "crops screen-share" in all_weaknesses:
        rule = "Auto-select blurred_bg_fit or split_screen_focus whenever edge_density > 0.105 (screen-share)."
        if rule not in memory["learned_rules"]:
            memory["learned_rules"].append(rule)
            updates.append("Added rule: preserve screen-share readability via blurred_bg_fit.")

    if "mid-sentence" in all_weaknesses or "before sentence completion" in all_weaknesses:
        rule = "Enforce strict Whisper word-level sentence boundary snapping on both clip start and payoff end."
        if rule not in memory["learned_rules"]:
            memory["learned_rules"].append(rule)
            updates.append("Added rule: strict Whisper sentence-boundary snapping.")

    if llm_provider != "local_offline" and evaluations:
        summary = [
            {"score": e["overall_score"], "weaknesses": e["weaknesses"][:3]}
            for e in evaluations
        ]
        raw = _call_llm(
            llm_provider,
            "You are an RSI Meta-Optimizer improving a video clipping prompt. "
            "Given current prompt and batch critiques, output JSON: "
            "{\"improved_strategy_prompt\": \"concise 2-3 sentence prompt\", \"new_rule\": \"optional 1-line rule or empty\"}.",
            f"Current Prompt:\n{memory['strategy_prompt']}\n\nBatch Critiques:\n{json.dumps(summary)}",
            json_mode=True,
        )
        if raw:
            try:
                j = json.loads(raw)
                if j.get("improved_strategy_prompt"):
                    memory["strategy_prompt"] = str(j["improved_strategy_prompt"]).strip()
                    updates.append("LLM Meta-Optimizer rewrote strategy_prompt.")
                if j.get("new_rule") and str(j["new_rule"]).strip() not in memory["learned_rules"]:
                    nr = str(j["new_rule"]).strip()
                    memory["learned_rules"].append(nr)
                    updates.append(f"LLM Meta-Optimizer learned rule: {nr}")
            except Exception:
                pass

    if updates:
        memory["generation"] = int(memory.get("generation", 1)) + 1
        memory["learned_rules"] = memory["learned_rules"][-12:]
        save_rsi_memory(memory)
    return updates


COPYRIGHT_SEGMENT_RE = re.compile(
    r"\b(bigg?\s*boss|salman\s*khan|weekend\s*ka\s*vaar|wkv|star\s*india|star\s*plus|"
    r"colors\s*tv|jiocinema|hotstar|housemates|eviction|nomination|watch\s*party|watchparty)\b",
    re.IGNORECASE,
)


def _overlaps_blocked(start: float, end: float, blocked_ranges: list[tuple[float, float]] | None) -> bool:
    if not blocked_ranges:
        return False
    for b_start, b_end in blocked_ranges:
        if min(end, b_end) - max(start, b_start) > 1.5:
            return True
    return False


def _is_copyright_risk_segment(text: str) -> bool:
    return bool(COPYRIGHT_SEGMENT_RE.search(str(text or "")))


def _propose_initial_candidates(
    segments: list[dict[str, Any]],
    audio: dict[str, Any],
    visual: dict[str, Any],
    video_duration: float,
    num_clips: int,
    layout_preference: str,
    memory: dict[str, Any],
    llm_provider: str,
    blocked_ranges: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build non-overlapping candidate windows across the video and rank top candidates."""
    if not segments:
        # Fallback fixed windows if video has no speech
        step = max(20.0, video_duration / max(1, num_clips))
        return [
            {
                "clip_id": f"clip_{i+1}",
                "start": round(i * step, 2),
                "end": round(min(video_duration, i * step + 30.0), 2),
                "hook_title": f"HIGHLIGHT {i+1}",
                "layout": "blurred_bg_fit" if layout_preference == "auto" else layout_preference,
            }
            for i in range(num_clips)
        ]

    # Build sentence-aligned sliding windows of ~24-50s
    windows: list[dict[str, Any]] = []
    min_d, max_d = memory.get("preferred_duration", [22, 52])
    target_d = (min_d + max_d) / 2.0

    for i, seg in enumerate(segments):
        s_t = float(seg["start"])
        if s_t > max(0.0, video_duration - 15.0):
            break
        e_t = s_t
        for j in range(i, len(segments)):
            e_t = float(segments[j]["end"])
            if e_t - s_t >= target_d or (e_t - s_t >= min_d and re.search(r"[.!?]$", segments[j]["text"].strip())):
                break
        dur = e_t - s_t
        if dur < 14.0 or dur > 68.0:
            continue
        if _overlaps_blocked(s_t, e_t, blocked_ranges):
            continue
        seg_txt = _text_in_range(segments, s_t, e_t)
        if _is_copyright_risk_segment(seg_txt):
            continue
        vstats = _visual_stats_for_range(visual, s_t, e_t)
        cand = {
            "start": round(s_t, 2),
            "end": round(min(video_duration, e_t), 2),
            "hook_title": _make_hook_banner(seg_txt),
            "layout": _recommend_layout(vstats, layout_preference),
        }
        ev = evaluate_clip(cand, segments, audio, visual, memory, llm_provider="local_offline")
        cand["_seed_score"] = ev["overall_score"]
        windows.append(cand)

    # Optional LLM global candidate selection if configured
    if llm_provider != "local_offline" and len(segments) > 3:
        compact_transcript = "\n".join(
            f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
            for s in segments[:180]
            if not _overlaps_blocked(float(s["start"]), float(s["end"]), blocked_ranges)
            and not _is_copyright_risk_segment(str(s.get("text") or ""))
        )
        raw = _call_llm(
            llm_provider,
            f"You are an expert Shorts Producer. Strategy:\n{memory['strategy_prompt']}\n"
            f"Rules: {json.dumps(memory['learned_rules'])}\n"
            "CRITICAL COPYRIGHT RULE: NEVER select segments where the creator is watching or reacting to "
            "Bigg Boss, TV episodes, movies, trailers, IPL/sports broadcasts, or copyrighted songs; pick ONLY "
            "original gameplay, GTA RP/action, or creator commentary moments.\n"
            f"Pick the best {num_clips} non-overlapping clips (20-55s each). "
            "Return JSON: {\"clips\": [{\"start\": float, \"end\": float, \"hook_title\": \"<=7 WORDS\"}]}.",
            f"Transcript:\n{compact_transcript[:12000]}",
            json_mode=True,
        )
        if raw:
            try:
                llm_clips = json.loads(raw).get("clips", [])
                picked: list[dict[str, Any]] = []
                for idx, lc in enumerate(llm_clips[:num_clips]):
                    st = max(0.0, float(lc["start"]))
                    en = min(video_duration, float(lc["end"]))
                    if en - st >= 12.0 and not _overlaps_blocked(st, en, blocked_ranges) and not _is_copyright_risk_segment(_text_in_range(segments, st, en)):
                        vstats = _visual_stats_for_range(visual, st, en)
                        picked.append(
                            {
                                "clip_id": f"clip_{len(picked)+1}",
                                "start": round(st, 2),
                                "end": round(en, 2),
                                "hook_title": str(lc.get("hook_title") or "MUST WATCH").upper()[:48],
                                "layout": _recommend_layout(vstats, layout_preference),
                            }
                        )
                if picked:
                    return picked
            except Exception:
                pass

    # Greedy non-overlapping top-K selection
    windows.sort(key=lambda c: c["_seed_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for w in windows:
        if len(selected) >= num_clips:
            break
        overlaps = any(not (w["end"] <= s["start"] + 2.0 or w["start"] >= s["end"] - 2.0) for s in selected)
        if not overlaps:
            w.pop("_seed_score", None)
            w["clip_id"] = f"clip_{len(selected)+1}"
            selected.append(w)

    if not selected and segments:
        selected.append(
            {
                "clip_id": "clip_1",
                "start": round(float(segments[0]["start"]), 2),
                "end": round(min(video_duration, float(segments[0]["start"]) + 35.0), 2),
                "hook_title": _make_hook_banner(segments[0]["text"]),
                "layout": _recommend_layout(_visual_stats_for_range(visual, 0, min(35, video_duration)), layout_preference),
            }
        )
    selected.sort(key=lambda c: c["start"])
    for idx, c in enumerate(selected, 1):
        c["clip_id"] = f"clip_{idx}"
    return selected


def run_rsi_pipeline(
    segments: list[dict[str, Any]],
    audio: dict[str, Any],
    visual: dict[str, Any],
    video_duration: float,
    num_clips: int = 3,
    iterations: int = 3,
    layout_preference: str = "auto",
    llm_provider: str = "auto",
    log_cb: Callable[[str], None] | None = None,
    initial_clips: list[dict[str, Any]] | None = None,
    blocked_ranges: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """
    Execute the multi-round Recursive Self-Improvement loop:
      Propose -> Critique -> Self-Refine Clips + Evolve Strategy Memory -> Re-Critique.
    """
    memory = load_rsi_memory()
    resolved_llm = detect_available_llm(llm_provider)
    if log_cb:
        log_cb(
            f"RSI Engine active (provider={resolved_llm}, memory_gen=v{memory['generation']}, rounds={iterations})"
        )

    if initial_clips:
        clips = []
        for c in initial_clips:
            c_copy = dict(c)
            if layout_preference == "auto":
                vstats = _visual_stats_for_range(visual, float(c_copy["start"]), float(c_copy["end"]))
                c_copy["layout"] = _recommend_layout(vstats, "auto")
            else:
                c_copy["layout"] = layout_preference
            clips.append(c_copy)
    else:
        clips = _propose_initial_candidates(
            segments,
            audio,
            visual,
            video_duration,
            num_clips,
            layout_preference,
            memory,
            resolved_llm,
            blocked_ranges=blocked_ranges,
        )

    traces: list[dict[str, Any]] = []

    for round_idx in range(1, max(1, iterations) + 1):
        if log_cb:
            log_cb(f"RSI Round {round_idx}/{iterations}: Evaluating {len(clips)} clips...")

        round_entries: list[dict[str, Any]] = []
        evals: list[dict[str, Any]] = []
        next_clips: list[dict[str, Any]] = []

        for clip in clips:
            ev = evaluate_clip(clip, segments, audio, visual, memory, llm_provider=resolved_llm)
            evals.append(ev)

            if round_idx < iterations:
                candidate, actions = _refine_clip(clip, ev, segments, video_duration, memory)
                if actions:
                    new_ev = evaluate_clip(candidate, segments, audio, visual, memory, llm_provider="local_offline")
                    if new_ev["overall_score"] >= ev["overall_score"]:
                        kept = candidate
                        score_after = new_ev["overall_score"]
                        accepted = True
                    else:
                        kept = clip
                        score_after = ev["overall_score"]
                        accepted = False
                        actions.append("Rolled back mutation (did not improve rubric score).")
                else:
                    kept = clip
                    score_after = ev["overall_score"]
                    accepted = True
            else:
                kept = clip
                actions = []
                score_after = ev["overall_score"]
                accepted = True

            kept["score"] = score_after
            kept["evaluation"] = ev
            next_clips.append(kept)
            round_entries.append(
                {
                    "clip_id": clip["clip_id"],
                    "start": kept["start"],
                    "end": kept["end"],
                    "layout": kept["layout"],
                    "hook_title": kept["hook_title"],
                    "score_before": ev["overall_score"],
                    "score_after": score_after,
                    "subscores": ev["subscores"],
                    "weaknesses": ev["weaknesses"],
                    "mutations": actions,
                    "accepted": accepted,
                }
            )

        meta_updates: list[str] = []
        if round_idx < iterations:
            meta_updates = _evolve_memory(memory, evals, resolved_llm)

        avg_before = round(sum(e["score_before"] for e in round_entries) / max(1, len(round_entries)), 1)
        avg_after = round(sum(e["score_after"] for e in round_entries) / max(1, len(round_entries)), 1)
        traces.append(
            {
                "round": round_idx,
                "memory_generation": memory["generation"],
                "avg_score_before": avg_before,
                "avg_score_after": avg_after,
                "meta_updates": meta_updates,
                "clips": round_entries,
            }
        )
        clips = next_clips

    # Record run summary in RSI memory
    if traces:
        memory.setdefault("history", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_clips": len(clips),
                "initial_avg": traces[0]["avg_score_before"],
                "final_avg": traces[-1]["avg_score_after"],
                "generation": memory["generation"],
            }
        )
        memory["history"] = memory["history"][-25:]
        save_rsi_memory(memory)

    return {
        "llm_provider": resolved_llm,
        "memory_generation": memory["generation"],
        "clips": clips,
        "rsi_traces": traces,
    }


def learn_from_human_feedback(
    clip_data: dict[str, Any],
    rating: str,
    note: str = "",
) -> dict[str, Any]:
    """Update persistent RSI memory based on human approval/rejection & note."""
    memory = load_rsi_memory()
    note_low = note.lower().strip()
    changes: list[str] = []

    if rating == "down":
        if any(k in note_low for k in ["crop", "cut off", "face", "screen", "zoom"]):
            rule = f"Human feedback fix: avoid tight cropping here — {note.strip() or 'prefer blurred_bg_fit or face_track_9_16'}."
            if rule not in memory["learned_rules"]:
                memory["learned_rules"].append(rule)
                changes.append(rule)
            memory["weights"]["visual_framing"] = min(0.32, round(memory["weights"]["visual_framing"] + 0.02, 3))
        if any(k in note_low for k in ["hook", "boring", "slow", "intro", "start"]):
            rule = f"Human feedback fix: tighten first 3s hook — {note.strip() or 'skip slow setup'}."
            if rule not in memory["learned_rules"]:
                memory["learned_rules"].append(rule)
                changes.append(rule)
            memory["weights"]["hook_strength"] = min(0.35, round(memory["weights"]["hook_strength"] + 0.02, 3))
        if any(k in note_low for k in ["long", "short", "duration"]):
            min_d, max_d = memory.get("preferred_duration", [22, 52])
            if "long" in note_low:
                memory["preferred_duration"] = [min_d, max(28, max_d - 6)]
                changes.append(f"Tightened max target duration to {memory['preferred_duration'][1]}s.")
        if not changes and note.strip():
            rule = f"Avoid pattern noted by user: {note.strip()}"
            memory["learned_rules"].append(rule)
            changes.append(rule)
    else:
        dur = float(clip_data.get("end", 30)) - float(clip_data.get("start", 0))
        rule = f"Reinforce winning style: layout={clip_data.get('layout')}, duration~{dur:.0f}s."
        if rule not in memory["learned_rules"]:
            memory["learned_rules"].append(rule)
            changes.append(rule)

    # Re-normalize weights to sum to 1.0
    w_sum = sum(memory["weights"].values())
    if w_sum > 0:
        memory["weights"] = {k: round(v / w_sum, 3) for k, v in memory["weights"].items()}

    memory["generation"] = int(memory.get("generation", 1)) + 1
    memory["learned_rules"] = memory["learned_rules"][-12:]
    save_rsi_memory(memory)
    return {"generation": memory["generation"], "changes": changes, "memory": memory}


def _has_indic_chars(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def _remap_segment_words(seg: dict[str, Any], new_text: str) -> list[dict[str, Any]]:
    """Map repaired words back onto segment timing monotonically."""
    new_tokens = new_text.split()
    old_words = seg.get("words") or []
    if not new_tokens:
        return old_words
    if len(new_tokens) == len(old_words):
        out: list[dict[str, Any]] = []
        for ow, nw in zip(old_words, new_tokens):
            out.append({"word": nw, "start": ow["start"], "end": ow["end"]})
        return out
    s0 = float(old_words[0]["start"]) if old_words else float(seg["start"])
    s1 = float(old_words[-1]["end"]) if old_words else float(seg["end"])
    dur = max(0.25, s1 - s0)
    step = dur / len(new_tokens)
    return [
        {
            "word": tok,
            "start": round(s0 + idx * step, 3),
            "end": round(min(s1, s0 + (idx + 1) * step), 3),
        }
        for idx, tok in enumerate(new_tokens)
    ]


def refine_subtitles_with_llm(
    segments: list[dict[str, Any]],
    video_title: str = "",
    script_mode: str = "roman",
    subtitle_language: str = "auto",
    llm_provider: str = "auto_free",
) -> list[dict[str, Any]]:
    """
    Post-Whisper ASR subtitle corrector.
    - When script_mode='roman': converts Devanagari/Indic script to natural Romanized Hinglish
      and repairs phonetic ASR mishearings anchored to video_title/dialogue context, while keeping
      exact spoken words & 1-to-1 word-level timing (NEVER translating Hindi into English!).
    - When script_mode='native': preserves original script while cleaning phonetic proper nouns.
    """
    if not segments:
        return segments
    resolved = detect_available_llm(llm_provider)
    if resolved == "local_offline":
        return segments

    flat_words: list[dict[str, Any]] = []
    word_refs: list[tuple[int, int]] = []
    for s_idx, seg in enumerate(segments):
        seg_words = seg.get("words") or []
        if not seg_words and seg.get("text", "").strip():
            seg["words"] = _remap_segment_words(seg, seg["text"].strip())
            seg_words = seg["words"]
        for w_idx, w in enumerate(seg_words):
            wt = str(w.get("word", "")).strip()
            if wt:
                flat_words.append(
                    {
                        "idx": len(flat_words),
                        "w": wt,
                        "s": round(float(w["start"]), 2),
                        "e": round(float(w["end"]), 2),
                    }
                )
                word_refs.append((s_idx, w_idx))

    if not flat_words:
        return segments

    # Batch up to 130 words so a single 25-45s Short fits in ONE fast LLM pass (avoids 429s)
    if len(flat_words) <= 130:
        if script_mode == "roman":
            script_rule = (
                "Transliterate any Hindi/Devanagari words into crisp, everyday Romanized Hinglish "
                "(e.g. 'भाई'->'Bhai', 'क्या छुपा है'->'kya chhupa hai', 'कसम'->'kasam', 'नहीं'->'nahi'). "
                "Keep English words in English."
            )
        else:
            script_rule = (
                "Preserve the spoken language's native script (Devanagari for Hindi words, Latin for English words). "
                "Do not transliterate."
            )

        sys_prompt = (
            "You are a precision spoken-subtitle editor for short-form videos. "
            f"Video context/title (use ONLY to spell names/topics, NEVER copy it as hook_title): {video_title or 'Creator stream'}. "
            f"Target language preference: {subtitle_language}. "
            f"{script_rule} "
            "Fix obvious Whisper phonetic mishearings using the video title and surrounding spoken context "
            "(for example misheard creator names like 'Mr. Pat'/'MistPaty'/'मिस्पैटी'/'मिथपाट'->'Mythpat', "
            "'फ्लीट'/'फ्रीट'->'Fleet', 'टेकनो'->'Techno', 'ट्रिगर'->'Triggered', 'लेंप'/'लेम्पी'->'lamp', "
            "'डिस्कॉर्ड'->'Discord', 'अजुबाई'/'जुबाई'->'Ajjubhai', 'बाई'/'बाइन'->'Bhai'). "
            "CRITICAL RULES: "
            "1) NEVER translate Hindi sentences into English meanings — keep the exact spoken Hinglish/Hindi words in exact spoken order! "
            "2) Return a strict 1-to-1 entry for every input `idx` so word-level karaoke timestamps stay locked to audio. "
            "3) For `hook_title`, write a crisp 3-6 word hook SPECIFICALLY about the funniest/most dramatic moment in THIS clip's dialogue (NEVER copy the video title). "
            "Return JSON ONLY: {\"words\": [{\"idx\": int, \"w\": str}], \"hook_title\": \"3-6 WORD CLIP-SPECIFIC HOOK\"}"
        )
        raw = _call_llm(
            resolved,
            sys_prompt,
            json.dumps(flat_words, ensure_ascii=False),
            json_mode=True,
        )
        if raw:
            try:
                parsed = json.loads(raw)
                items = parsed.get("words") or []
                clean_map = {
                    int(it["idx"]): str(it["w"]).strip()
                    for it in items
                    if isinstance(it, dict) and "idx" in it and str(it.get("w", "")).strip()
                }
                if len(clean_map) >= max(3, int(len(flat_words) * 0.7)):
                    updated = json.loads(json.dumps(segments))
                    for flat_idx, (s_idx, w_idx) in enumerate(word_refs):
                        if flat_idx in clean_map:
                            updated[s_idx]["words"][w_idx]["word"] = clean_map[flat_idx]
                    for seg in updated:
                        if seg.get("words"):
                            seg["text"] = " ".join(w["word"] for w in seg["words"]).strip()
                    cand_hook = str(parsed.get("hook_title") or "").strip().upper()[:48]
                    title_toks = set(re.findall(r"[a-z0-9]+", video_title.lower()))
                    hook_toks = set(re.findall(r"[a-z0-9]+", cand_hook.lower()))
                    overlap = len(title_toks & hook_toks) / max(1, len(hook_toks))
                    if cand_hook and overlap < 0.65:
                        updated[0]["_suggested_hook"] = cand_hook
                    elif updated:
                        updated[0]["_suggested_hook"] = _make_hook_banner(
                            " ".join(s.get("text", "") for s in updated)
                        )
                    return updated
            except Exception:
                pass

    # Fallback: segment-level clean if word list exceeded batch size or 1-to-1 failed
    seg_payload = [
        {"index": i, "text": s.get("text", "").strip()}
        for i, s in enumerate(segments)
        if s.get("text", "").strip()
    ]
    raw = _call_llm(
        resolved,
        "Clean ASR phonetic typos and Romanize Hindi to natural Hinglish if mixed/Hindi, "
        "preserving exact spoken words (never translate Hindi into English). "
        "Return JSON: {\"segments\": [{\"index\": int, \"text\": str}]}",
        json.dumps({"title": video_title, "segments": seg_payload}, ensure_ascii=False),
        json_mode=True,
    )
    if raw:
        try:
            items = json.loads(raw).get("segments") or []
            by_idx = {int(it["index"]): str(it["text"]).strip() for it in items if "index" in it and it.get("text")}
            updated = json.loads(json.dumps(segments))
            for i, seg in enumerate(updated):
                if i in by_idx and by_idx[i]:
                    seg["text"] = by_idx[i]
                    seg["words"] = _remap_segment_words(seg, by_idx[i])
            return updated
        except Exception:
            pass
    return segments

