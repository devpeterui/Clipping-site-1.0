"""
highlights.py - transcribe a video you have rights to and pick the best clip candidates.

Install:  pip install faster-whisper anthropic numpy
Also need: ffmpeg on PATH
Env:       ANTHROPIC_API_KEY=...   (optional: CLIP_LLM_MODEL, WHISPER_MODEL)

Usage:     python highlights.py my_video.mp4 --top 8 --out clips.json
"""
import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict, field

import numpy as np

MIN_LEN, MAX_LEN = 20.0, 60.0


@dataclass
class Candidate:
    id: str
    start: float
    end: float
    text: str
    energy: float = 0.0       # 0-1, audio loudness/excitement
    llm_score: float = 0.0    # 0-1, hook/punchline/emotion quality
    hook_score: float = 0.0   # final blended score (feeds trends.rank_segments)
    title: str = ""
    reason: str = ""
    words: list = field(default_factory=list)  # word timestamps, for captions later


# ---------- 1. transcription ----------
def transcribe(path: str):
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), compute_type="int8")
    segs, info = model.transcribe(path, word_timestamps=True, vad_filter=True)
    out = []
    for s in segs:
        out.append({
            "start": s.start, "end": s.end, "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or [])],
        })
    return out


# ---------- 2. audio energy ----------
def audio_energy(path: str, sr: int = 16000) -> np.ndarray:
    """RMS loudness per second, via ffmpeg (no heavy audio libs needed)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(audio) // sr
    if n == 0:
        return np.zeros(1)
    return np.sqrt((audio[: n * sr].reshape(n, sr) ** 2).mean(axis=1))


def window_energy(rms: np.ndarray, start: float, end: float) -> float:
    ref = np.percentile(rms, 90) or 1e-6
    chunk = rms[int(start): max(int(end), int(start) + 1)]
    return float(min(chunk.mean() / ref, 1.0)) if len(chunk) else 0.0


# ---------- 3. candidate windows ----------
def build_windows(segments, rms, stride: int = 2):
    """Slide over Whisper segments so clips start/end on natural sentence boundaries."""
    cands = []
    for i in range(0, len(segments), stride):
        for j in range(i, len(segments)):
            start, end = segments[i]["start"], segments[j]["end"]
            dur = end - start
            if dur > MAX_LEN:
                break
            if dur >= MIN_LEN:
                text = " ".join(s["text"] for s in segments[i: j + 1])
                words = [w for s in segments[i: j + 1] for w in s["words"]]
                cands.append(Candidate(
                    id=f"c{len(cands)}", start=round(start, 2), end=round(end, 2),
                    text=text, energy=window_energy(rms, start, end), words=words))
                break  # shortest valid window for this start; keeps things tidy
    return cands


# ---------- 4. LLM scoring ----------
PROMPT = """You are a short-form video editor picking clips for TikTok/Reels/Shorts.
Score each transcript excerpt 0-10 on: strong opening hook, self-contained story or point,
punchline/insight/emotional peak, and whether it makes sense WITHOUT extra context.
Penalize rambling, filler, and excerpts that start or end mid-thought.

Return ONLY a JSON array, no markdown:
[{"id": "c0", "score": 7.5, "title": "<=8 word title", "reason": "<=15 words"}]

Excerpts:
"""


def score_with_llm(cands, batch_size: int = 15):
    if not os.getenv("ANTHROPIC_API_KEY"):      # no key: fall back to audio energy so everything still works
        for c in cands:
            c.llm_score = round(0.4 + 0.6 * c.energy, 3)
            c.title = " ".join(c.text.split()[:8])
            c.reason = "Scored by audio energy only (no ANTHROPIC_API_KEY set)"
        return
    import anthropic
    client = anthropic.Anthropic()
    model = os.getenv("CLIP_LLM_MODEL", "claude-sonnet-5-5")
    by_id = {c.id: c for c in cands}
    for k in range(0, len(cands), batch_size):
        batch = cands[k: k + batch_size]
        body = "\n\n".join(f'[{c.id}] ({c.end - c.start:.0f}s) {c.text[:1500]}' for c in batch)
        resp = client.messages.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": PROMPT + body}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            for item in json.loads(text):
                c = by_id.get(item["id"])
                if c:
                    c.llm_score = float(item["score"]) / 10
                    c.title, c.reason = item.get("title", ""), item.get("reason", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # leave that batch at 0 rather than crash the pipeline


# ---------- 5. combine + de-duplicate ----------
def pick_top(cands, top_n: int):
    for c in cands:
        c.hook_score = round(0.75 * c.llm_score + 0.25 * c.energy, 4)
    chosen = []
    for c in sorted(cands, key=lambda x: x.hook_score, reverse=True):
        overlaps = any(min(c.end, k.end) - max(c.start, k.start) > 0.3 * (c.end - c.start)
                       for k in chosen)
        if not overlaps:
            chosen.append(c)
        if len(chosen) == top_n:
            break
    return sorted(chosen, key=lambda x: x.start)


def find_highlights(path: str, top_n: int = 8, prefilter: int = 40):
    print("Transcribing...")
    segments = transcribe(path)
    print("Analyzing audio...")
    rms = audio_energy(path)
    cands = build_windows(segments, rms)
    # cheap prefilter so the LLM only sees the most promising windows (saves cost)
    pool = sorted(cands, key=lambda c: c.energy, reverse=True)[:prefilter]
    print(f"Scoring {len(pool)} of {len(cands)} candidate windows with LLM...")
    score_with_llm(pool)
    return pick_top(pool, top_n)


def to_trend_segments(cands):
    """Bridge to trends.rank_segments()."""
    from trends import Segment
    return [Segment(id=c.id, text=c.text, hook_score=c.hook_score) for c in cands]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="clips.json")
    args = ap.parse_args()
    results = find_highlights(args.video, args.top)
    with open(args.out, "w") as f:
        json.dump([asdict(c) for c in results], f, indent=2)
    for c in results:
        print(f"[{c.start:7.1f}-{c.end:7.1f}] {c.hook_score:.2f}  {c.title}  | {c.reason}")
    print(f"\nSaved {len(results)} clips to {args.out}")
