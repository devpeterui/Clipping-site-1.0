"""
render.py - turn a clip (start/end/words) into a finished 1080x1920 video with captions.

Needs: ffmpeg + ffprobe on PATH. Optional: pip install opencv-python  (face-tracked crop)
"""
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

OUT_W, OUT_H = 1080, 1920


def probe(path: str):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def face_center_x(path: str, start: float, end: float, width: int) -> int:
    """Median horizontal face position over the clip. Falls back to center if no face/OpenCV."""
    try:
        import cv2
    except ImportError:
        return width // 2
    cap = cv2.VideoCapture(path)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs, t = [], start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces):
                x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])  # biggest face
                xs.append(x + w / 2)
        t += 0.75
    cap.release()
    return int(statistics.median(xs)) if xs else width // 2


def _ts(t: float) -> str:
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(words, clip_start: float, clip_end: float, chunk: int = 3) -> str:
    """Word-by-word captions: groups of 3 words, the spoken word highlighted in yellow."""
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,"
        "Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Cap,Arial,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,80,80,420,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ) % (OUT_W, OUT_H)
    ws = [w for w in words if clip_start - 0.05 <= w["s"] <= clip_end]
    lines = []
    for i in range(0, len(ws), chunk):
        group = ws[i:i + chunk]
        for k, w in enumerate(group):
            s = w["s"] - clip_start
            e = (group[k + 1]["s"] if k + 1 < len(group) else w["e"]) - clip_start
            text = " ".join(
                (r"{\c&H00FFFF&}" + g["w"].upper() + r"{\c&HFFFFFF&}") if j == k else g["w"].upper()
                for j, g in enumerate(group))
            lines.append(f"Dialogue: 0,{_ts(s)},{_ts(max(e, s + 0.1))},Cap,,0,0,0,,{text}")
    return head + "\n".join(lines) + "\n"


def render_clip(src: str, start: float, end: float, words, out: str, captions: bool = True):
    w, h = probe(src)
    crop_w = min(w, int(h * 9 / 16))
    cx = face_center_x(src, start, end, w)
    x = max(0, min(w - crop_w, cx - crop_w // 2))
    vf = f"crop={crop_w}:{h}:{x}:0,scale={OUT_W}:{OUT_H}:flags=lanczos"
    with tempfile.TemporaryDirectory() as tmp:
        if captions and words:
            ass = Path(tmp) / "cap.ass"
            ass.write_text(build_ass(words, start, end), encoding="utf-8")
            vf += f",subtitles={ass.as_posix().replace(':', r'\:')}"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-to", str(end), "-i", src,
            "-vf", vf, "-af", "loudnorm=I=-16:TP=-1.5",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out,
        ], check=True)
    return out
