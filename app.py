"""
app.py - FastAPI backend tying together highlights.py + trends.py.

Install: pip install fastapi uvicorn python-multipart httpx scikit-learn faster-whisper anthropic numpy pytrends
Run:     uvicorn app:app --reload --port 8000
Env:     YOUTUBE_API_KEY, ANTHROPIC_API_KEY, APP_TOKEN (optional but recommended)
"""
import json
import re
import secrets
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

DATA = Path(os.getenv("DATA_DIR", "data"))
if os.getenv("RENDER_EXTERNAL_URL"):   # Render tells every web service its own public address
    os.environ.setdefault("PUBLIC_BASE_URL", os.environ["RENDER_EXTERNAL_URL"])
os.environ.setdefault("TIKTOK_TOKEN_FILE", str(DATA / "tiktok_token.json"))
os.environ.setdefault("YOUTUBE_TOKEN_FILE", str(DATA / "youtube_token.json"))
os.environ.setdefault("YOUTUBE_CLIENT_SECRET", str(DATA / "client_secret.json"))
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
MEDIA = DATA / "media"
MEDIA.mkdir(parents=True, exist_ok=True)
WATCH = DATA / "watch"
WATCH.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
DB_PATH = DATA / "clips.db"
TREND_TTL = 6 * 3600  # refresh trends every 6 hours
_process_lock = threading.Semaphore(1)  # one heavy job at a time (Whisper is hungry)

app = FastAPI(title="Clipper")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])


# ---------- auth (simple shared token; set APP_TOKEN in env) ----------
def require_token(authorization: Optional[str] = Header(None)):
    token = os.getenv("APP_TOKEN")
    if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "Unauthorized")





# ---------- db ----------
@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS videos(
            id TEXT PRIMARY KEY, filename TEXT, path TEXT,
            status TEXT DEFAULT 'queued', error TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS clips(
            id TEXT PRIMARY KEY, video_id TEXT, start REAL, end REAL, text TEXT,
            title TEXT, reason TEXT, hook_score REAL, energy REAL, words TEXT,
            status TEXT DEFAULT 'pending');
        CREATE TABLE IF NOT EXISTS trends(
            id INTEGER PRIMARY KEY CHECK (id = 1), fetched REAL, data TEXT);
        CREATE TABLE IF NOT EXISTS renders(
            clip_id TEXT PRIMARY KEY, status TEXT, file TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS posts(
            id TEXT PRIMARY KEY, clip_id TEXT, platform TEXT, caption TEXT, privacy TEXT,
            scheduled REAL, status TEXT DEFAULT 'scheduled', result TEXT, error TEXT, posted_at REAL);
        CREATE TABLE IF NOT EXISTS settings(
            id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT);
        """)


init_db()


# ---------- background processing ----------
def process_video(video_id: str):
    from highlights import find_highlights
    with _process_lock:
        with db() as con:
            con.execute("UPDATE videos SET status='processing' WHERE id=?", (video_id,))
            path = con.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()["path"]
        try:
            cands = find_highlights(path, top_n=12)
            with db() as con:
                for c in cands:
                    con.execute(
                        "INSERT INTO clips(id,video_id,start,end,text,title,reason,hook_score,energy,words)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (f"{video_id}:{c.id}", video_id, c.start, c.end, c.text, c.title,
                         c.reason, c.hook_score, c.energy, json.dumps(c.words)))
                con.execute("UPDATE videos SET status='done' WHERE id=?", (video_id,))
        except Exception as e:  # surface the error in the UI instead of dying silently
            with db() as con:
                con.execute("UPDATE videos SET status='failed', error=? WHERE id=?", (str(e)[:500], video_id))
    auto_process(video_id)   # outside the lock: it renders clips, which needs the lock too


# ---------- videos ----------
@app.post("/videos", dependencies=[Depends(require_token)])
def upload_video(file: UploadFile, background: BackgroundTasks):
    vid = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{vid}_{Path(file.filename).name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    with db() as con:
        con.execute("INSERT INTO videos(id,filename,path,created) VALUES(?,?,?,?)",
                    (vid, file.filename, str(dest), time.time()))
    background.add_task(process_video, vid)
    return {"id": vid, "status": "queued"}


@app.get("/videos", dependencies=[Depends(require_token)])
def list_videos():
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT id,filename,status,error,created FROM videos ORDER BY created DESC")]


@app.get("/videos/{video_id}", dependencies=[Depends(require_token)])
def get_video(video_id: str):
    with db() as con:
        r = con.execute("SELECT id,filename,status,error FROM videos WHERE id=?", (video_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    return dict(r)


# ---------- trends ----------
def get_trends(force: bool = False):
    from trends import Trend, fetch_google_trends, fetch_youtube_trends
    min_views = get_settings()["min_trend_views"]
    with db() as con:
        row = con.execute("SELECT fetched,data FROM trends WHERE id=1").fetchone()
    if row and not force and time.time() - row["fetched"] < TREND_TTL:
        cached = json.loads(row["data"])
        if isinstance(cached, dict) and cached.get("min_views") == min_views:   # refetch if the views rule changed
            return [Trend(**t) for t in cached["items"]]
    items = fetch_youtube_trends(min_views=min_views)
    if min_views == 0:          # Google Trends has no view counts, so it can't satisfy a views rule
        items += fetch_google_trends()
    with db() as con:
        con.execute("INSERT OR REPLACE INTO trends(id,fetched,data) VALUES(1,?,?)",
                    (time.time(), json.dumps({"min_views": min_views, "items": [t.__dict__ for t in items]})))
    return items


@app.post("/trends/refresh", dependencies=[Depends(require_token)])
def refresh_trends():
    items = get_trends(force=True)
    return {"count": len(items), "top": [t.topic for t in items[:20]]}


@app.get("/trends", dependencies=[Depends(require_token)])
def trends(limit: int = 30):
    return [t.__dict__ for t in get_trends()[:limit]]


# ---------- clips ----------
def _clip_out(r):
    d = dict(r)
    d.pop("words", None)  # keep list payloads small; fetch via /clips/{id}
    return d


@app.get("/clips", dependencies=[Depends(require_token)])
def list_clips(status: Optional[str] = None, video_id: Optional[str] = None):
    q, args = "SELECT * FROM clips WHERE 1=1", []
    if status:
        q += " AND status=?"; args.append(status)
    if video_id:
        q += " AND video_id=?"; args.append(video_id)
    with db() as con:
        return [_clip_out(r) for r in con.execute(q + " ORDER BY hook_score DESC", args)]


@app.get("/clips/ranked", dependencies=[Depends(require_token)])
def ranked_clips(top: int = 10):
    """Clips from your library re-ranked by what's trending right now."""
    from trends import Segment, rank_segments
    with db() as con:
        rows = {r["id"]: r for r in con.execute("SELECT * FROM clips WHERE status!='rejected'")}
    segs = [Segment(id=i, text=r["text"], hook_score=r["hook_score"]) for i, r in rows.items()]
    out = []
    for seg, score, matched in rank_segments(segs, get_trends(), top_n=top):
        d = _clip_out(rows[seg.id])
        d.update(final_score=round(score, 3), matched_trends=matched)
        out.append(d)
    return out


@app.get("/clips/{clip_id}", dependencies=[Depends(require_token)])
def get_clip(clip_id: str):
    with db() as con:
        r = con.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    d = dict(r)
    d["words"] = json.loads(d["words"] or "[]")
    return d


class ClipUpdate(BaseModel):
    status: Optional[str] = None   # pending | approved | rejected
    start: Optional[float] = None
    end: Optional[float] = None
    title: Optional[str] = None


@app.patch("/clips/{clip_id}", dependencies=[Depends(require_token)])
def update_clip(clip_id: str, body: ClipUpdate):
    fields = body.model_dump(exclude_none=True)
    if "status" in fields and fields["status"] not in {"pending", "approved", "rejected"}:
        raise HTTPException(400, "Invalid status")
    if not fields:
        raise HTTPException(400, "Nothing to update")
    sets = ", ".join(f"{k}=?" for k in fields)
    with db() as con:
        cur = con.execute(f"UPDATE clips SET {sets} WHERE id=?", (*fields.values(), clip_id))
        if cur.rowcount == 0:
            raise HTTPException(404)
    return {"ok": True}


# ---------- render ----------
def do_render(clip_id: str):
    from render import render_clip
    with _process_lock:
        with db() as con:
            r = con.execute("SELECT c.start,c.end,c.words,v.path FROM clips c JOIN videos v ON v.id=c.video_id "
                            "WHERE c.id=?", (clip_id,)).fetchone()
            con.execute("UPDATE renders SET status='rendering' WHERE clip_id=?", (clip_id,))
        fname = f"{uuid.uuid4().hex}.mp4"   # unguessable name: /media is public so Instagram can fetch it
        try:
            render_clip(r["path"], r["start"], r["end"], json.loads(r["words"] or "[]"), str(MEDIA / fname))
            with db() as con:
                con.execute("UPDATE renders SET status='done', file=? WHERE clip_id=?", (fname, clip_id))
        except Exception as e:
            with db() as con:
                con.execute("UPDATE renders SET status='failed', error=? WHERE clip_id=?", (str(e)[:500], clip_id))


@app.post("/clips/{clip_id}/render", dependencies=[Depends(require_token)])
def render_endpoint(clip_id: str, background: BackgroundTasks):
    with db() as con:
        if not con.execute("SELECT 1 FROM clips WHERE id=?", (clip_id,)).fetchone():
            raise HTTPException(404)
        con.execute("INSERT OR REPLACE INTO renders(clip_id,status) VALUES(?, 'queued')", (clip_id,))
    background.add_task(do_render, clip_id)
    return {"status": "queued"}


@app.get("/clips/{clip_id}/render", dependencies=[Depends(require_token)])
def render_status(clip_id: str):
    with db() as con:
        r = con.execute("SELECT status,file,error FROM renders WHERE clip_id=?", (clip_id,)).fetchone()
    if not r:
        return {"status": "none"}
    return {"status": r["status"], "error": r["error"],
            "url": f"/media/{r['file']}" if r["file"] else None}


@app.get("/media/{name}")
def media(name: str, dl: Optional[str] = None):
    path = MEDIA / Path(name).name
    if not path.exists():
        raise HTTPException(404)
    fname = None
    if dl:   # ?dl=Title makes the browser save it as a file called "Title.mp4"
        safe = re.sub(r"[^A-Za-z0-9 _.-]", "", dl)[:80].strip() or "clip"
        fname = safe if safe.lower().endswith(".mp4") else safe + ".mp4"
    return FileResponse(path, media_type="video/mp4", filename=fname)


# ---------- posting + scheduler ----------
class PostIn(BaseModel):
    clip_id: str
    platform: str                      # youtube | tiktok | instagram
    caption: str = ""
    privacy: str = "private"           # private | public  (start private until you've checked results)
    scheduled_at: Optional[float] = None   # unix seconds; omit to post as soon as possible


@app.post("/posts", dependencies=[Depends(require_token)])
def create_post(body: PostIn):
    if body.platform not in {"youtube", "tiktok", "instagram"}:
        raise HTTPException(400, "Unknown platform")
    with db() as con:
        r = con.execute("SELECT status FROM renders WHERE clip_id=?", (body.clip_id,)).fetchone()
        c = con.execute("SELECT status FROM clips WHERE id=?", (body.clip_id,)).fetchone()
        if not r or r["status"] != "done":
            raise HTTPException(400, "Render this clip first")
        if not c or c["status"] != "approved":
            raise HTTPException(400, "Approve this clip before scheduling it")
        pid = uuid.uuid4().hex[:10]
        con.execute("INSERT INTO posts(id,clip_id,platform,caption,privacy,scheduled) VALUES(?,?,?,?,?,?)",
                    (pid, body.clip_id, body.platform, body.caption, body.privacy,
                     body.scheduled_at or time.time()))
    return {"id": pid}


@app.get("/posts", dependencies=[Depends(require_token)])
def list_posts():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM posts ORDER BY scheduled DESC")]


@app.delete("/posts/{post_id}", dependencies=[Depends(require_token)])
def cancel_post(post_id: str):
    with db() as con:
        cur = con.execute("DELETE FROM posts WHERE id=? AND status='scheduled'", (post_id,))
        if cur.rowcount == 0:
            raise HTTPException(400, "Only scheduled posts can be cancelled")
    return {"ok": True}


def run_due_posts():
    from publish import publish
    limit = get_settings()["daily_limit"]
    due = []
    with db() as con:
        for platform in ("youtube", "tiktok", "instagram"):    # the daily limit applies to each platform separately
            used = con.execute("SELECT COUNT(*) n FROM posts WHERE platform=? AND status IN ('posting','posted') "
                               "AND posted_at>?", (platform, time.time() - 86400)).fetchone()["n"]
            remaining = max(0, limit - used)      # posts over the limit simply wait until room frees up
            if remaining == 0:
                continue
            due += [dict(r) for r in con.execute(
                "SELECT p.*, r.file, c.title FROM posts p JOIN renders r ON r.clip_id=p.clip_id "
                "JOIN clips c ON c.id=p.clip_id WHERE p.status='scheduled' AND p.scheduled<=? AND p.platform=? "
                "ORDER BY p.scheduled LIMIT ?", (time.time(), platform, remaining))]
        for p in due:
            con.execute("UPDATE posts SET status='posting', posted_at=? WHERE id=?", (time.time(), p["id"]))
    for p in due:
        try:
            result = publish(p["platform"], str(MEDIA / p["file"]), p["title"] or "Clip", p["caption"], p["privacy"])
            status, err = "posted", None
        except Exception as e:
            result, status, err = None, "failed", str(e)[:500]
        with db() as con:
            con.execute("UPDATE posts SET status=?, result=?, error=? WHERE id=?", (status, result, err, p["id"]))


def _scheduler_loop():
    while True:
        for job in (scan_watch_folder, run_due_posts):
            try:
                job()
            except Exception:
                pass   # never let the background thread die
        time.sleep(30)


@app.on_event("startup")
def start_scheduler():
    with db() as con:   # anything stuck mid-post when the server stopped is marked failed, not re-posted
        con.execute("UPDATE posts SET status='failed', error='Interrupted by restart' WHERE status='posting'")
    threading.Thread(target=_scheduler_loop, daemon=True).start()


# ---------- the website itself ----------
@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "clipper-dashboard.html").read_text(encoding="utf-8")
    return html.replace("<script>", "<script>window.SAME_ORIGIN=true;", 1)


# ---------- settings ----------
DEFAULTS = {"auto_enabled": False, "threshold": 0.75, "require_trend": True, "min_trend_views": 500000,
            "max_per_video": 5, "daily_limit": 3, "auto_post": False, "auto_post_privacy": "private", "auto_post_tiktok": False}


def get_settings():
    with db() as con:
        row = con.execute("SELECT data FROM settings WHERE id=1").fetchone()
    return {**DEFAULTS, **(json.loads(row["data"]) if row else {})}


class SettingsIn(BaseModel):
    auto_enabled: bool
    threshold: float = Field(ge=0.3, le=0.95)    # minimum clip score (0.75 = 75%)
    require_trend: bool
    min_trend_views: int = Field(500000, ge=0, le=100000000)   # only trends backed by videos with this many views
    max_per_video: int = Field(ge=1, le=20)
    daily_limit: int = Field(ge=1, le=50)
    auto_post: bool
    auto_post_privacy: str = Field("private", pattern="^(private|unlisted|public)$")
    auto_post_tiktok: bool = False


@app.get("/settings", dependencies=[Depends(require_token)])
def read_settings():
    return get_settings()


@app.put("/settings", dependencies=[Depends(require_token)])
def write_settings(body: SettingsIn):
    if body.auto_post:
        import publish
        if not publish.youtube_connected():
            raise HTTPException(400, "Connect your YouTube channel first")
    if body.auto_post_tiktok:
        import publish
        if not publish.tiktok_connected():
            raise HTTPException(400, "Connect your TikTok account first")
    with db() as con:
        con.execute("INSERT OR REPLACE INTO settings(id,data) VALUES(1,?)", (json.dumps(body.model_dump()),))
    return body.model_dump()


# ---------- auto mode ----------
def auto_process(video_id: str):
    """After a video is analysed: approve clips at/above the score threshold, render them, optionally queue uploads."""
    cfg = get_settings()
    if not cfg["auto_enabled"]:
        return
    with db() as con:
        v = con.execute("SELECT status FROM videos WHERE id=?", (video_id,)).fetchone()
        if not v or v["status"] != "done":
            return
        rows = [dict(r) for r in con.execute(
            "SELECT id,text,hook_score FROM clips WHERE video_id=? AND status='pending' AND hook_score>=? "
            "ORDER BY hook_score DESC", (video_id, cfg["threshold"]))]
    if cfg["require_trend"] and rows:
        try:
            from trends import Segment, rank_segments
            segs = [Segment(id=r["id"], text=r["text"], hook_score=r["hook_score"]) for r in rows]
            hit = {sg.id for sg, _, matched in rank_segments(segs, get_trends(), top_n=len(segs)) if matched}
            rows = [r for r in rows if r["id"] in hit]
        except Exception:
            rows = []    # can't verify trends, so approve nothing rather than approve blindly
    import publish
    platforms = []   # only connected platforms; with none connected, clips still render but nothing uploads
    if cfg["auto_post"] and publish.youtube_connected():
        platforms.append("youtube")
    if cfg["auto_post_tiktok"] and publish.tiktok_connected():
        platforms.append("tiktok")
    for r in rows[: cfg["max_per_video"]]:
        with db() as con:
            con.execute("UPDATE clips SET status='approved' WHERE id=?", (r["id"],))
            con.execute("INSERT OR REPLACE INTO renders(clip_id,status) VALUES(?, 'queued')", (r["id"],))
        do_render(r["id"])
        if platforms:
            with db() as con:
                done = con.execute("SELECT status FROM renders WHERE clip_id=?", (r["id"],)).fetchone()
                if done and done["status"] == "done":     # always private; the daily limit is applied by the scheduler
                    for plat in platforms:
                        con.execute("INSERT INTO posts(id,clip_id,platform,caption,privacy,scheduled) VALUES(?,?,?,?,?,?)",
                                    (uuid.uuid4().hex[:10], r["id"], plat, "", cfg["auto_post_privacy"], time.time()))


# ---------- watch folder ----------
_seen_sizes: dict = {}


def scan_watch_folder():
    """Videos dropped into DATA/watch are picked up once their size stops changing (i.e. the copy finished)."""
    for f in WATCH.iterdir():
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXT:
            continue
        size = f.stat().st_size
        if size == 0 or _seen_sizes.get(f) != size:
            _seen_sizes[f] = size
            continue
        _seen_sizes.pop(f, None)
        vid = uuid.uuid4().hex[:12]
        dest = UPLOADS / f"{vid}_{f.name}"
        shutil.move(str(f), dest)
        with db() as con:
            con.execute("INSERT INTO videos(id,filename,path,created) VALUES(?,?,?,?)",
                        (vid, f.name, str(dest), time.time()))
        threading.Thread(target=process_video, args=(vid,), daemon=True).start()


# ---------- YouTube sign-in ----------
def _redirect_uri(request: Request) -> str:
    base = os.getenv("PUBLIC_BASE_URL") or str(request.base_url)
    return base.rstrip("/") + "/auth/youtube/callback"


@app.get("/auth/youtube/status", dependencies=[Depends(require_token)])
def youtube_status():
    import publish
    out = {"configured": publish.youtube_configured(), "connected": publish.youtube_connected(), "channel": None}
    if out["connected"]:
        try:
            out["channel"] = publish.youtube_channel_name()
        except Exception:
            pass     # token may have expired; the dashboard will offer reconnect
    return out


@app.get("/auth/youtube/start", dependencies=[Depends(require_token)])
def youtube_start(request: Request):
    import publish
    if not publish.youtube_configured():
        raise HTTPException(400, "client_secret.json is missing from the data folder")
    return {"url": publish.youtube_auth_url(_redirect_uri(request))}


@app.get("/auth/youtube/callback")
def youtube_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None,
                     error: Optional[str] = None):
    import publish     # Google sends the browser here; `state` proves it came from a sign-in we started
    if error or not code or not state:
        return RedirectResponse("/?youtube=denied")
    try:
        publish.youtube_finish(_redirect_uri(request), state, code)
    except Exception:
        return RedirectResponse("/?youtube=failed")
    return RedirectResponse("/?youtube=connected")


@app.post("/auth/youtube/disconnect", dependencies=[Depends(require_token)])
def youtube_disconnect():
    import publish
    Path(publish.YT_TOKEN).unlink(missing_ok=True)
    return {"ok": True}


# ---------- TikTok sign-in ----------
def _tt_redirect_uri(request: Request) -> str:
    base = os.getenv("PUBLIC_BASE_URL") or str(request.base_url)
    return base.rstrip("/") + "/auth/tiktok/callback"


@app.get("/auth/tiktok/status", dependencies=[Depends(require_token)])
def tiktok_status():
    import publish
    out = {"configured": publish.tiktok_configured(), "connected": publish.tiktok_connected(), "name": None}
    if out["connected"]:
        try:
            out["name"] = publish.tiktok_username()
        except Exception:
            pass
    return out


@app.get("/auth/tiktok/start", dependencies=[Depends(require_token)])
def tiktok_start(request: Request):
    import publish
    if not publish.tiktok_configured():
        raise HTTPException(400, "TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET are not set")
    return {"url": publish.tiktok_auth_url(_tt_redirect_uri(request))}


@app.get("/auth/tiktok/callback")
def tiktok_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None,
                    error: Optional[str] = None):
    import publish
    if error or not code or not state:
        return RedirectResponse("/?tiktok=denied")
    try:
        publish.tiktok_finish(_tt_redirect_uri(request), state, code)
    except Exception:
        return RedirectResponse("/?tiktok=failed")
    return RedirectResponse("/?tiktok=connected")


@app.post("/auth/tiktok/disconnect", dependencies=[Depends(require_token)])
def tiktok_disconnect():
    import publish
    Path(publish.TT_TOKEN).unlink(missing_ok=True)
    return {"ok": True}
