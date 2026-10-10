"""
publish.py - upload finished clips through each platform's OFFICIAL API.

Setup (once per platform) - credentials come from environment variables, never from code:
  YouTube : pip install google-api-python-client google-auth-oauthlib
            Create an OAuth "Desktop app" client in Google Cloud, download it as client_secret.json,
            then run:  python publish.py auth-youtube     (opens a browser, saves youtube_token.json)
  TikTok  : Register an app at developers.tiktok.com, enable "Content Posting API", then set
            TIKTOK_ACCESS_TOKEN (user token with video.publish scope). Until TikTok audits your app,
            posts can only be SELF_ONLY (private).
  Instagram: Business/Creator account linked to a Facebook Page. Set IG_USER_ID, IG_ACCESS_TOKEN and
            PUBLIC_BASE_URL (a public HTTPS address where /media/<file> is reachable; Instagram
            downloads the video from that URL).

Every uploader returns a string id/url and raises on failure.
"""
import json
import os
import secrets
import sys
import time
from urllib.parse import urlencode
from pathlib import Path

import httpx

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
             "https://www.googleapis.com/auth/youtube.readonly"]
YT_TOKEN = os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")


# ---------------- YouTube (Shorts) ----------------
_pending = {}   # OAuth state -> (timestamp, PKCE verifier) for sign-ins in progress


def _client_file():
    return os.getenv("YOUTUBE_CLIENT_SECRET", "client_secret.json")


def youtube_configured() -> bool:
    return Path(_client_file()).exists()


def youtube_connected() -> bool:
    return Path(YT_TOKEN).exists()


def _flow(redirect_uri: str):
    from google_auth_oauthlib.flow import Flow
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    if redirect_uri.startswith(("http://localhost", "http://127.0.0.1")):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"      # plain http is only OK for local testing
    return Flow.from_client_secrets_file(_client_file(), scopes=YT_SCOPES, redirect_uri=redirect_uri)


def youtube_auth_url(redirect_uri: str) -> str:
    flow = _flow(redirect_uri)
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _pending[state] = (time.time(), getattr(flow, "code_verifier", None))
    return url


def youtube_finish(redirect_uri: str, state: str, code: str):
    entry = _pending.pop(state, None)
    if not entry or time.time() - entry[0] > 900:
        raise ValueError("Sign-in expired or invalid. Start again.")
    flow = _flow(redirect_uri)
    if entry[1]:
        flow.code_verifier = entry[1]
    flow.fetch_token(code=code)
    Path(YT_TOKEN).write_text(flow.credentials.to_json())


def _load_creds():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(YT_TOKEN, YT_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        Path(YT_TOKEN).write_text(creds.to_json())
    return creds


def youtube_channel_name():
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", credentials=_load_creds())
    items = yt.channels().list(part="snippet", mine=True).execute().get("items", [])
    return items[0]["snippet"]["title"] if items else None


def youtube_auth():
    """Command-line alternative: python publish.py auth-youtube (needs a Desktop-type OAuth client)."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(_client_file(), YT_SCOPES)
    Path(YT_TOKEN).write_text(flow.run_local_server(port=0).to_json())
    print("Saved", YT_TOKEN)


def upload_youtube(path: str, title: str, description: str = "", privacy: str = "private") -> str:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    yt = build("youtube", "v3", credentials=_load_creds())
    body = {"snippet": {"title": (title or "Clip")[:90] + " #Shorts", "description": description[:4900],
                        "categoryId": "22"},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(path, chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return f"https://youtube.com/shorts/{resp['id']}"


# ---------------- TikTok (Login Kit + Content Posting API) ----------------
TT_TOKEN = os.getenv("TIKTOK_TOKEN_FILE", "tiktok_token.json")
TT_API = "https://open.tiktokapis.com"
_tt_pending = {}   # OAuth state -> timestamp for sign-ins in progress


def tiktok_configured() -> bool:
    return bool(os.getenv("TIKTOK_CLIENT_KEY") and os.getenv("TIKTOK_CLIENT_SECRET"))


def tiktok_connected() -> bool:
    return Path(TT_TOKEN).exists() or bool(os.getenv("TIKTOK_ACCESS_TOKEN"))


def tiktok_auth_url(redirect_uri: str) -> str:
    state = secrets.token_urlsafe(24)
    _tt_pending[state] = time.time()
    return "https://www.tiktok.com/v2/auth/authorize/?" + urlencode({
        "client_key": os.environ["TIKTOK_CLIENT_KEY"], "scope": "user.info.basic,video.publish",
        "response_type": "code", "redirect_uri": redirect_uri, "state": state})


def _tt_token_request(**fields):
    r = httpx.post(f"{TT_API}/v2/oauth/token/", timeout=30, data={
        "client_key": os.environ["TIKTOK_CLIENT_KEY"], "client_secret": os.environ["TIKTOK_CLIENT_SECRET"], **fields})
    data = r.json()
    if "access_token" not in data:
        raise ValueError(data.get("error_description") or "TikTok sign-in failed")
    data["expires_at"] = time.time() + int(data.get("expires_in", 86400))
    Path(TT_TOKEN).write_text(json.dumps(data))
    return data


def tiktok_finish(redirect_uri: str, state: str, code: str):
    started = _tt_pending.pop(state, None)
    if not started or time.time() - started > 900:
        raise ValueError("Sign-in expired or invalid. Start again.")
    _tt_token_request(code=code, grant_type="authorization_code", redirect_uri=redirect_uri)


def _tiktok_access_token() -> str:
    if not Path(TT_TOKEN).exists():
        return os.environ["TIKTOK_ACCESS_TOKEN"]          # manual token fallback
    d = json.loads(Path(TT_TOKEN).read_text())
    if time.time() > d["expires_at"] - 300:               # access tokens last about 24 hours
        d = _tt_token_request(grant_type="refresh_token", refresh_token=d["refresh_token"])
    return d["access_token"]


def tiktok_username():
    r = httpx.get(f"{TT_API}/v2/user/info/", params={"fields": "display_name"}, timeout=30,
                  headers={"Authorization": f"Bearer {_tiktok_access_token()}"})
    r.raise_for_status()
    return r.json().get("data", {}).get("user", {}).get("display_name")


def upload_tiktok(path: str, title: str, privacy: str = "SELF_ONLY") -> str:
    h = {"Authorization": f"Bearer {_tiktok_access_token()}", "Content-Type": "application/json; charset=UTF-8"}
    # Ask TikTok which visibility options this account may use. Apps TikTok hasn't audited only get SELF_ONLY.
    info = httpx.post(f"{TT_API}/v2/post/publish/creator_info/query/", headers=h, timeout=30)
    info.raise_for_status()
    options = info.json().get("data", {}).get("privacy_level_options", [])
    if options and privacy not in options:
        privacy = "SELF_ONLY" if "SELF_ONLY" in options else options[0]
    size = os.path.getsize(path)
    chunk = min(size, 10 * 1024 * 1024)
    total = max(1, size // chunk)          # last chunk absorbs the remainder
    init = httpx.post(f"{TT_API}/v2/post/publish/video/init/", headers=h, timeout=60, json={
        "post_info": {"title": title[:150], "privacy_level": privacy,
                      "disable_duet": False, "disable_comment": False, "disable_stitch": False},
        "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                        "chunk_size": chunk, "total_chunk_count": total}})
    init.raise_for_status()
    data = init.json()["data"]
    with open(path, "rb") as f:
        for i in range(total):
            start = i * chunk
            blob = f.read(chunk if i < total - 1 else size - start)
            r = httpx.put(data["upload_url"], content=blob, timeout=300, headers={
                "Content-Type": "video/mp4", "Content-Length": str(len(blob)),
                "Content-Range": f"bytes {start}-{start + len(blob) - 1}/{size}"})
            r.raise_for_status()
    return f"tiktok:{data['publish_id']}"


# ---------------- Instagram Reels (Graph API) ----------------
def upload_instagram(video_url: str, caption: str) -> str:
    ig, token = os.environ["IG_USER_ID"], os.environ["IG_ACCESS_TOKEN"]
    base = f"https://graph.facebook.com/v21.0/{ig}"
    c = httpx.post(f"{base}/media", timeout=60, data={
        "media_type": "REELS", "video_url": video_url, "caption": caption[:2000], "access_token": token})
    c.raise_for_status()
    cid = c.json()["id"]
    for _ in range(40):                      # wait for Instagram to finish processing
        st = httpx.get(f"https://graph.facebook.com/v21.0/{cid}", timeout=30,
                       params={"fields": "status_code", "access_token": token}).json()
        if st.get("status_code") == "FINISHED":
            break
        if st.get("status_code") == "ERROR":
            raise RuntimeError("Instagram failed to process the video")
        time.sleep(5)
    p = httpx.post(f"{base}/media_publish", timeout=60, data={"creation_id": cid, "access_token": token})
    p.raise_for_status()
    return f"instagram:{p.json()['id']}"


def publish(platform: str, path: str, title: str, caption: str = "", privacy: str = "private") -> str:
    if platform == "youtube":
        return upload_youtube(path, title, caption, privacy)
    if platform == "tiktok":
        return upload_tiktok(path, f"{title} {caption}".strip(),
                             "PUBLIC_TO_EVERYONE" if privacy == "public" else "SELF_ONLY")
    if platform == "instagram":
        return upload_instagram(f"{os.environ['PUBLIC_BASE_URL'].rstrip('/')}/media/{Path(path).name}",
                                f"{title}\n\n{caption}".strip())
    raise ValueError(f"Unknown platform: {platform}")


if __name__ == "__main__":
    if sys.argv[1:] == ["auth-youtube"]:
        youtube_auth()
    else:
        print(__doc__)
