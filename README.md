# Clipper

Upload (or drop in) videos you own or have permission to use. Clipper transcribes them, picks the best
moments, ranks them against current trends, renders vertical captioned clips, and can upload them to
your YouTube channel.

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render: New > Blueprint > choose the repo. No environment variables are required to start.
   Optional, add later under Environment: `YOUTUBE_API_KEY` (trend ranking) and `ANTHROPIC_API_KEY` (smarter clip scoring).
3. Open your `onrender.com` address. There is no sign-in, so anyone with the link can use the site.

### Connect YouTube
1. Google Cloud > Credentials > Create OAuth client ID > **Web application**.
2. Redirect URI: `https://YOUR-SERVICE.onrender.com/auth/youtube/callback`
3. Render > Environment > Secret Files: add `client_secret.json` with the downloaded JSON, then redeploy.
4. In the site, open the Auto mode tab and click Connect YouTube.

### Connect TikTok
1. developers.tiktok.com > create an app > add **Login Kit** and **Content Posting API** (enable Direct Post).
2. Scopes: `user.info.basic`, `video.publish`. Redirect URI: `https://YOUR-SERVICE.onrender.com/auth/tiktok/callback`
3. Put the app's Client key and Client secret in Render as `TIKTOK_CLIENT_KEY` and `TIKTOK_CLIENT_SECRET`, then redeploy.
4. In the site, open the Auto mode tab and click Connect TikTok.
Until TikTok audits your app, every post it makes is private (only you can see it).

Notes: YouTube forces uploads from unaudited API projects to private. While the Google project is in
Testing mode the sign-in expires after about 7 days.

## Run locally
    pip install -r requirements.txt        # and install ffmpeg
    export YOUTUBE_API_KEY=... ANTHROPIC_API_KEY=...
    uvicorn app:app --port 8000            # open http://localhost:8000

## Files
- `app.py` web server, API, scheduler, watch folder, auto mode
- `highlights.py` transcription and clip scoring
- `trends.py` trending topics and matching
- `render.py` cutting, 9:16 crop, captions
- `publish.py` YouTube / TikTok / Instagram uploads
- `clipper-dashboard.html` the website
