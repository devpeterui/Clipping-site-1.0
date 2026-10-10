"""
trends.py - trend discovery + matching against YOUR rights-cleared clip library.

Install: pip install httpx pytrends scikit-learn
Env:     YOUTUBE_API_KEY=...
"""
import os
import re
from collections import Counter
from dataclasses import dataclass

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

YT_URL = "https://www.googleapis.com/youtube/v3/videos"
DEFAULT_REGIONS = ["US", "GB", "KE", "NG", "ZA", "IN", "BR", "JP", "DE", "FR"]

STOPWORDS = set(
    "the a an and or of to in on for with is are was be this that it you your "
    "i we they official video new full hd 4k ft feat vs part episode".split()
)


@dataclass
class Trend:
    topic: str          # keyword/phrase
    score: float        # weighted popularity
    regions: list[str]  # where it's trending
    source: str         # "youtube" | "google_trends"


@dataclass
class Segment:
    """A highlight from YOUR library (output of the transcribe + highlight step)."""
    id: str
    text: str
    hook_score: float   # 0-1 from your LLM / audio-energy scoring


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def fetch_youtube_trends(regions=None, per_region=50, min_views=0) -> list[Trend]:
    """Most-popular videos per region -> keyword frequency, weighted by rank.
    Videos with fewer than min_views views are ignored, so only strong trends count."""
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return []          # no key set: the site still works, just without trend ranking
    regions = regions or DEFAULT_REGIONS
    scores, seen_in = Counter(), {}
    with httpx.Client(timeout=15) as client:
        for region in regions:
            r = client.get(YT_URL, params={
                "part": "snippet,statistics", "chart": "mostPopular",
                "regionCode": region, "maxResults": per_region, "key": key,
            })
            if r.status_code != 200:
                continue
            for rank, item in enumerate(r.json().get("items", [])):
                views = int(item.get("statistics", {}).get("viewCount", 0))
                if views < min_views:
                    continue
                weight = 1 - rank / per_region          # higher rank = more weight
                sn = item["snippet"]
                text = f"{sn['title']} {' '.join(sn.get('tags', [])[:10])}"
                for tok in set(_tokens(text)):
                    scores[tok] += weight
                    seen_in.setdefault(tok, set()).add(region)
    return [Trend(t, s, sorted(seen_in[t]), "youtube")
            for t, s in scores.most_common(100)]


def fetch_google_trends(geo="US") -> list[Trend]:
    """Daily trending searches (best-effort; pytrends is unofficial and can break)."""
    try:
        from pytrends.request import TrendReq
        df = TrendReq(hl="en-US").trending_searches(pn="united_states")
        return [Trend(str(t).lower(), 1.0 - i / len(df), [geo], "google_trends")
                for i, t in enumerate(df[0].tolist())]
    except Exception:
        return []


def rank_segments(segments: list[Segment], trends: list[Trend], top_n=10):
    """
    Final score = 0.6 * trend relevance + 0.4 * hook quality.
    Returns [(segment, final_score, matched_topics)] for the review queue.
    """
    if not segments or not trends:
        return []
    trend_docs = [t.topic for t in trends]
    vec = TfidfVectorizer(stop_words="english").fit(
        trend_docs + [s.text for s in segments])
    seg_m = vec.transform([s.text for s in segments])
    trend_m = vec.transform(trend_docs)
    sim = cosine_similarity(seg_m, trend_m)             # segments x trends
    weights = [t.score for t in trends]
    max_w = max(weights) or 1.0

    results = []
    for i, seg in enumerate(segments):
        weighted = [(sim[i][j] * weights[j] / max_w, j) for j in range(len(trends))]
        weighted.sort(reverse=True)
        relevance = sum(w for w, _ in weighted[:3])
        matched = [trends[j].topic for w, j in weighted[:3] if w > 0]
        results.append((seg, 0.6 * min(relevance, 1.0) + 0.4 * seg.hook_score, matched))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


if __name__ == "__main__":
    trends = fetch_youtube_trends() + fetch_google_trends()
    print("Top trending topics:", [t.topic for t in trends[:15]])
