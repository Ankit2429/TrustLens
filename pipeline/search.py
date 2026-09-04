"""Multi-source / multi-platform visual search via SerpApi's Google Lens engine.

Discovers indexed public web and social media candidates matching the query face image.
Classifies candidates by domain/platform (Instagram, LinkedIn, Facebook, X, Reddit,
YouTube, TikTok, Pinterest, News/Media, General Web).
"""
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

import requests

SERPAPI_URL = "https://serpapi.com/search"

# Known platform domain mapping
PLATFORM_DOMAIN_PATTERNS = [
    (r"(?:^|\.)instagram\.com$", "Instagram"),
    (r"(?:^|\.)linkedin\.com$", "LinkedIn"),
    (r"(?:^|\.)(?:facebook\.com|fb\.com|fb\.watch)$", "Facebook"),
    (r"(?:^|\.)(?:twitter\.com|x\.com|t\.co)$", "X (Twitter)"),
    (r"(?:^|\.)reddit\.com$", "Reddit"),
    (r"(?:^|\.)(?:youtube\.com|youtu\.be)$", "YouTube"),
    (r"(?:^|\.)tiktok\.com$", "TikTok"),
    (r"(?:^|\.)(?:pinterest\.com|pinimg\.com)$", "Pinterest"),
    (
        r"(?:^|\.)(?:bbc\.(?:com|co\.uk)|cnn\.com|nytimes\.com|theguardian\.com|reuters\.com|"
        r"forbes\.com|bloomberg\.com|techcrunch\.com|medium\.com|ndtv\.com|indiatimes\.com|"
        r"hindustantimes\.com|washingtonpost\.com|wsj\.com)$",
        "News / Media",
    ),
]


def extract_domain(url: str) -> str:
    """Extract clean lowercased domain name from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # Strip port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Strip leading 'www.'
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def classify_platform(domain: str) -> str:
    """Classify domain name into a human-readable platform or category."""
    if not domain:
        return "Unknown"
    for pattern, platform_name in PLATFORM_DOMAIN_PATTERNS:
        if re.search(pattern, domain, re.IGNORECASE):
            return platform_name
    return "General Web"


def reverse_image_search(public_image_url: str) -> list[dict[str, Any]]:
    """Query Google Lens (via SerpApi) with a publicly reachable image URL.

    Args:
        public_image_url: Public gateway URL of the query image.

    Returns:
        List of structured, classified, and deduplicated candidate dictionaries.

    Raises:
        EnvironmentError: If SERPAPI_KEY is not configured.
        RuntimeError: If SerpApi returns an API error or network failure.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key or api_key == "your_serpapi_key":
        raise EnvironmentError(
            "Missing or unconfigured SERPAPI_KEY. "
            "Please set your valid SerpApi key in the .env file."
        )

    params = {
        "engine": "google_lens",
        "url": public_image_url,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=35)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error communicating with SerpApi: {e}") from e

    if resp.status_code != 200:
        error_msg = f"SerpApi HTTP {resp.status_code}"
        try:
            err_json = resp.json()
            if "error" in err_json:
                error_msg += f": {err_json['error']}"
        except Exception:
            error_msg += f": {resp.text[:200]}"
        raise RuntimeError(error_msg)

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"SerpApi Error: {data['error']}")

    raw_matches = data.get("visual_matches", [])

    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for idx, match in enumerate(raw_matches, start=1):
        link = match.get("link", "").strip()
        if not link:
            continue

        # Intelligent URL deduplication
        norm_link = link.rstrip("/").lower()
        if norm_link in seen_urls:
            continue
        seen_urls.add(norm_link)

        domain = extract_domain(link)
        platform = classify_platform(domain)

        candidate = {
            "search_rank": idx,
            "title": match.get("title", "").strip() or f"Discovered Content #{idx}",
            "link": link,
            "domain": domain,
            "platform": platform,
            "thumbnail": match.get("thumbnail"),
            "source": match.get("source", domain or "Web"),
        }
        candidates.append(candidate)

    return candidates


def filter_social_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter candidate matches to prioritize recognized social media platforms."""
    social_platforms = {"Instagram", "LinkedIn", "Facebook", "X (Twitter)", "Reddit", "YouTube", "TikTok", "Pinterest"}
    return [m for m in matches if m.get("platform") in social_platforms]

