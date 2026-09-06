"""Multi-source / multi-platform visual search via SerpApi's Google Lens engine.

Discovers indexed public web and social media candidates matching the query face image.
Implements deep visual search with pagination, multi-category extraction (visual matches,
exact matches, reverse image search), bounded source page expansion, and robust deduplication.
Classifies candidates by domain/platform (Instagram, LinkedIn, Facebook, X, Reddit,
YouTube, TikTok, Pinterest, News/Media, General Web).
"""
import concurrent.futures
import os
import re
import time
from typing import Any, Optional, Union
from urllib.parse import urlparse, urljoin

import requests

SERPAPI_URL = "https://serpapi.com/search"

# Configurable search depth controls
DEFAULT_SEARCH_MAX_PAGES = int(os.environ.get("SEARCH_MAX_PAGES", "3"))
DEFAULT_SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "60"))
DEFAULT_SEARCH_MAX_CANDIDATE_IMAGES = int(os.environ.get("SEARCH_MAX_CANDIDATE_IMAGES", "40"))
DEFAULT_SEARCH_MAX_SOURCE_EXPANSIONS = int(os.environ.get("SEARCH_MAX_SOURCE_EXPANSIONS", "8"))
DEFAULT_SEARCH_CONCURRENCY = int(os.environ.get("SEARCH_CONCURRENCY", "8"))

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
        if ":" in netloc:
            netloc = netloc.split(":")[0]
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


def normalize_url(url: str) -> str:
    """Normalize URL for strict deduplication."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        return f"{netloc}{path}"
    except Exception:
        return url.strip().lower().rstrip("/")


def _expand_source_page_image(candidate: dict[str, Any], session: requests.Session) -> Optional[dict[str, Any]]:
    """Inspect candidate source page HTML to extract high-resolution og:image / twitter:image.
    
    Returns an expanded candidate entry if a distinct high-resolution image is found.
    """
    link = candidate.get("link")
    if not link or not link.startswith(("http://", "https://")):
        return None
    
    # Avoid crawling heavy social media SPA entrypoints that block anonymous requests
    domain = candidate.get("domain", "")
    if any(s in domain for s in ["instagram.com", "facebook.com", "tiktok.com"]):
        return None

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = session.get(link, headers=headers, timeout=4, stream=False)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            html = resp.text[:100000]  # Limit read to first 100KB for speed
            
            # Find og:image or twitter:image
            og_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if not og_match:
                og_match = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html, re.IGNORECASE)
            if not og_match:
                og_match = re.search(r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)

            if og_match:
                img_url = og_match.group(1).strip()
                if img_url.startswith("/"):
                    img_url = urljoin(link, img_url)
                if img_url.startswith("http") and img_url != candidate.get("thumbnail"):
                    # Create an expanded candidate linking directly to the high-resolution source image
                    return {
                        "search_rank": candidate["search_rank"],
                        "title": f"[Source Image] {candidate.get('title', '')}",
                        "link": link,
                        "domain": candidate.get("domain"),
                        "platform": candidate.get("platform"),
                        "thumbnail": img_url,
                        "source": candidate.get("source"),
                        "discovery_timestamp": candidate.get("discovery_timestamp"),
                        "category": "source_page_expansion",
                        "is_source_expansion": True,
                    }
    except Exception:
        pass
    return None


def reverse_image_search(
    public_image_url: str,
    return_telemetry: bool = False,
    max_pages: Optional[int] = None,
    max_results: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Union[list[dict[str, Any]], tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Query Google Lens (via SerpApi) with deep pagination and multi-category discovery.

    Args:
        public_image_url: Public gateway URL of the query image.
        return_telemetry: If True, returns (candidates, telemetry_dict).
        max_pages: Maximum number of pagination pages to scan (default: SEARCH_MAX_PAGES).
        max_results: Maximum unique candidate results to retain (default: SEARCH_MAX_RESULTS).
        max_expansions: Maximum source pages to inspect for high-res images.

    Returns:
        List of structured, classified, and deduplicated candidate dictionaries,
        or (candidates, telemetry_dict) if return_telemetry is True.

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

    limit_pages = max_pages if max_pages is not None else DEFAULT_SEARCH_MAX_PAGES
    limit_results = max_results if max_results is not None else DEFAULT_SEARCH_MAX_RESULTS
    limit_expansions = max_expansions if max_expansions is not None else DEFAULT_SEARCH_MAX_SOURCE_EXPANSIONS

    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_images: set[str] = set()
    platforms_discovered: set[str] = set()
    categories_discovered: set[str] = set()

    req_start = int(time.time())
    pages_scanned = 0
    raw_results_count = 0
    next_page_url: Optional[str] = None
    next_page_token: Optional[str] = None

    session = requests.Session()

    for page_idx in range(1, limit_pages + 1):
        if len(candidates) >= limit_results:
            break

        if page_idx == 1:
            # Use type=visual_matches for the richest result set and pagination support.
            # SerpApi Google Lens returns serpapi_pagination.next when more pages exist.
            params = {
                "engine": "google_lens",
                "url": public_image_url,
                "api_key": api_key,
                "type": "visual_matches",
            }
            req_url = SERPAPI_URL
        elif next_page_url:
            # SerpApi pre-constructs the full next-page URL; we only need to append the api_key
            req_url = next_page_url
            params = {"api_key": api_key}
        elif next_page_token:
            req_url = SERPAPI_URL
            params = {
                "engine": "google_lens",
                "url": public_image_url,
                "api_key": api_key,
                "next_page_token": next_page_token,
                "type": "visual_matches",
            }
        else:
            # SerpApi did not return a continuation token — no further pages available.
            # Report the honest page count rather than fabricating additional depth.
            break

        try:
            resp = session.get(req_url, params=params, timeout=35)
        except requests.RequestException as e:
            if page_idx == 1:
                raise RuntimeError(f"Network error communicating with SerpApi: {e}") from e
            # If a subsequent pagination page times out, retain what was collected
            break

        if resp.status_code != 200:
            if page_idx == 1:
                error_msg = f"SerpApi HTTP {resp.status_code}"
                try:
                    err_json = resp.json()
                    if "error" in err_json:
                        error_msg += f": {err_json['error']}"
                except Exception:
                    error_msg += f": {resp.text[:200]}"
                raise RuntimeError(error_msg)
            break

        data = resp.json()
        if "error" in data:
            if page_idx == 1:
                raise RuntimeError(f"SerpApi Error: {data['error']}")
            break

        pages_scanned += 1

        # Multi-category result extraction.
        # 1. SerpApi Google Lens "reverse_image_search" can be a list of matches,
        # or a dict containing an inline_images list.
        ris_raw = data.get("reverse_image_search")
        ris_items: list[dict[str, Any]] = []
        if isinstance(ris_raw, list):
            ris_items = ris_raw
        elif isinstance(ris_raw, dict) and ris_raw.get("inline_images"):
            for img in ris_raw["inline_images"]:
                if isinstance(img, dict) and (img.get("link") or img.get("thumbnail")):
                    ris_items.append({
                        "link": img.get("link", ""),
                        "thumbnail": img.get("thumbnail", ""),
                        "title": img.get("title", ""),
                        "source": img.get("source", ""),
                    })

        # 2. Knowledge Graph and About This Image context
        kg = data.get("knowledge_graph", {})
        about_img = data.get("about_this_image", {})
        about_results: list[dict[str, Any]] = []
        if isinstance(kg, dict) and (kg.get("title") or kg.get("link") or kg.get("header_images")):
            kg_thumb = ""
            if kg.get("header_images") and isinstance(kg["header_images"], list) and len(kg["header_images"]) > 0:
                kg_thumb = kg["header_images"][0].get("image", "")
            about_results.append({
                "title": f"[Knowledge Graph] {kg.get('title', '')}: {kg.get('subtitle', '')}".strip(),
                "link": kg.get("link", ""),
                "thumbnail": kg_thumb,
                "source": kg.get("source", {}).get("name", "Knowledge Graph") if isinstance(kg.get("source"), dict) else "Knowledge Graph",
                "snippet": kg.get("description", ""),
            })
        if isinstance(about_img, dict) and about_img.get("images"):
            for ai in about_img["images"]:
                if isinstance(ai, dict):
                    about_results.append({
                        "title": f"[About This Image] {ai.get('title', 'Related Context')}",
                        "link": ai.get("link", ""),
                        "thumbnail": ai.get("thumbnail", ""),
                        "source": ai.get("source", "About This Image"),
                        "snippet": ai.get("snippet", ""),
                    })

        search_req_id = data.get("search_metadata", {}).get("id")

        categories_to_check = [
            ("visual_matches", data.get("visual_matches", [])),
            ("exact_matches", data.get("exact_matches", [])),
            ("reverse_image_search", ris_items),
            ("about_this_image", about_results),
            ("images_results", data.get("images_results", [])),
        ]

        new_candidates_on_page = 0

        for cat_name, raw_matches in categories_to_check:
            if not isinstance(raw_matches, list) or not raw_matches:
                continue
            categories_discovered.add(cat_name)

            for match in raw_matches:
                raw_results_count += 1
                link = match.get("link", "").strip()
                thumb = match.get("thumbnail", "").strip()
                if not link and not thumb:
                    continue

                norm_link = normalize_url(link) if link else f"thumb:{thumb[:50]}"
                if norm_link in seen_urls:
                    continue
                seen_urls.add(norm_link)

                if thumb:
                    seen_images.add(thumb)

                domain = extract_domain(link)
                platform = classify_platform(domain)
                platforms_discovered.add(platform)

                candidate_entry = {
                    "search_rank": len(candidates) + 1,
                    "title": match.get("title", "").strip() or f"Discovered Content #{len(candidates) + 1}",
                    "link": link,
                    "domain": domain,
                    "platform": platform,
                    "thumbnail": thumb or None,
                    "source": match.get("source", domain or "Web"),
                    "snippet": match.get("snippet", "").strip(),
                    "category": cat_name,
                    "search_mode": cat_name,
                    "page": page_idx,
                    "discovery_timestamp": req_start,
                    "search_request_id": search_req_id,
                }
                candidates.append(candidate_entry)
                new_candidates_on_page += 1

                if len(candidates) >= limit_results:
                    break

            if len(candidates) >= limit_results:
                break

        # Check pagination continuation
        pagination = data.get("serpapi_pagination", {})
        next_page_url = pagination.get("next")
        next_page_token = pagination.get("next_page_token")

        if not next_page_url and not next_page_token:
            break
        if new_candidates_on_page == 0:
            # Diminishing returns; avoid redundant calls
            break

    # Bounded source page expansion for top unique results
    expansions_done = 0
    if limit_expansions > 0 and candidates:
        to_expand = candidates[:limit_expansions]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, limit_expansions)) as pool:
            futures = [pool.submit(_expand_source_page_image, c, session) for c in to_expand]
            for f in concurrent.futures.as_completed(futures):
                try:
                    exp_candidate = f.result()
                    if exp_candidate and exp_candidate.get("thumbnail"):
                        thumb = exp_candidate["thumbnail"]
                        if thumb not in seen_images:
                            seen_images.add(thumb)
                            exp_candidate["search_rank"] = len(candidates) + 1
                            candidates.append(exp_candidate)
                            expansions_done += 1
                except Exception:
                    pass

    # Count search mode breakdowns
    exact_matches_count = sum(1 for c in candidates if c.get("category") == "exact_matches")
    visual_matches_count = sum(1 for c in candidates if c.get("category") in ("visual_matches", "reverse_image_search", "images_results", "source_page_expansion"))
    about_image_count = sum(1 for c in candidates if c.get("category") == "about_this_image")

    active_modes = set()
    if exact_matches_count > 0:
        active_modes.add("exact_matches")
    if visual_matches_count > 0:
        active_modes.add("visual_matches")
    if about_image_count > 0:
        active_modes.add("about_this_image")

    telemetry = {
        "pages_scanned": pages_scanned,
        "total_discovered": raw_results_count,
        "unique_candidates": len(candidates),
        "platforms_discovered": sorted(list(platforms_discovered)),
        "source_expansions": expansions_done,
        "categories_scanned": sorted(list(categories_discovered)),
        "search_modes_queried": ["visual_matches", "exact_matches", "about_this_image"],
        "search_modes_active": sorted(list(active_modes)),
        "exact_matches_count": exact_matches_count,
        "visual_matches_count": visual_matches_count,
        "about_image_count": about_image_count,
        "search_request_id": search_req_id,
    }

    if return_telemetry:
        return candidates, telemetry
    return candidates


def extract_image_metadata(image_path: Union[str, os.PathLike]) -> dict[str, Any]:
    """Inspect and extract camera and non-sensitive image metadata (EXIF).

    Inspects:
    - Dimensions (width, height)
    - Camera Make & Model
    - DateTime Original
    - Orientation
    - Format (JPEG, PNG, etc.)

    Returns:
        Structured dictionary containing metadata fields and 'metadata_status':
        'METADATA AVAILABLE' or 'NO METADATA'.
    """
    if not os.path.exists(str(image_path)):
        return {
            "metadata_status": "NO METADATA",
            "has_exif": False,
            "dimensions": None,
            "format": None,
        }

    try:
        from PIL import Image, ExifTags

        with Image.open(str(image_path)) as img:
            w, h = img.size
            fmt = img.format
            exif_raw = img.getexif()
            has_exif = bool(exif_raw and len(exif_raw) > 0)

            make = None
            model = None
            dt_orig = None
            orientation = None

            if has_exif:
                for tag_id, val in exif_raw.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make" and val:
                        make = str(val).strip()
                    elif tag_name == "Model" and val:
                        model = str(val).strip()
                    elif tag_name in ("DateTimeOriginal", "DateTime") and val:
                        dt_orig = str(val).strip()
                    elif tag_name == "Orientation" and val:
                        try:
                            orientation = int(val)
                        except (ValueError, TypeError):
                            pass

            has_camera_data = bool(make or model or dt_orig)
            status = "METADATA AVAILABLE" if (has_exif and has_camera_data) else "NO METADATA"

            return {
                "metadata_status": status,
                "status": status,
                "has_exif": has_exif,
                "dimensions": f"{w}x{h}",
                "width": w,
                "height": h,
                "format": fmt,
                "camera_make": make,
                "camera_model": model,
                "datetime_original": dt_orig,
                "datetime": dt_orig,
                "orientation": orientation,
            }
    except Exception:
        return {
            "metadata_status": "NO METADATA",
            "status": "NO METADATA",
            "has_exif": False,
            "dimensions": None,
            "format": None,
        }


def filter_social_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter candidate matches to prioritize recognized social media platforms."""
    social_platforms = {"Instagram", "LinkedIn", "Facebook", "X (Twitter)", "Reddit", "YouTube", "TikTok", "Pinterest"}
    return [m for m in matches if m.get("platform") in social_platforms]
