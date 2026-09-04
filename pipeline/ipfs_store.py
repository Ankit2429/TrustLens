"""Pin JSON evidence records and query images to IPFS via Pinata.

Supports resilient multi-gateway fallback retrieval (Pinata, Cloudflare, IPFS.io, dweb.link).
"""
import os
from typing import Any, Optional

import requests

PIN_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
PIN_FILE_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
DEFAULT_GATEWAY = os.environ.get(
    "PINATA_GATEWAY", "https://amaranth-improved-bedbug-30.mypinata.cloud/ipfs/"
)

FALLBACK_GATEWAYS = [
    DEFAULT_GATEWAY,
    "https://ipfs.io/ipfs/",
    "https://dweb.link/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
]


def _headers() -> dict[str, str]:
    jwt = os.environ.get("PINATA_JWT")
    if not jwt or jwt == "your_pinata_jwt":
        raise EnvironmentError(
            "Missing or unconfigured PINATA_JWT. "
            "Please set your Pinata JWT token in the .env file."
        )
    return {"Authorization": f"Bearer {jwt}"}


def pin_json(record: dict[str, Any], name: Optional[str] = None) -> str:
    """Pin a canonical JSON evidence dictionary to IPFS via Pinata.

    Args:
        record: Dictionary to pin.
        name: Optional custom pin name.

    Returns:
        IPFS CID (Content Identifier) string.
    """
    body: dict[str, Any] = {"pinataContent": record}
    if name:
        body["pinataMetadata"] = {"name": name}

    try:
        resp = requests.post(PIN_JSON_URL, json=body, headers=_headers(), timeout=35)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error connecting to Pinata API: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Pinata JSON pinning failed (HTTP {resp.status_code}): {resp.text}")

    data = resp.json()
    return data["IpfsHash"]


def pin_file(file_path: str, name: Optional[str] = None) -> str:
    """Pin a local image file to IPFS via Pinata.

    Args:
        file_path: Local path to the image file.
        name: Optional custom pin name.

    Returns:
        IPFS CID (Content Identifier) string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File to pin does not exist: {file_path}")

    filename = name or os.path.basename(file_path)
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f)}
            resp = requests.post(PIN_FILE_URL, files=files, headers=_headers(), timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error uploading file to Pinata: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Pinata file pinning failed (HTTP {resp.status_code}): {resp.text}")

    data = resp.json()
    return data["IpfsHash"]


def gateway_url(cid: str, gateway: Optional[str] = None) -> str:
    """Generate a public IPFS gateway URL for a CID."""
    base = (gateway or DEFAULT_GATEWAY).rstrip("/") + "/"
    return f"{base}{cid}"


def fetch_json(cid: str) -> dict[str, Any]:
    """Fetch and parse JSON record from IPFS using resilient multi-gateway fallback.

    Args:
        cid: IPFS Content Identifier.

    Returns:
        Parsed JSON dictionary.

    Raises:
        RuntimeError: If all gateways fail to return the content.
    """
    errors: list[str] = []

    for gw in FALLBACK_GATEWAYS:
        url = f"{gw.rstrip('/')}/{cid}"
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            errors.append(f"{gw} -> HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{gw} -> {e}")

    raise RuntimeError(
        f"Failed to fetch IPFS CID '{cid}' across all fallback gateways:\n  "
        + "\n  ".join(errors)
    )

