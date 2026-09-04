"""Face detection + embedding using InsightFace (buffalo_l / ArcFace, 512-d).

Features:
- Robust face detection via RetinaFace / SCRFD (buffalo_l on CPU)
- Largest face selection for multi-face / cluttered images
- Complete 512-dimensional ArcFace embedding extraction
- Consistent L2 vector normalization (float32)
- Deterministic SHA-256 hashing of complete normalized embedding bytes
- Rich metadata export (model, dimension, dtype, normalization, hash)
- Deterministic cosine similarity calculation
"""
import hashlib
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# Global cached FaceAnalysis instance (lazy initialized on first use)
_app = None


def get_app():
    """Lazily initialize and return the InsightFace FaceAnalysis instance."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize a 512-d embedding vector using float32 precision.

    Args:
        embedding: Raw 1D or 2D numpy array embedding.

    Returns:
        L2-normalized float32 numpy array with unit norm.
    """
    arr = np.asarray(embedding, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    return arr


def hash_embedding(embedding: np.ndarray) -> str:
    """Compute deterministic SHA-256 hex digest of the complete 512-d normalized embedding.

    Uses deterministic C-contiguous IEEE-754 float32 byte representation.

    Args:
        embedding: 512-d ArcFace embedding vector.

    Returns:
        64-character SHA-256 hexadecimal string.
    """
    normed = normalize_embedding(embedding)
    # tobytes() on C-contiguous float32 produces exact IEEE-754 32-bit little-endian bytes
    raw_bytes = normed.astype(np.float32, order="C").tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def get_embedding_metadata(embedding: np.ndarray) -> dict[str, Any]:
    """Return complete cryptographic and model metadata for a face embedding."""
    normed = normalize_embedding(embedding)
    return {
        "algorithm": "InsightFace buffalo_l",
        "model": "ArcFace",
        "dimension": int(normed.shape[0]),
        "dtype": "float32",
        "normalization_status": "L2_normalized",
        "embedding_hash": hash_embedding(normed),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute deterministic cosine similarity between two face embeddings.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1.0, 1.0].
    """
    norm_a = normalize_embedding(a)
    norm_b = normalize_embedding(b)
    dot_prod = float(np.dot(norm_a, norm_b))
    return max(-1.0, min(1.0, dot_prod))


def analyze_face(image_path: str) -> dict[str, Any]:
    """Perform face detection and return rich analysis including largest face embedding.

    Args:
        image_path: Path to the input image file.

    Returns:
        Dictionary containing:
        - face_detected: bool
        - face_count: int
        - det_score: float (confidence of detection)
        - bbox: list[float] [x1, y1, x2, y2]
        - embedding: np.ndarray (raw 512-d)
        - normalized_embedding: np.ndarray (L2-normalized 512-d float32)
        - embedding_hash: str (SHA-256 of complete normalized embedding)
        - metadata: dict

    Raises:
        FileNotFoundError: If image cannot be read from path.
        ValueError: If no face is detected in the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image (unsupported or corrupted format): {image_path}")

    app = get_app()
    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face detected in image: {image_path}")

    # Sort faces by bounding box area (largest first)
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    primary_face = faces[0]

    raw_emb = primary_face.embedding
    norm_emb = normalize_embedding(raw_emb)
    emb_hash = hash_embedding(norm_emb)

    return {
        "face_detected": True,
        "face_count": len(faces),
        "det_score": float(primary_face.det_score) if hasattr(primary_face, "det_score") else 1.0,
        "bbox": [float(x) for x in primary_face.bbox],
        "embedding": raw_emb,
        "normalized_embedding": norm_emb,
        "embedding_hash": emb_hash,
        "metadata": {
            "algorithm": "InsightFace buffalo_l",
            "model": "ArcFace",
            "dimension": int(norm_emb.shape[0]),
            "dtype": "float32",
            "normalization_status": "L2_normalized",
            "embedding_hash": emb_hash,
        },
    }


def get_embedding(image_path: str) -> np.ndarray:
    """Detect the largest face in image_path and return its normalized 512-d ArcFace embedding.

    Args:
        image_path: Path to the image file.

    Returns:
        L2-normalized 512-d float32 numpy array.
    """
    analysis = analyze_face(image_path)
    return analysis["normalized_embedding"]
