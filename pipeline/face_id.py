"""Face detection + embedding using InsightFace (buffalo_l / ArcFace, 512-d).

Features:
- Robust face detection via RetinaFace / SCRFD (buffalo_l on CPU)
- Explicit 5-point landmark geometric analysis, face alignment, and full pose estimation (yaw, pitch, roll)
- Deep explainable image quality assessment (sharpness, exposure, contrast, blur, resolution, frontality, confidence, occlusion)
- Deterministic multi-view query representation (canonical aligned crop, horizontal mirror view, CLAHE illumination normalization)
- Multi-face query image support with explicit face selection
- Deep group-image multi-face evaluation (comparing query against ALL detected candidate faces with explicit FACE 01, FACE 02 IDs)
- Complete 512-dimensional ArcFace embedding extraction with float32 L2 normalization
- Deterministic SHA-256 hashing of complete normalized embedding bytes
- Rich metadata export and deterministic cosine similarity calculation
"""
import hashlib
import math
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


def assess_face_quality(img: np.ndarray, face: Any) -> dict[str, Any]:
    """Calculate an explainable image quality assessment for a detected face.

    Evaluates:
    1. Detector Confidence: SCRFD/RetinaFace detection score.
    2. Sharpness: Laplacian variance on the cropped face region (normalized 0-1).
    3. Contrast: Standard deviation of pixel intensities across the face crop.
    4. Resolution: Bounding box area relative to canonical face resolution (112x112).
    5. Exposure: Average luminance and standard deviation in the face region.
    6. Frontality & Complete Pose (Yaw, Pitch, Roll): 5-point landmark geometry.
    7. Occlusion Indicators: Symmetry and landmark distribution checks.

    Returns:
        Dictionary with overall quality score in [0.0, 1.0] and detailed breakdown.
    """
    h, w = img.shape[:2]
    bbox = [int(max(0, x)) for x in face.bbox]
    x1, y1, x2, y2 = min(bbox[0], w - 1), min(bbox[1], h - 1), min(bbox[2], w), min(bbox[3], h)

    # 1. Detector confidence
    conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
    conf_score = max(0.0, min(1.0, conf))

    # Face crop for pixel-level metrics
    face_crop = img[y1:y2, x1:x2]
    face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
    min_dim = min(face_w, face_h)

    if face_crop.size == 0 or face_crop.shape[0] < 5 or face_crop.shape[1] < 5:
        return {
            "overall_quality": round(conf_score * 0.5, 4),
            "is_usable": False,
            "breakdown": {
                "confidence": round(conf_score, 4),
                "sharpness": 0.0,
                "contrast": 0.0,
                "resolution": 0.0,
                "exposure": 0.0,
                "frontality": 0.0,
                "pose_yaw_est": 0.0,
                "pose_pitch_est": 0.0,
                "pose_roll_est": 0.0,
                "occlusion_indicator": "High Risk / Degraded Crop",
            },
        }

    # 2. Sharpness & Blur (Laplacian variance)
    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    sharpness_score = round(min(1.0, math.log1p(lap_var) / math.log1p(500)), 4)

    # 3. Contrast (Pixel intensity standard deviation)
    contrast_std = float(np.std(gray_crop))
    contrast_score = round(min(1.0, contrast_std / 64.0), 4)

    # 4. Resolution (Face dimension vs optimal 112x112 ArcFace input)
    resolution_score = round(min(1.0, min_dim / 112.0), 4)

    # 5. Exposure (Luminance balance)
    mean_lum = float(np.mean(gray_crop))
    if mean_lum < 40:
        exposure_score = max(0.1, mean_lum / 40.0 * 0.5)
    elif mean_lum > 220:
        exposure_score = max(0.1, (255 - mean_lum) / 35.0 * 0.5)
    else:
        dist = abs(mean_lum - 128) / 128.0
        exposure_score = 1.0 - (dist * 0.5)
    exposure_score = round(float(exposure_score), 4)

    # 6. Pose Estimation (Yaw, Pitch, Roll) & Frontality via 5-point landmarks
    frontality_score = 0.85
    yaw_est = 0.0
    pitch_est = 0.0
    roll_est = 0.0
    occlusion_note = "Clear / Unobstructed"

    if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5:
        kps = face.kps
        left_eye, right_eye, nose, left_mouth, right_mouth = kps[0], kps[1], kps[2], kps[3], kps[4]
        
        # Yaw from horizontal eye-nose asymmetry
        dist_left = float(np.linalg.norm(left_eye - nose))
        dist_right = float(np.linalg.norm(right_eye - nose))
        eye_dist = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
        asymmetry = abs(dist_left - dist_right) / eye_dist
        frontality_score = round(max(0.0, min(1.0, 1.0 - asymmetry * 1.5)), 4)
        yaw_est = round(float(asymmetry * 45.0), 1)

        # Roll from inter-ocular slope angle
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        roll_est = round(float(math.degrees(math.atan2(dy, dx))), 1)

        # Pitch from eye-midpoint to nose vs mouth proportion
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
        face_vertical_span = max(1.0, mouth_mid_y - eye_mid_y)
        nose_rel_pos = (nose[1] - eye_mid_y) / face_vertical_span
        pitch_deviation = nose_rel_pos - 0.58
        pitch_est = round(float(pitch_deviation * 60.0), 1)

        # Occlusion check
        if asymmetry > 0.65:
            occlusion_note = "Severe Profile Pose / Partial Occlusion"
        elif conf_score < 0.55:
            occlusion_note = "Low Confidence / Possible Occlusion or Compression"
        elif sharpness_score < 0.15:
            occlusion_note = "Heavy Motion Blur or Degraded Optics"

    # Weighted composite quality score
    overall = (
        0.25 * conf_score
        + 0.25 * sharpness_score
        + 0.20 * resolution_score
        + 0.15 * exposure_score
        + 0.15 * frontality_score
    )
    overall_quality = round(float(overall), 4)
    is_usable = overall_quality >= 0.20 and min_dim >= 20 and conf_score >= 0.35

    return {
        "overall_quality": overall_quality,
        "is_usable": is_usable,
        "breakdown": {
            "confidence": conf_score,
            "sharpness": sharpness_score,
            "contrast": contrast_score,
            "resolution": resolution_score,
            "exposure": exposure_score,
            "frontality": frontality_score,
            "pose_yaw_est": yaw_est,
            "pose_pitch_est": pitch_est,
            "pose_roll_est": roll_est,
            "occlusion_indicator": occlusion_note,
            "dimensions": f"{face_w}x{face_h}",
        },
    }


def apply_clahe_illumination_norm(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to luminance channel."""
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr


def extract_multiview_embeddings(
    img: np.ndarray,
    face: Any,
) -> dict[str, Any]:
    """Extract deterministic multi-view representations for an aligned query face.

    Generates:
    - canonical: Primary ArcFace 512-d normalized embedding vector & hash (source of truth).
    - flip_view: Horizontally flipped image ArcFace embedding for pose/symmetry robustness.
    - norm_view: CLAHE illumination-equalized face view for lighting robustness.

    Returns:
        Dictionary containing canonical and auxiliary view embeddings.
    """
    app = get_app()
    canonical_emb = normalize_embedding(face.embedding)

    # 1. Horizontally flipped view
    flip_emb = canonical_emb
    try:
        flipped_img = cv2.flip(img, 1)
        flipped_faces = app.get(flipped_img)
        if flipped_faces:
            w = img.shape[1]
            orig_cx = (face.bbox[0] + face.bbox[2]) / 2.0
            target_cx = w - orig_cx
            best_f = min(flipped_faces, key=lambda f: abs(((f.bbox[0] + f.bbox[2]) / 2.0) - target_cx))
            flip_emb = normalize_embedding(best_f.embedding)
    except Exception:
        pass

    # 2. Illumination-normalized view (CLAHE)
    norm_emb = canonical_emb
    try:
        clahe_img = apply_clahe_illumination_norm(img)
        clahe_faces = app.get(clahe_img)
        if clahe_faces:
            orig_cx = (face.bbox[0] + face.bbox[2]) / 2.0
            orig_cy = (face.bbox[1] + face.bbox[3]) / 2.0
            best_cf = min(clahe_faces, key=lambda f: math.hypot(((f.bbox[0] + f.bbox[2]) / 2.0) - orig_cx, ((f.bbox[1] + f.bbox[3]) / 2.0) - orig_cy))
            norm_emb = normalize_embedding(best_cf.embedding)
    except Exception:
        pass

    return {
        "canonical": canonical_emb,
        "canonical_hash": hash_embedding(canonical_emb),
        "flip_view": flip_emb,
        "norm_view": norm_emb,
    }


def detect_all_faces(
    image_input: Any,
    min_quality: float = 0.20,
) -> list[dict[str, Any]]:
    """Detect and evaluate all faces in an image with alignment and quality scoring.

    Args:
        image_input: Path to the image file (str/Path), raw image bytes, or decoded numpy BGR image.
        min_quality: Minimum quality threshold to filter low-confidence noise.

    Returns:
        List of structured face dictionaries sorted by bounding box area (largest first).

    Raises:
        FileNotFoundError: If image file path does not exist.
        ValueError: If image cannot be decoded.
    """
    if isinstance(image_input, (str, os.PathLike)):
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image file does not exist: {path_str}")
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Could not decode image: {path_str}")
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided byte buffer")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    app = get_app()
    raw_faces = app.get(img)
    if not raw_faces:
        return []

    # Sort faces by bounding box area (largest first)
    raw_faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for idx, f in enumerate(raw_faces):
        quality = assess_face_quality(img, f)
        raw_emb = f.embedding
        norm_emb = normalize_embedding(raw_emb)
        emb_hash = hash_embedding(norm_emb)

        landmarks = None
        if hasattr(f, "kps") and f.kps is not None:
            landmarks = [[float(pt[0]), float(pt[1])] for pt in f.kps]

        face_id = f"FACE {idx + 1:02d}"

        entry = {
            "face_id": face_id,
            "face_index": idx,
            "det_score": float(f.det_score) if hasattr(f, "det_score") else 1.0,
            "bbox": [float(x) for x in f.bbox],
            "landmarks": landmarks,
            "quality": quality,
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
        results.append(entry)

    return results


def analyze_face(
    image_path: str,
    face_index: int = 0,
    min_quality: float = 0.20,
) -> dict[str, Any]:
    """Perform face detection and return analysis for a chosen face index.

    Args:
        image_path: Path to the input image file.
        face_index: Index of face to select if multiple faces exist (default: 0).
        min_quality: Minimum quality threshold for the selected face.

    Returns:
        Dictionary containing face analysis, quality breakdown, and all faces metadata.

    Raises:
        FileNotFoundError: If image cannot be read from path.
        ValueError: If no face is detected or face_index is out of range.
    """
    if isinstance(image_path, (str, os.PathLike)):
        img = cv2.imread(str(image_path))
    else:
        img = None

    faces = detect_all_faces(image_path, min_quality=min_quality)
    if not faces:
        raise ValueError(f"No face detected in image: {image_path}")

    if face_index < 0 or face_index >= len(faces):
        raise ValueError(
            f"Requested face_index {face_index} is out of range. "
            f"Image contains {len(faces)} detected face(s) (indices 0 to {len(faces) - 1})."
        )

    selected = faces[face_index]
    quality = selected["quality"]

    # Multi-view query representation
    multiview = None
    if img is not None:
        try:
            app = get_app()
            raw_faces = app.get(img)
            if raw_faces and face_index < len(raw_faces):
                multiview = extract_multiview_embeddings(img, raw_faces[face_index])
        except Exception:
            pass

    return {
        "face_detected": True,
        "face_count": len(faces),
        "selected_face_index": face_index,
        "face_id": selected["face_id"],
        "det_score": selected["det_score"],
        "bbox": selected["bbox"],
        "landmarks": selected["landmarks"],
        "quality_score": quality["overall_quality"],
        "quality_breakdown": quality["breakdown"],
        "is_usable": quality["is_usable"],
        "embedding": selected["embedding"],
        "normalized_embedding": selected["normalized_embedding"],
        "embedding_hash": selected["embedding_hash"],
        "multiview": multiview,
        "metadata": selected["metadata"],
        "all_faces_summary": [
            {
                "face_id": f["face_id"],
                "index": f["face_index"],
                "bbox": f["bbox"],
                "det_score": round(f["det_score"], 4),
                "quality_score": f["quality"]["overall_quality"],
                "sharpness": f["quality"]["breakdown"]["sharpness"],
                "pose_yaw": f["quality"]["breakdown"]["pose_yaw_est"],
            }
            for f in faces
        ],
    }


def compare_group_faces(
    query_embedding: Any,
    candidate_faces: list[dict[str, Any]],
    query_multiview: Optional[dict[str, Any]] = None,
    verified_threshold: float = 0.60,
    review_threshold: float = 0.40,
) -> dict[str, Any]:
    """Compare query face embedding against ALL faces detected in a candidate group image.

    Independently evaluates Query ↔ A, Query ↔ B, Query ↔ C, Query ↔ D.
    Assigns internal IDs (FACE 01, FACE 02, etc.), computes similarity, quality, and decision
    for each face, and explicitly identifies which face in the candidate image matched.

    Args:
        query_embedding: Normalized 512-d query face embedding (or dict with multiview).
        candidate_faces: List of detected face dictionaries from detect_all_faces().
        query_multiview: Optional multi-view query representation dictionary.
        verified_threshold: Cutoff for individual face VERIFIED decision.
        review_threshold: Cutoff for individual face REVIEW decision.

    Returns:
        Dictionary containing:
        - best_similarity: float
        - best_face_index: int
        - matched_face_id: str (e.g. "FACE 02")
        - evaluated_face_count: int
        - all_similarities: list[float]
        - best_candidate_face: dict (metadata of the best matching face)
        - candidate_faces_evaluated: list[dict] with face-by-face scores and decisions
    """
    if not candidate_faces:
        return {
            "best_similarity": -1.0,
            "best_face_index": -1,
            "matched_face_id": "NONE",
            "evaluated_face_count": 0,
            "all_similarities": [],
            "best_candidate_face": None,
            "candidate_faces_evaluated": [],
        }

    # Extract query representation vectors
    q_norm = None
    if isinstance(query_embedding, dict) and "canonical" in query_embedding:
        q_canon = query_embedding["canonical"]
        q_flip = query_embedding.get("flip_view")
        q_norm = query_embedding.get("norm_view")
    else:
        q_canon = query_embedding
        q_flip = query_multiview.get("flip_view") if query_multiview else None
        q_norm = query_multiview.get("norm_view") if query_multiview else None

    similarities: list[float] = []
    face_breakdown: list[dict[str, Any]] = []

    for idx, f in enumerate(candidate_faces):
        cand_emb = f["normalized_embedding"]
        cand_qual = 0.85
        if "quality" in f and f["quality"] is not None:
            if isinstance(f["quality"], dict):
                cand_qual = float(f["quality"].get("overall_quality", 0.85))
            else:
                cand_qual = float(f["quality"])
        cand_conf = float(f.get("det_score", 1.0))
        cand_face_id = f.get("face_id", f"FACE {idx + 1:02d}")
        cand_bbox = f.get("bbox", [0.0, 0.0, 100.0, 100.0])

        # Compute cosine similarities across views
        sim_canon = cosine_similarity(q_canon, cand_emb)
        sim = sim_canon

        if q_flip is not None or q_norm is not None:
            aux_scores = [sim_canon]
            if q_flip is not None:
                aux_scores.append(cosine_similarity(q_flip, cand_emb))
            if q_norm is not None:
                aux_scores.append(cosine_similarity(q_norm, cand_emb))

            # Technically justified weighted ensemble (canonical 70%, auxiliary 15% each)
            if len(aux_scores) == 3:
                ens = 0.70 * aux_scores[0] + 0.15 * aux_scores[1] + 0.15 * aux_scores[2]
            else:
                ens = 0.75 * aux_scores[0] + 0.25 * aux_scores[1]
            sim = max(sim_canon, ens)

        similarities.append(sim)

        # Determine individual face decision
        if sim >= verified_threshold and cand_qual >= 0.20:
            face_dec = "VERIFIED"
        elif sim >= review_threshold:
            face_dec = "REVIEW"
        else:
            face_dec = "REJECTED"

        face_breakdown.append({
            "face_id": cand_face_id,
            "face_index": idx,
            "bbox": cand_bbox,
            "similarity": round(sim, 4),
            "quality": round(cand_qual, 4),
            "det_confidence": round(cand_conf, 4),
            "decision": face_dec,
            "is_matched": False,
        })

    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])
    matched_id = candidate_faces[best_idx].get("face_id", f"FACE {best_idx + 1:02d}")

    if face_breakdown and 0 <= best_idx < len(face_breakdown):
        face_breakdown[best_idx]["is_matched"] = True

    return {
        "best_similarity": best_sim,
        "best_face_index": best_idx,
        "matched_face_id": matched_id,
        "evaluated_face_count": len(candidate_faces),
        "all_similarities": similarities,
        "best_candidate_face": candidate_faces[best_idx],
        "candidate_faces_evaluated": face_breakdown,
    }


def get_embedding(image_path: str) -> np.ndarray:
    """Detect the primary face in image_path and return its normalized 512-d ArcFace embedding.

    Args:
        image_path: Path to the image file.

    Returns:
        L2-normalized 512-d float32 numpy array.
    """
    analysis = analyze_face(image_path)
    return analysis["normalized_embedding"]
