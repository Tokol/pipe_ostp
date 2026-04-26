from __future__ import annotations

import base64
import html
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - depends on deployment environment.
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from scipy.optimize import least_squares, minimize
except Exception:  # pragma: no cover - app still works with algebraic fit fallback.
    least_squares = None
    minimize = None


if CV2_IMPORT_ERROR is not None:  # pragma: no cover - UI guard for missing host dependency.
    st.error(
        "OpenCV is not installed in this environment. "
        "Add `opencv-python-headless==4.10.0.84` to `requirements.txt` "
        "and redeploy the Streamlit app."
    )
    st.caption(f"Import error: {CV2_IMPORT_ERROR}")
    st.stop()


APP_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = APP_DIR / "camera_calibration.json"
SAVED_SCALE_PATH = APP_DIR / "saved_scale_config.json"
TOLERANCE_PATH = APP_DIR / "TOLERANSTABELL.xlsx"
CONTRACT_DIR = APP_DIR / "ostb_data_map_contract"
STANDARDS_CONTRACT_PATH = CONTRACT_DIR / "standards.json"
EXECUTION_CONTRACT_PATH = CONTRACT_DIR / "standards.execution.json"
OVALITY_LOOKUP_PATH = CONTRACT_DIR / "ovality_lookup.json"
DIAMETER_LOOKUP_PATH = CONTRACT_DIR / "diameter_lookup.json"
NOVIA_LOGO_PATH = APP_DIR / "images" / "novia_uas.png"
OSTP_LOGO_PATH = APP_DIR / "images" / "ostp_logo.png"
SUCCESS_GIF_PATH = APP_DIR / "images" / "sucess_mario.gif"
FAIL_GIF_PATH = APP_DIR / "images" / "pie_fail_match_success.gif"
STANDARD_IMAGE_BY_ID = {
    "1": "1t3.jpg",
    "2": "1t3.jpg",
    "3": "1t3.jpg",
    "3.1": "1t3.jpg",
    "4": "4n5.jpg",
    "5": "4n5.jpg",
    "6": "6n7.jpg",
    "7": "6n7.jpg",
    "8": "8.jpg",
    "9": "9.jpg",
    "10": "10.jpg",
}
STANDARD_UI_TITLE_BY_ID = {
    "1": "EN 10217-7:2005 pipes with dimensions and tolerances defined by EN ISO 1127",
    "2": "EN 10217-7:2005 double dimension pipes with dimensions and tolerances defined by EN ISO 1127",
    "3": "ASTM A312 / A790 / A928 pipes with dimensions defined by ASTM A999 and ASTM A358",
    "3.1": "ASTM A249 / A1016 pipes with dimensions defined by ASTM A358",
    "4": "EN 10253-4:2008 fittings with defined geometry and dimensional specifications",
    "5": "ASTM A403-07 fittings with dimensions defined by ASME B16.9-2007",
    "6": "EN 10253-4:2008 and ASTM A403-07 equal tee fittings with dimensions defined by ASME B16.9-2007",
    "7": "EN 10253-4:2008 and ASTM A403-07 reduced tee fittings with dimensions defined by ASME B16.9-2007",
    "8": "EN 10253-4:2008 formed tee fittings (Vedetty type) with defined dimensions",
    "9": "EN 10253-4:2008 type 3 tee fittings with defined dimensions",
    "10": "Collar components with defined flatness and dimensional tolerances",
}


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    reprojection_error: Optional[float] = None


@dataclass
class PreprocessConfig:
    blur_kernel: int = 5
    canny_sigma: float = 0.33
    morphology_kernel: int = 3
    morphology_close_iterations: int = 1


@dataclass
class ContourConfig:
    min_area: float = 30.0
    min_points: int = 30
    min_radius_fraction: float = 0.005
    max_radius_fraction: float = 0.30
    circular_smoothing_window: int = 5


@dataclass
class RansacConfig:
    max_iterations: int = 1000
    inlier_threshold_px: Optional[float] = None
    threshold_radius_fraction: float = 0.006
    min_inlier_ratio: float = 0.35
    random_seed: int = 42


@dataclass
class PipeMeasurement:
    filename: str
    center_x_px: float
    center_y_px: float
    radius_px: float
    diameter_px: float
    diameter_mm: float
    roundness_px: float
    roundness_mm: float
    max_abs_deviation_px: float
    max_abs_deviation_mm: float
    contour_points: np.ndarray
    fitting_inliers: np.ndarray
    signed_errors_px: np.ndarray
    mm_per_pixel: float


@dataclass
class PipeWallDetection:
    inner_center_x_px: float
    inner_center_y_px: float
    inner_radius_px: float
    inner_points: np.ndarray
    outer_center_x_px: float
    outer_center_y_px: float
    outer_radius_px: float
    outer_points: np.ndarray
    wall_thickness_px: float
    wall_thickness_min_px: float
    wall_thickness_max_px: float
    center_offset_px: float
    method: str


STANDARD_LIBRARY = {
    "EN 10217-7 / EN ISO 1127 (sheet 1)": {"sheet": "1", "mode": "EN_DCLASS"},
    "EN 10217-7 double dimensions (sheet 2)": {"sheet": "2", "mode": "DIRECT_TABLE"},
    "ASTM A312 / A790 / A928 / A358 (sheet 3)": {"sheet": "3", "mode": "DIRECT_TABLE"},
    "ASTM A249 / A1016 (sheet 3.1)": {"sheet": "3.1", "mode": "DIRECT_TABLE"},
    "EN 10253-4 (sheet 4)": {"sheet": "4", "mode": "EN_DCLASS"},
    "ASTM A403 / ASME B16.9 (sheet 5)": {"sheet": "5", "mode": "DIRECT_TABLE"},
    "EN 10253-4 alternate table (sheet 6)": {"sheet": "6", "mode": "EN_DCLASS"},
    "Reducers / special dimensions (sheet 7)": {"sheet": "7", "mode": "DIRECT_TABLE"},
    "Hydro equal table (sheet 8)": {"sheet": "8", "mode": "DIRECT_TABLE"},
    "Hydro reduced table (sheet 9)": {"sheet": "9", "mode": "DIRECT_TABLE"},
    "ASTM A358 style table (sheet 10)": {"sheet": "10", "mode": "EN_DCLASS"},
}


@st.cache_resource
def load_static_calibration() -> Optional[CameraCalibration]:
    if not CALIBRATION_PATH.exists():
        return None
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return CameraCalibration(
        camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64),
        image_size=tuple(payload["image_size"]),
        reprojection_error=payload.get("reprojection_error"),
    )


@st.cache_data
def load_static_excel_bytes() -> Optional[bytes]:
    if not TOLERANCE_PATH.exists():
        return None
    return TOLERANCE_PATH.read_bytes()


@st.cache_data
def load_saved_scale_config() -> Optional[Dict[str, object]]:
    if not SAVED_SCALE_PATH.exists():
        return None
    try:
        payload = json.loads(SAVED_SCALE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        mm_per_pixel = float(payload.get("mm_per_pixel", 0.0))
    except (TypeError, ValueError):
        return None
    if mm_per_pixel <= 0:
        return None
    payload["mm_per_pixel"] = mm_per_pixel
    return payload


def save_scale_config(
    mm_per_pixel: float,
    source: str,
    reference_diameter_mm: float,
    reference_diameter_px: float,
    reference_filename: str,
    measurement_target: str,
) -> None:
    payload = {
        "mm_per_pixel": float(mm_per_pixel),
        "source": source,
        "reference_diameter_mm": float(reference_diameter_mm),
        "reference_diameter_px": float(reference_diameter_px),
        "reference_filename": reference_filename,
        "measurement_target": measurement_target,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "valid_for": "same camera, same distance, same zoom, same resolution, same pipe plane",
    }
    SAVED_SCALE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_saved_scale_config.clear()


def load_standards_contract() -> Tuple[Dict[str, object], Dict[str, object]]:
    if not STANDARDS_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Standards contract not found: {STANDARDS_CONTRACT_PATH}")
    if not EXECUTION_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Execution contract not found: {EXECUTION_CONTRACT_PATH}")
    standards = json.loads(STANDARDS_CONTRACT_PATH.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract = standards.get("metadata", {}).get("contract", {})
    execution_ref = contract.get("execution_spec", {})
    if execution_ref.get("$ref") != "standards.execution.json" or execution_ref.get("version") != execution.get("version"):
        raise ValueError("Standards contract and execution contract versions do not match.")
    return standards, execution


def load_ovality_lookup() -> Dict[str, object]:
    if not OVALITY_LOOKUP_PATH.exists():
        return {}
    return json.loads(OVALITY_LOOKUP_PATH.read_text(encoding="utf-8"))


def load_diameter_lookup() -> Dict[str, object]:
    if not DIAMETER_LOOKUP_PATH.exists():
        return {}
    return json.loads(DIAMETER_LOOKUP_PATH.read_text(encoding="utf-8"))


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def undistort_image(image_bgr: np.ndarray, calibration: Optional[CameraCalibration]) -> np.ndarray:
    if calibration is None:
        return image_bgr
    return cv2.undistort(image_bgr, calibration.camera_matrix, calibration.dist_coeffs)


def adaptive_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if lower >= upper:
        lower, upper = 30, 100
    return cv2.Canny(gray, lower, upper)


def preprocess_image(
    image_bgr: np.ndarray,
    calibration: Optional[CameraCalibration],
    config: PreprocessConfig,
) -> Dict[str, np.ndarray]:
    corrected = undistort_image(image_bgr, calibration)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    kernel_size = int(config.blur_kernel)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)

    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = adaptive_canny(blurred, sigma=config.canny_sigma)
    morph_size = max(1, int(config.morphology_kernel))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    clean_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        morph_kernel,
        iterations=max(0, int(config.morphology_close_iterations)),
    )

    return {
        "image_bgr": corrected,
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "clean_edges": clean_edges,
    }


def detect_pipe_roi(
    image_bgr: np.ndarray,
    min_radius_fraction: float = 0.008,
    max_radius_fraction: float = 0.25,
    crop_radius_multiplier: float = 2.0,
    min_crop_half_size: int = 90,
) -> Tuple[Tuple[int, int, int, int], str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    height, width = gray.shape[:2]
    min_dim = min(width, height)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    min_radius = max(5, int(min_dim * min_radius_fraction))
    max_radius = max(min_radius + 2, int(min_dim * max_radius_fraction))
    hough_max_radius = max(min_radius + 2, min(max_radius, int(min_dim * 0.18)))
    center_limit = min_dim * 0.38

    def add_ellipse_candidate(contour: np.ndarray, method: str, score_bonus: float = 0.0) -> None:
        if len(contour) < 5:
            return
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            return
        if max(w, h) < min_radius * 1.5 or max(w, h) > max_radius * 2.4:
            return
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if area < 20 or perimeter <= 0:
            return
        try:
            (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        except Exception:
            return
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if major <= 0 or minor <= 0:
            return
        radius = major / 2.0
        axis_ratio = minor / major
        if radius < min_radius or radius > max_radius:
            return
        if axis_ratio < 0.42:
            return
        center_distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - image_center))
        if center_distance > center_limit:
            return
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.18 and axis_ratio < 0.55:
            return
        score = score_bonus + radius - 0.07 * center_distance + 28.0 * axis_ratio + 12.0 * circularity
        candidates.append((score, float(cx), float(cy), float(radius), method))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim * 0.08),
        param1=120,
        param2=18,
        minRadius=min_radius,
        maxRadius=hough_max_radius,
    )

    candidates = []
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(np.float32):
            center_distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - image_center))
            if center_distance > center_limit:
                continue
            score = float(radius) - 0.08 * center_distance
            candidates.append((score, float(cx), float(cy), float(radius), "Hough compact ring"))

    if not candidates:
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda item: abs(float(cv2.contourArea(item))), reverse=True)[:60]
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            perimeter = float(cv2.arcLength(contour, closed=True))
            if area < 20 or perimeter <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if min_radius <= radius <= max_radius:
                circularity = 4.0 * math.pi * area / (perimeter * perimeter)
                center_distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - image_center))
                if center_distance <= center_limit and circularity >= 0.30:
                    score = 20.0 + 30.0 * circularity + radius - 0.08 * center_distance
                    candidates.append((score, float(cx), float(cy), radius, "dark circular opening"))
            add_ellipse_candidate(contour, "dark elliptical opening", score_bonus=18.0)

    if not candidates:
        edge_map = adaptive_canny(blur, sigma=0.40)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
        edge_contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        edge_contours = sorted(edge_contours, key=lambda item: abs(float(cv2.contourArea(item))), reverse=True)[:60]
        for contour in edge_contours:
            if len(contour) < 35:
                continue
            add_ellipse_candidate(contour, "edge elliptical rim", score_bonus=8.0)

    if not candidates:
        raise ValueError("Automatic pipe ROI detection failed. The pipe opening/rim was not found clearly.")

    _, cx, cy, radius, method = max(candidates, key=lambda item: item[0])
    effective_crop_multiplier = crop_radius_multiplier
    if "opening" in method:
        effective_crop_multiplier = max(effective_crop_multiplier, 3.2)
    half_size = int(max(min_crop_half_size, radius * effective_crop_multiplier))
    x1 = max(0, int(round(cx - half_size)))
    y1 = max(0, int(round(cy - half_size)))
    x2 = min(width, int(round(cx + half_size)))
    y2 = min(height, int(round(cy + half_size)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Automatic pipe ROI detection produced an empty crop.")

    return (
        x1,
        y1,
        x2,
        y2,
    ), f"auto ROI via {method}: center=({cx:.1f}, {cy:.1f}), seed_radius={radius:.1f}px"


def circular_smooth_points(points: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(points) < window:
        return points.astype(np.float32)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.vstack([points[-half:], points, points[:half]]).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.zeros_like(points, dtype=np.float32)
    smoothed[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
    smoothed[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
    return smoothed


def select_pipe_contour(clean_edges: np.ndarray, image_shape: Tuple[int, int, int], config: ContourConfig) -> np.ndarray:
    contours, _ = cv2.findContours(clean_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found.")

    height, width = image_shape[:2]
    min_dim = min(width, height)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    min_radius = min_dim * config.min_radius_fraction
    max_radius = min_dim * config.max_radius_fraction
    candidates = []

    for contour in contours:
        if len(contour) < config.min_points:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < config.min_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = float(radius)
        if radius < min_radius or radius > max_radius:
            continue
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        center_distance = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - image_center))
        ellipse_axis_ratio = None
        if len(contour) >= 5:
            try:
                (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
                major = max(axis_a, axis_b)
                minor = min(axis_a, axis_b)
                ellipse_axis_ratio = float(minor / major) if major > 0 else None
            except Exception:
                ellipse_axis_ratio = None
        if circularity < 0.35 and (ellipse_axis_ratio is None or ellipse_axis_ratio < 0.55):
            continue
        candidates.append(
            {
                "contour": contour,
                "radius": radius,
                "circularity": circularity,
                "ellipse_axis_ratio": ellipse_axis_ratio,
                "center_distance": center_distance,
            }
        )

    if not candidates:
        raise ValueError("No contour passed the pipe filters. Try a clearer image or relaxed settings.")

    central_limit = min_dim * 0.22
    central_candidates = [
        item
        for item in candidates
        if item["center_distance"] <= central_limit
        and (item["circularity"] >= 0.45 or (item["ellipse_axis_ratio"] or 0.0) >= 0.65)
    ]
    pool = central_candidates if central_candidates else candidates
    best = max(
        pool,
        key=lambda item: (
            item["radius"]
            - 0.20 * item["center_distance"]
            + 20.0 * item["circularity"]
            + 10.0 * (item["ellipse_axis_ratio"] or 0.0)
        ),
    )
    points = best["contour"].reshape(-1, 2).astype(np.float32)
    return circular_smooth_points(points, config.circular_smoothing_window)


def circle_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    denominator = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(denominator) < 1e-9:
        raise ValueError("Sampled points are collinear.")
    x1_sq = x1 * x1 + y1 * y1
    x2_sq = x2 * x2 + y2 * y2
    x3_sq = x3 * x3 + y3 * y3
    xc = (x1_sq * (y2 - y3) + x2_sq * (y3 - y1) + x3_sq * (y1 - y2)) / denominator
    yc = (x1_sq * (x3 - x2) + x2_sq * (x1 - x3) + x3_sq * (x2 - x1)) / denominator
    radius = float(np.linalg.norm(np.array([xc, yc]) - p1))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid circle radius.")
    return float(xc), float(yc), radius


def algebraic_circle_fit(points: np.ndarray) -> Tuple[float, float, float]:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y
    xc, yc, c = np.linalg.lstsq(design, target, rcond=None)[0]
    radius = math.sqrt(max(float(c + xc * xc + yc * yc), 0.0))
    return float(xc), float(yc), float(radius)


def refine_circle_least_squares(points: np.ndarray, initial_circle: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if least_squares is None:
        return algebraic_circle_fit(points)

    def residual(params: np.ndarray) -> np.ndarray:
        xc, yc, radius = params
        return np.linalg.norm(points - np.array([xc, yc]), axis=1) - radius

    result = least_squares(residual, np.array(initial_circle, dtype=np.float64), max_nfev=300)
    xc, yc, radius = result.x
    return float(xc), float(yc), abs(float(radius))


def fit_circle_ransac(points: np.ndarray, config: RansacConfig) -> Tuple[Tuple[float, float, float], np.ndarray]:
    if len(points) < 3:
        raise ValueError("At least 3 points are required for circle fitting.")

    center_guess = np.mean(points, axis=0)
    radius_guess = float(np.median(np.linalg.norm(points - center_guess, axis=1)))
    threshold = config.inlier_threshold_px
    if threshold is None:
        threshold = max(1.0, radius_guess * config.threshold_radius_fraction)

    rng = np.random.default_rng(config.random_seed)
    best_inliers = np.array([], dtype=np.int64)
    point_count = len(points)
    for _ in range(config.max_iterations):
        sample_idx = rng.choice(point_count, 3, replace=False)
        try:
            xc, yc, radius = circle_from_three_points(points[sample_idx[0]], points[sample_idx[1]], points[sample_idx[2]])
        except ValueError:
            continue
        distances = np.linalg.norm(points - np.array([xc, yc]), axis=1)
        residuals = np.abs(distances - radius)
        inliers = np.where(residuals <= threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    min_required = max(20, int(point_count * config.min_inlier_ratio))
    if len(best_inliers) < min_required:
        raise ValueError(f"RANSAC failed: only {len(best_inliers)} inliers, required {min_required}.")

    initial = algebraic_circle_fit(points[best_inliers])
    refined = refine_circle_least_squares(points[best_inliers], initial)
    return refined, best_inliers


def compute_roundness(points: np.ndarray, circle: Tuple[float, float, float]) -> Dict[str, object]:
    xc, yc, radius = circle
    distances = np.linalg.norm(points - np.array([xc, yc]), axis=1)
    signed_errors = distances - radius
    return {
        "signed_errors_px": signed_errors,
        "roundness_px": float(np.max(signed_errors) - np.min(signed_errors)),
        "max_abs_deviation_px": float(np.max(np.abs(signed_errors))),
    }


def measurement_diameter_stats(measurement: PipeMeasurement) -> Dict[str, float]:
    if len(measurement.contour_points) == 0:
        return {
            "fitted_diameter_px": float(measurement.diameter_px),
            "real_edge_min_diameter_px": float("nan"),
            "real_edge_max_diameter_px": float("nan"),
            "real_edge_diameter_range_px": float("nan"),
        }
    center = np.array([measurement.center_x_px, measurement.center_y_px], dtype=np.float64)
    points = measurement.contour_points.astype(np.float64)
    diameters = 2.0 * np.linalg.norm(points - center, axis=1)
    return {
        "fitted_diameter_px": float(measurement.diameter_px),
        "real_edge_min_diameter_px": float(np.min(diameters)),
        "real_edge_max_diameter_px": float(np.max(diameters)),
        "real_edge_diameter_range_px": float(np.max(diameters) - np.min(diameters)),
    }


def compute_roundness_method_stats(measurement: PipeMeasurement) -> Dict[str, object]:
    points = measurement.contour_points.astype(np.float64)
    scale = float(measurement.mm_per_pixel)
    if len(points) == 0:
        return {
            "lsc_roundness_px": float("nan"),
            "lsc_roundness_mm": float("nan"),
            "mzc_roundness_px": float("nan"),
            "mzc_roundness_mm": float("nan"),
            "mzc_center_x_px": float("nan"),
            "mzc_center_y_px": float("nan"),
            "mzc_inner_radius_px": float("nan"),
            "mzc_outer_radius_px": float("nan"),
            "mzc_optimizer": "not available",
            "mcc_roundness_px": float("nan"),
            "mcc_roundness_mm": float("nan"),
            "mcc_radius_px": float("nan"),
            "mic_diameter_px": float("nan"),
            "mic_diameter_mm": float("nan"),
            "mic_center_x_px": float("nan"),
            "mic_center_y_px": float("nan"),
            "mic_radius_px": float("nan"),
        }

    lsc_center = np.array([measurement.center_x_px, measurement.center_y_px], dtype=np.float64)
    ordered_points = points[np.argsort(np.arctan2(points[:, 1] - lsc_center[1], points[:, 0] - lsc_center[0]))]

    def zone_for_center(center: np.ndarray) -> Tuple[float, float, float]:
        distances = np.linalg.norm(points - center, axis=1)
        inner_radius = float(np.min(distances))
        outer_radius = float(np.max(distances))
        return outer_radius - inner_radius, inner_radius, outer_radius

    lsc_roundness_px, lsc_inner_px, lsc_outer_px = zone_for_center(lsc_center)
    mzc_center = lsc_center.copy()
    mzc_roundness_px = lsc_roundness_px
    mzc_inner_px = lsc_inner_px
    mzc_outer_px = lsc_outer_px
    mzc_optimizer = "LSC fallback"

    if minimize is not None and len(points) >= 3:
        point_span = np.ptp(points, axis=0)
        tolerance = max(float(np.max(point_span)) * 1e-7, 1e-5)

        def objective(center_xy: np.ndarray) -> float:
            return zone_for_center(center_xy)[0]

        result = minimize(
            objective,
            lsc_center,
            method="Nelder-Mead",
            options={"maxiter": 700, "xatol": tolerance, "fatol": tolerance},
        )
        if result.success and np.isfinite(result.fun):
            candidate_roundness, candidate_inner, candidate_outer = zone_for_center(result.x)
            if candidate_roundness <= mzc_roundness_px:
                mzc_center = np.array(result.x, dtype=np.float64)
                mzc_roundness_px = candidate_roundness
                mzc_inner_px = candidate_inner
                mzc_outer_px = candidate_outer
                mzc_optimizer = "Nelder-Mead minimum zone"

    (mcc_center_x, mcc_center_y), mcc_radius_px = cv2.minEnclosingCircle(points.astype(np.float32))
    mcc_center = np.array([mcc_center_x, mcc_center_y], dtype=np.float64)
    mcc_distances = np.linalg.norm(points - mcc_center, axis=1)
    mcc_inner_radius_px = float(np.min(mcc_distances))
    mcc_roundness_px = float(mcc_radius_px - mcc_inner_radius_px)
    mic_center_x = float("nan")
    mic_center_y = float("nan")
    mic_radius_px = float("nan")
    mic_diameter_px = float("nan")
    if len(ordered_points) >= 3:
        min_xy = np.floor(np.min(ordered_points, axis=0)).astype(int)
        max_xy = np.ceil(np.max(ordered_points, axis=0)).astype(int)
        padding = max(8, int(round(measurement.radius_px * 0.08)))
        width = int(max(1, max_xy[0] - min_xy[0] + 1 + 2 * padding))
        height = int(max(1, max_xy[1] - min_xy[1] + 1 + 2 * padding))
        shifted = np.round(ordered_points - min_xy + padding).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 255)
        if np.count_nonzero(mask) > 0:
            distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, max_distance, _, max_location = cv2.minMaxLoc(distances)
            mic_radius_px = float(max_distance)
            mic_diameter_px = float(2.0 * mic_radius_px)
            mic_center_x = float(max_location[0] + min_xy[0] - padding)
            mic_center_y = float(max_location[1] + min_xy[1] - padding)

    return {
        "lsc_roundness_px": float(lsc_roundness_px),
        "lsc_roundness_mm": float(lsc_roundness_px * scale),
        "lsc_inner_radius_px": float(lsc_inner_px),
        "lsc_outer_radius_px": float(lsc_outer_px),
        "mzc_roundness_px": float(mzc_roundness_px),
        "mzc_roundness_mm": float(mzc_roundness_px * scale),
        "mzc_center_x_px": float(mzc_center[0]),
        "mzc_center_y_px": float(mzc_center[1]),
        "mzc_inner_radius_px": float(mzc_inner_px),
        "mzc_outer_radius_px": float(mzc_outer_px),
        "mzc_mean_diameter_px": float((mzc_inner_px + mzc_outer_px)),
        "mzc_optimizer": mzc_optimizer,
        "mcc_roundness_px": mcc_roundness_px,
        "mcc_roundness_mm": float(mcc_roundness_px * scale),
        "mcc_center_x_px": float(mcc_center_x),
        "mcc_center_y_px": float(mcc_center_y),
        "mcc_radius_px": float(mcc_radius_px),
        "mic_diameter_px": mic_diameter_px,
        "mic_diameter_mm": float(mic_diameter_px * scale),
        "mic_center_x_px": mic_center_x,
        "mic_center_y_px": mic_center_y,
        "mic_radius_px": mic_radius_px,
    }


def fit_circle_with_outlier_trim(points: np.ndarray) -> Tuple[float, float, float]:
    if len(points) < 5:
        return algebraic_circle_fit(points)
    initial = refine_circle_least_squares(points, algebraic_circle_fit(points))
    xc, yc, radius = initial
    distances = np.linalg.norm(points - np.array([xc, yc], dtype=np.float64), axis=1)
    residuals = np.abs(distances - radius)
    trim_limit = max(2.5, float(np.percentile(residuals, 85)))
    trimmed = points[residuals <= trim_limit]
    if len(trimmed) < max(5, int(len(points) * 0.45)):
        return initial
    return refine_circle_least_squares(trimmed, algebraic_circle_fit(trimmed))


def build_outer_wall_measurement(base_measurement: PipeMeasurement, wall_detection: PipeWallDetection) -> PipeMeasurement:
    outer_points = wall_detection.outer_points.astype(np.float32)
    outer_circle = (
        wall_detection.outer_center_x_px,
        wall_detection.outer_center_y_px,
        wall_detection.outer_radius_px,
    )
    roundness = compute_roundness(outer_points, outer_circle)
    mm_per_pixel = float(base_measurement.mm_per_pixel)
    return PipeMeasurement(
        filename=base_measurement.filename,
        center_x_px=wall_detection.outer_center_x_px,
        center_y_px=wall_detection.outer_center_y_px,
        radius_px=wall_detection.outer_radius_px,
        diameter_px=2.0 * wall_detection.outer_radius_px,
        diameter_mm=2.0 * wall_detection.outer_radius_px * mm_per_pixel,
        roundness_px=float(roundness["roundness_px"]),
        roundness_mm=float(roundness["roundness_px"]) * mm_per_pixel,
        max_abs_deviation_px=float(roundness["max_abs_deviation_px"]),
        max_abs_deviation_mm=float(roundness["max_abs_deviation_px"]) * mm_per_pixel,
        contour_points=outer_points,
        fitting_inliers=outer_points,
        signed_errors_px=roundness["signed_errors_px"],
        mm_per_pixel=mm_per_pixel,
    )


def build_inner_wall_measurement(base_measurement: PipeMeasurement, wall_detection: PipeWallDetection) -> PipeMeasurement:
    inner_points = wall_detection.inner_points.astype(np.float32)
    inner_circle = (
        wall_detection.inner_center_x_px,
        wall_detection.inner_center_y_px,
        wall_detection.inner_radius_px,
    )
    roundness = compute_roundness(inner_points, inner_circle)
    mm_per_pixel = float(base_measurement.mm_per_pixel)
    return PipeMeasurement(
        filename=base_measurement.filename,
        center_x_px=wall_detection.inner_center_x_px,
        center_y_px=wall_detection.inner_center_y_px,
        radius_px=wall_detection.inner_radius_px,
        diameter_px=2.0 * wall_detection.inner_radius_px,
        diameter_mm=2.0 * wall_detection.inner_radius_px * mm_per_pixel,
        roundness_px=float(roundness["roundness_px"]),
        roundness_mm=float(roundness["roundness_px"]) * mm_per_pixel,
        max_abs_deviation_px=float(roundness["max_abs_deviation_px"]),
        max_abs_deviation_mm=float(roundness["max_abs_deviation_px"]) * mm_per_pixel,
        contour_points=inner_points,
        fitting_inliers=inner_points,
        signed_errors_px=roundness["signed_errors_px"],
        mm_per_pixel=mm_per_pixel,
    )


def detect_pipe_wall_rims(processed: Dict[str, np.ndarray], measurement: PipeMeasurement) -> Optional[PipeWallDetection]:
    image_bgr = processed["full_image_bgr"]
    x1, y1, x2, y2 = processed.get("roi_bbox", (0, 0, image_bgr.shape[1], image_bgr.shape[0]))
    roi_bgr = image_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    height, width = gray.shape[:2]
    center_roi = np.array([measurement.center_x_px - x1, measurement.center_y_px - y1], dtype=np.float32)
    inner_radius = float(measurement.radius_px)
    if inner_radius <= 0:
        return None

    otsu_value, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = min(255.0, float(otsu_value) + 12.0)
    angles = np.linspace(0, 2.0 * math.pi, 720, endpoint=False)
    outer_points_roi = []
    outer_radii = []

    for angle in angles:
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        r_start = max(inner_radius * 1.08, inner_radius + 12.0)
        r_end = min(inner_radius * 2.4, max(width, height))
        radii = np.linspace(r_start, r_end, 460)
        xs = np.round(center_roi[0] + radii * direction[0]).astype(int)
        ys = np.round(center_roi[1] + radii * direction[1]).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.count_nonzero(valid) < 20:
            continue

        radii_v = radii[valid]
        xs_v = xs[valid]
        ys_v = ys[valid]
        values = blur[ys_v, xs_v].astype(np.float32)
        gradients = np.diff(values)
        if len(gradients) == 0:
            continue

        candidate_indices = np.where((values[:-1] < threshold) & (values[1:] >= threshold))[0]
        if len(candidate_indices) > 0:
            idx = int(candidate_indices[0] + 1)
        else:
            idx = int(np.argmax(gradients) + 1)
            if gradients[idx - 1] < 8.0:
                continue

        outer_r = float(radii_v[idx])
        if outer_r <= inner_radius * 1.15:
            continue
        outer_radii.append(outer_r)
        outer_points_roi.append([float(xs_v[idx]), float(ys_v[idx])])

    if len(outer_points_roi) < 40:
        measured_outer_radius = inner_radius
        inner_points_roi = []
        inner_radii = []
        for angle in angles:
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            r_start = max(8.0, measured_outer_radius * 0.25)
            r_end = measured_outer_radius * 0.96
            radii = np.linspace(r_start, r_end, 460)
            xs = np.round(center_roi[0] + radii * direction[0]).astype(int)
            ys = np.round(center_roi[1] + radii * direction[1]).astype(int)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if np.count_nonzero(valid) < 20:
                continue

            radii_v = radii[valid]
            xs_v = xs[valid]
            ys_v = ys[valid]
            values = blur[ys_v, xs_v].astype(np.float32)
            gradients = np.diff(values)
            if len(gradients) == 0:
                continue

            candidate_indices = np.where((values[:-1] < threshold) & (values[1:] >= threshold))[0]
            if len(candidate_indices) > 0:
                idx = int(candidate_indices[0] + 1)
            else:
                idx = int(np.argmax(gradients) + 1)
                if gradients[idx - 1] < 8.0:
                    continue

            inner_r = float(radii_v[idx])
            if inner_r >= measured_outer_radius * 0.94 or inner_r <= measured_outer_radius * 0.35:
                continue
            inner_radii.append(inner_r)
            inner_points_roi.append([float(xs_v[idx]), float(ys_v[idx])])

        if len(inner_points_roi) < 40:
            return None

        inner_points_roi_array = np.array(inner_points_roi, dtype=np.float32)
        inner_points_full = inner_points_roi_array + np.array([x1, y1], dtype=np.float32)
        inner_center_x, inner_center_y, inner_radius_detected = fit_circle_with_outlier_trim(inner_points_full)
        inner_radii_array = np.array(inner_radii, dtype=np.float32)
        wall_thicknesses = measured_outer_radius - inner_radii_array
        thickness_median = float(np.median(wall_thicknesses))
        thickness_span = float(np.max(wall_thicknesses) - np.min(wall_thicknesses))
        if thickness_median <= 0 or thickness_span > max(80.0, thickness_median * 0.85):
            return None

        return PipeWallDetection(
            inner_center_x_px=float(inner_center_x),
            inner_center_y_px=float(inner_center_y),
            inner_radius_px=inner_radius_detected,
            inner_points=inner_points_full,
            outer_center_x_px=measurement.center_x_px,
            outer_center_y_px=measurement.center_y_px,
            outer_radius_px=measured_outer_radius,
            outer_points=measurement.contour_points.astype(np.float32),
            wall_thickness_px=thickness_median,
            wall_thickness_min_px=float(np.min(wall_thicknesses)),
            wall_thickness_max_px=float(np.max(wall_thicknesses)),
            center_offset_px=0.0,
            method="radial inward inner rim scan",
        )

    wall_thicknesses = np.array(outer_radii, dtype=np.float32) - inner_radius
    thickness_median = float(np.median(wall_thicknesses))
    thickness_span = float(np.max(wall_thicknesses) - np.min(wall_thicknesses))
    outer_points_roi_array = np.array(outer_points_roi, dtype=np.float32)
    outer_points_full = outer_points_roi_array + np.array([x1, y1], dtype=np.float32)
    outer_center_x, outer_center_y, outer_radius = fit_circle_with_outlier_trim(outer_points_full)
    radius_ratio = outer_radius / inner_radius if inner_radius > 0 else float("inf")
    if outer_radius > min(width, height) * 0.44 or radius_ratio > 1.75 or thickness_span > max(80.0, thickness_median * 0.75):
        return None

    outer_center_full = np.array([outer_center_x, outer_center_y], dtype=np.float32)
    center_offset = float(
        np.linalg.norm(outer_center_full - np.array([measurement.center_x_px, measurement.center_y_px], dtype=np.float32))
    )

    return PipeWallDetection(
        inner_center_x_px=measurement.center_x_px,
        inner_center_y_px=measurement.center_y_px,
        inner_radius_px=inner_radius,
        inner_points=measurement.contour_points.astype(np.float32),
        outer_center_x_px=float(outer_center_full[0]),
        outer_center_y_px=float(outer_center_full[1]),
        outer_radius_px=outer_radius,
        outer_points=outer_points_full,
        wall_thickness_px=thickness_median,
        wall_thickness_min_px=float(np.min(wall_thicknesses)),
        wall_thickness_max_px=float(np.max(wall_thicknesses)),
        center_offset_px=center_offset,
        method="radial inner/outer rim scan",
    )


def measure_pipe_roundness_pixels(
    filename: str,
    image_bytes: bytes,
    calibration: Optional[CameraCalibration],
    preprocess_config: PreprocessConfig,
    contour_config: ContourConfig,
    ransac_config: RansacConfig,
    use_auto_roi: bool = True,
) -> Tuple[PipeMeasurement, Dict[str, np.ndarray], str]:
    image = decode_image(image_bytes)
    if image is None:
        raise ValueError(f"Could not decode image: {filename}")

    corrected = undistort_image(image, calibration)
    roi_bbox = (0, 0, corrected.shape[1], corrected.shape[0])
    roi_note = "full image"
    roi_image = corrected
    if use_auto_roi:
        roi_bbox, roi_note = detect_pipe_roi(corrected)
        x1, y1, x2, y2 = roi_bbox
        roi_image = corrected[y1:y2, x1:x2]

    processed = preprocess_image(roi_image, None, preprocess_config)
    processed["full_image_bgr"] = corrected
    processed["roi_bbox"] = roi_bbox
    processed["roi_note"] = roi_note

    points_roi = select_pipe_contour(processed["clean_edges"], processed["image_bgr"].shape, contour_config)
    circle_roi, inlier_indices = fit_circle_ransac(points_roi, ransac_config)
    roundness = compute_roundness(points_roi, circle_roi)

    x1, y1, _, _ = roi_bbox
    points_full = points_roi + np.array([x1, y1], dtype=np.float32)
    inliers_full = points_full[inlier_indices]
    xc_roi, yc_roi, radius_px = circle_roi
    measurement = PipeMeasurement(
        filename=filename,
        center_x_px=float(xc_roi + x1),
        center_y_px=float(yc_roi + y1),
        radius_px=float(radius_px),
        diameter_px=float(2.0 * radius_px),
        diameter_mm=float(2.0 * radius_px),
        roundness_px=float(roundness["roundness_px"]),
        roundness_mm=float(roundness["roundness_px"]),
        max_abs_deviation_px=float(roundness["max_abs_deviation_px"]),
        max_abs_deviation_mm=float(roundness["max_abs_deviation_px"]),
        contour_points=points_full,
        fitting_inliers=inliers_full,
        signed_errors_px=roundness["signed_errors_px"],
        mm_per_pixel=1.0,
    )
    return measurement, processed, roi_note


def select_hough_annulus_points(processed: Dict[str, np.ndarray], contour_config: ContourConfig) -> np.ndarray:
    blurred = processed["blurred"]
    edges = processed["clean_edges"]
    height, width = edges.shape[:2]
    min_dim = min(width, height)
    min_radius = int(max(5, min_dim * contour_config.min_radius_fraction))
    max_radius = int(max(min_radius + 1, min_dim * contour_config.max_radius_fraction))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim * 0.25),
        param1=120,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        raise ValueError("Hough fallback found no circles.")

    cx, cy, radius = max(np.round(circles[0]).astype(np.float32), key=lambda c: c[2])
    ys, xs = np.nonzero(edges)
    edge_points = np.column_stack((xs, ys)).astype(np.float32)
    if len(edge_points) == 0:
        raise ValueError("Hough fallback found no edge pixels.")

    radial_distances = np.linalg.norm(edge_points - np.array([cx, cy], dtype=np.float32), axis=1)
    annulus_width = max(4.0, float(radius) * 0.045)
    annulus_points = edge_points[np.abs(radial_distances - radius) <= annulus_width]
    if len(annulus_points) < max(40, contour_config.min_points):
        raise ValueError(f"Hough fallback found only {len(annulus_points)} edge points near the outer circle.")
    return circular_smooth_points(annulus_points, contour_config.circular_smoothing_window)


def measure_pipe_roundness_pixels_robust(
    filename: str,
    image_bytes: bytes,
    calibration: Optional[CameraCalibration],
    preprocess_config: PreprocessConfig,
) -> Tuple[PipeMeasurement, Dict[str, np.ndarray], str]:
    attempts = [
        ("default contour settings", ContourConfig(), RansacConfig(max_iterations=1000, min_inlier_ratio=0.35)),
        (
            "relaxed contour settings",
            ContourConfig(min_area=30.0, min_points=40, min_radius_fraction=0.015, max_radius_fraction=0.78),
            RansacConfig(max_iterations=1200, min_inlier_ratio=0.20, threshold_radius_fraction=0.012),
        ),
    ]

    errors = []
    for note, contour_config, ransac_config in attempts:
        try:
            measurement, processed, roi_note = measure_pipe_roundness_pixels(
                filename,
                image_bytes,
                calibration,
                preprocess_config,
                contour_config,
                ransac_config,
            )
            return measurement, processed, f"{note}; {roi_note}"
        except ValueError as exc:
            errors.append(f"{note}: {exc}")

    try:
        image = decode_image(image_bytes)
        if image is None:
            raise ValueError(f"Could not decode image: {filename}")
        processed = preprocess_image(image, calibration, preprocess_config)
        contour_config = ContourConfig(min_points=40, min_radius_fraction=0.015, max_radius_fraction=0.85)
        points = select_hough_annulus_points(processed, contour_config)
        circle, inlier_indices = fit_circle_ransac(
            points,
            RansacConfig(max_iterations=1500, min_inlier_ratio=0.18, threshold_radius_fraction=0.014),
        )
        roundness = compute_roundness(points, circle)
        xc, yc, radius_px = circle
        measurement = PipeMeasurement(
            filename=filename,
            center_x_px=xc,
            center_y_px=yc,
            radius_px=radius_px,
            diameter_px=2.0 * radius_px,
            diameter_mm=2.0 * radius_px,
            roundness_px=float(roundness["roundness_px"]),
            roundness_mm=float(roundness["roundness_px"]),
            max_abs_deviation_px=float(roundness["max_abs_deviation_px"]),
            max_abs_deviation_mm=float(roundness["max_abs_deviation_px"]),
            contour_points=points,
            fitting_inliers=points[inlier_indices],
            signed_errors_px=roundness["signed_errors_px"],
            mm_per_pixel=1.0,
        )
        processed["full_image_bgr"] = processed["image_bgr"]
        processed["roi_bbox"] = (0, 0, processed["image_bgr"].shape[1], processed["image_bgr"].shape[0])
        processed["roi_note"] = "Hough-assisted full image"
        return measurement, processed, "Hough-assisted edge annulus fallback"
    except ValueError as exc:
        errors.append(f"Hough fallback: {exc}")
        raise ValueError("Pipe outer-circle detection failed. " + " | ".join(errors)) from exc


def convert_measurement_scale(measurement: PipeMeasurement, mm_per_pixel: float) -> PipeMeasurement:
    return PipeMeasurement(
        filename=measurement.filename,
        center_x_px=measurement.center_x_px,
        center_y_px=measurement.center_y_px,
        radius_px=measurement.radius_px,
        diameter_px=measurement.diameter_px,
        diameter_mm=measurement.diameter_px * mm_per_pixel,
        roundness_px=measurement.roundness_px,
        roundness_mm=measurement.roundness_px * mm_per_pixel,
        max_abs_deviation_px=measurement.max_abs_deviation_px,
        max_abs_deviation_mm=measurement.max_abs_deviation_px * mm_per_pixel,
        contour_points=measurement.contour_points,
        fitting_inliers=measurement.fitting_inliers,
        signed_errors_px=measurement.signed_errors_px,
        mm_per_pixel=mm_per_pixel,
    )


def compute_mm_per_pixel_from_known_diameter(diameter_px: float, known_diameter_mm: float) -> float:
    if diameter_px <= 0 or known_diameter_mm <= 0:
        raise ValueError("Reference diameter in pixels and millimeters must be positive.")
    return float(known_diameter_mm) / float(diameter_px)


def parse_tolerance_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    clean = str(text).replace(",", ".").replace(" ", "")
    plus_minus = chr(177)
    percent_patterns = [rf"{plus_minus}([0-9.]+)%", r"\+/-([0-9.]+)%", r"\+\-([0-9.]+)%", r"([0-9.]+)%"]
    min_patterns = [
        rf"min{plus_minus}([0-9.]+)",
        r"min\+/-([0-9.]+)",
        r"min\+\-([0-9.]+)",
        r"min\.?(?:imum)?[+-/]*([0-9.]+)",
    ]
    percent = None
    min_abs = None
    for pattern in percent_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            percent = float(match.group(1))
            break
    for pattern in min_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            min_abs = float(match.group(1))
            break
    return percent, min_abs


def default_diameter_tolerance_class(diameter_class: str) -> Dict[str, object]:
    defaults = {
        "D2": {"percent": 1.0, "min_abs_mm": 0.5, "text": "D2 +/- 1.0% min +/- 0.5"},
        "D3": {"percent": 0.75, "min_abs_mm": 0.3, "text": "D3 +/- 0.75% min +/- 0.3"},
        "D4": {"percent": 0.5, "min_abs_mm": 0.1, "text": "D4 +/- 0.5% min +/- 0.1"},
    }
    key = diameter_class.upper().strip()
    if key not in defaults:
        raise ValueError(f"Unsupported diameter tolerance class: {diameter_class}")
    return defaults[key]


def extract_dclass_text_from_sheet(excel_bytes: bytes, sheet_name: str, diameter_class: str) -> Optional[str]:
    workbook = pd.ExcelFile(io.BytesIO(excel_bytes))
    selected_sheet = sheet_name if sheet_name in workbook.sheet_names else workbook.sheet_names[0]
    df = pd.read_excel(workbook, sheet_name=selected_sheet, header=None)
    target = diameter_class.upper().strip()
    for value in df.values.flatten():
        text = str(value)
        clean = text.upper().replace(" ", "")
        if target in clean and "%" in clean:
            percent, _ = parse_tolerance_text(text)
            if percent is not None:
                return text
    return None


def build_dclass_row(nominal_diameter_mm: float, diameter_class: str, excel_bytes: bytes, sheet_name: str) -> Dict[str, object]:
    selected = default_diameter_tolerance_class(diameter_class)
    source = "built-in D-class default"
    tolerance_text = selected["text"]
    excel_text = extract_dclass_text_from_sheet(excel_bytes, sheet_name, diameter_class)
    if excel_text:
        percent, min_abs = parse_tolerance_text(excel_text)
        if percent is not None:
            selected = {
                "percent": percent,
                "min_abs_mm": min_abs if min_abs is not None else 0.0,
                "text": excel_text,
            }
            source = f"Excel sheet {sheet_name}"
            tolerance_text = excel_text

    percent = float(selected["percent"])
    min_abs = float(selected["min_abs_mm"])
    abs_tol = max(nominal_diameter_mm * percent / 100.0, min_abs)
    return {
        "class": diameter_class.upper(),
        "rule": tolerance_text,
        "source": source,
        "percent": percent,
        "min_abs_mm": min_abs,
        "abs_tol_mm": abs_tol,
        "min_diameter_mm": nominal_diameter_mm - abs_tol,
        "max_diameter_mm": nominal_diameter_mm + abs_tol,
    }


def _cell_to_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _normalise_header(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_direct_table_columns(row_values) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    normalised = [_normalise_header(value) for value in row_values]
    dia_col = max_col = min_col = None

    for col_idx, value in enumerate(normalised):
        if value in {"diamm", "dianom", "diamnom", "diameter", "diam"}:
            dia_col = col_idx
        elif value in {"diamax", "dmax", "diametermax", "maxdia"}:
            max_col = col_idx
        elif value in {"diamin", "dmin", "diametermin", "mindia"}:
            min_col = col_idx

    # Reducer/special sheets often use D, Dmax, Dmin instead of Dia mm.
    if dia_col is None and max_col is not None and min_col is not None:
        for col_idx, value in enumerate(normalised):
            if value in {"d", "od", "o", "omm"}:
                dia_col = col_idx
                break

    return dia_col, max_col, min_col


def find_direct_diameter_limits(
    excel_bytes: bytes,
    sheet_name: str,
    nominal_diameter_mm: float,
    match_tolerance_mm: Optional[float] = None,
) -> Dict[str, object]:
    workbook = pd.ExcelFile(io.BytesIO(excel_bytes))
    selected_sheet = sheet_name if sheet_name in workbook.sheet_names else workbook.sheet_names[0]
    df = pd.read_excel(workbook, sheet_name=selected_sheet, header=None)
    best_table_match = None

    for header_idx in range(len(df)):
        dia_col, max_col, min_col = _find_direct_table_columns(df.iloc[header_idx].values)
        if dia_col is None or max_col is None or min_col is None:
            continue

        best = None
        best_diff = float("inf")
        for data_idx in range(header_idx + 1, len(df)):
            dia = _cell_to_float(df.iloc[data_idx, dia_col])
            dia_max = _cell_to_float(df.iloc[data_idx, max_col])
            dia_min = _cell_to_float(df.iloc[data_idx, min_col])
            if dia is None or dia_max is None or dia_min is None:
                continue
            diff = abs(dia - nominal_diameter_mm)
            if diff < best_diff:
                best_diff = diff
                best = {
                    "sheet_name": selected_sheet,
                    "row_index": int(data_idx),
                    "table_nominal_mm": dia,
                    "min_diameter_mm": dia_min,
                    "max_diameter_mm": dia_max,
                    "difference_to_requested_nominal_mm": diff,
                }
        if best is not None:
            if best_table_match is None or best["difference_to_requested_nominal_mm"] < best_table_match["difference_to_requested_nominal_mm"]:
                best_table_match = best

    if best_table_match is not None:
        allowed_difference = match_tolerance_mm
        if allowed_difference is None:
            allowed_difference = max(2.0, nominal_diameter_mm * 0.03)
        if best_table_match["difference_to_requested_nominal_mm"] <= allowed_difference:
            return best_table_match
        raise ValueError(
            f"Sheet {sheet_name} has a readable diameter table, but no nominal diameter close to "
            f"{nominal_diameter_mm:.3f} mm. Closest table nominal is "
            f"{best_table_match['table_nominal_mm']:.3f} mm at row {best_table_match['row_index']} "
            f"(difference {best_table_match['difference_to_requested_nominal_mm']:.3f} mm). "
            "Choose a standard/sheet that covers the pipe size."
        )

    raise ValueError(
        f"No readable diameter limit table found in sheet {sheet_name}. "
        "Expected columns like Dia mm/Dia max/Dia min, Dia.nom/Dia.max/Dia.min, or D/Dmax/Dmin."
    )


def add_tolerance_diagnostics(report: Dict[str, object], measurement: PipeMeasurement) -> Dict[str, object]:
    table = report["tolerance_table"]
    row = table.iloc[0] if report["mode"] == "DIRECT_TABLE" else table[table["class"] == report["best_diameter_class"]].iloc[0] if report["best_diameter_class"] != "NONE" else table.iloc[0]
    min_dia = float(row["min_diameter_mm"])
    max_dia = float(row["max_diameter_mm"])
    diameter_ok = bool(min_dia <= measurement.diameter_mm <= max_dia)
    roundness_ok = bool(measurement.roundness_mm <= report["max_ovality_mm"])

    if measurement.diameter_mm < min_dia:
        diameter_fail_amount = min_dia - measurement.diameter_mm
        diameter_reason = f"Diameter is {diameter_fail_amount:.4f} mm below the minimum."
    elif measurement.diameter_mm > max_dia:
        diameter_fail_amount = measurement.diameter_mm - max_dia
        diameter_reason = f"Diameter is {diameter_fail_amount:.4f} mm above the maximum."
    else:
        diameter_fail_amount = 0.0
        diameter_reason = "Diameter is within the allowed range."

    roundness_excess = max(0.0, measurement.roundness_mm - float(report["max_ovality_mm"]))
    roundness_reason = (
        f"Roundness exceeds the limit by {roundness_excess:.4f} mm."
        if roundness_excess > 0
        else "Roundness is within the allowed limit."
    )
    if diameter_ok and roundness_ok:
        failure_reason = "PASS: diameter and roundness are both within tolerance."
    elif not diameter_ok and not roundness_ok:
        failure_reason = f"FAIL: {diameter_reason} {roundness_reason}"
    elif not diameter_ok:
        failure_reason = f"FAIL: {diameter_reason}"
    else:
        failure_reason = f"FAIL: {roundness_reason}"

    report.update(
        {
            "min_diameter_mm": min_dia,
            "max_diameter_mm": max_dia,
            "diameter_ok": diameter_ok,
            "roundness_ok": roundness_ok,
            "diameter_fail_amount_mm": diameter_fail_amount,
            "roundness_excess_mm": roundness_excess,
            "failure_reason": failure_reason,
        }
    )
    return report


def build_standard_tolerance_report(
    measurement: PipeMeasurement,
    nominal_diameter_mm: float,
    standard_label: str,
    excel_bytes: bytes,
    fallback_ovality_fraction: float,
) -> Dict[str, object]:
    if nominal_diameter_mm <= 0:
        raise ValueError("Nominal diameter must be positive.")
    config = STANDARD_LIBRARY[standard_label]
    sheet_name = config["sheet"]
    mode = config["mode"]
    max_ovality_mm = nominal_diameter_mm * fallback_ovality_fraction

    if mode == "EN_DCLASS":
        rows = []
        for cls in ["D4", "D3", "D2"]:
            row = build_dclass_row(nominal_diameter_mm, cls, excel_bytes, sheet_name)
            row["diameter_ok"] = bool(row["min_diameter_mm"] <= measurement.diameter_mm <= row["max_diameter_mm"])
            rows.append(row)
        table = pd.DataFrame(rows)
        roundness_ok = bool(measurement.roundness_mm <= max_ovality_mm)
        best_diameter_class = next((r["class"] for r in rows if r["diameter_ok"]), "NONE")
        best_overall_class = next((r["class"] for r in rows if r["diameter_ok"] and roundness_ok), "NONE")
        report = {
            "standard_label": standard_label,
            "mode": mode,
            "sheet_name": sheet_name,
            "nominal_mm": nominal_diameter_mm,
            "tolerance_table": table,
            "max_ovality_mm": max_ovality_mm,
            "ovality_source": f"fallback {fallback_ovality_fraction * 100:.2f}% of nominal",
            "best_diameter_class": best_diameter_class,
            "best_overall_class": best_overall_class,
            "overall": best_overall_class != "NONE",
        }
    else:
        limits = find_direct_diameter_limits(excel_bytes, sheet_name, nominal_diameter_mm)
        diameter_ok = bool(limits["min_diameter_mm"] <= measurement.diameter_mm <= limits["max_diameter_mm"])
        roundness_ok = bool(measurement.roundness_mm <= max_ovality_mm)
        table = pd.DataFrame(
            [
                {
                    "class": "DIRECT_TABLE",
                    "rule": f"Dia min/max from sheet {sheet_name}, row {limits['row_index']}",
                    "table_nominal_mm": limits["table_nominal_mm"],
                    "min_diameter_mm": limits["min_diameter_mm"],
                    "max_diameter_mm": limits["max_diameter_mm"],
                    "diameter_ok": diameter_ok,
                }
            ]
        )
        report = {
            "standard_label": standard_label,
            "mode": mode,
            "sheet_name": sheet_name,
            "nominal_mm": nominal_diameter_mm,
            "tolerance_table": table,
            "direct_limits": limits,
            "max_ovality_mm": max_ovality_mm,
            "ovality_source": f"fallback {fallback_ovality_fraction * 100:.2f}% of requested nominal",
            "best_diameter_class": "DIRECT_TABLE" if diameter_ok else "NONE",
            "best_overall_class": "DIRECT_TABLE" if diameter_ok and roundness_ok else "NONE",
            "overall": diameter_ok and roundness_ok,
        }

    return add_tolerance_diagnostics(report, measurement)


def standard_display_name(standard: Dict[str, object]) -> str:
    title = STANDARD_UI_TITLE_BY_ID.get(str(standard["id"]), str(standard["name"]))
    return f"{standard['id']} | {standard.get('type', 'standard')} | {title}"


def entry_display_name(entry: Dict[str, object]) -> str:
    parts = [f"row {entry['source_row']}"]
    if "nominal_mm" in entry:
        parts.append(f"nominal {float(entry['nominal_mm']):g} mm")
    if "nominal_inch" in entry:
        parts.append(f"NPS {entry['nominal_inch']}")
    if "secondary_nominal_mm" in entry:
        parts.append(f"secondary {float(entry['secondary_nominal_mm']):g} mm")
    if "diameter_pair_mm" in entry:
        pair = entry["diameter_pair_mm"]
        pair_text = " / ".join(f"{float(value):g} mm" for value in pair.values())
        parts.append(f"pair {pair_text}")
    if "dn" in entry:
        parts.append(f"DN {entry['dn']}")
    return " | ".join(parts)


def standard_range_mm(standard: Dict[str, object], selected_ot_code: Optional[str] = None) -> Optional[Tuple[float, float]]:
    ranges = []
    for ot in standard.get("ot_codes", []):
        if selected_ot_code is not None and ot.get("code") != selected_ot_code:
            continue
        if "range_mm" in ot:
            ranges.append(tuple(float(value) for value in ot["range_mm"]))
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def tolerance_classes(standard: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return (standard.get("diameter_tolerance") or {}).get("classes", {})


def compute_contract_diameter_row(nominal_mm: float, class_name: str, class_rule: Dict[str, object]) -> Dict[str, object]:
    percent = float(class_rule["percent"])
    min_mm = float(class_rule["min_mm"])
    percent_branch = nominal_mm * percent / 100.0
    abs_tol = max(percent_branch, min_mm)
    return {
        "class": class_name,
        "nominal_mm": nominal_mm,
        "percent": percent,
        "min_mm": min_mm,
        "percent_branch_mm": percent_branch,
        "applied_branch": "percent" if percent_branch >= min_mm else "minimum",
        "abs_tol_mm": abs_tol,
        "min_diameter_mm": nominal_mm - abs_tol,
        "max_diameter_mm": nominal_mm + abs_tol,
        "source_text": class_rule.get("source_text", ""),
    }


def nominal_candidates_from_entry(entry: Dict[str, object]) -> List[Tuple[str, float]]:
    candidates = []
    if "nominal_mm" in entry:
        candidates.append(("nominal_mm", float(entry["nominal_mm"])))
    if "diameter_pair_mm" in entry:
        for key, value in entry["diameter_pair_mm"].items():
            candidates.append((f"diameter_pair_{key}", float(value)))
    return [(label, value) for label, value in candidates if value > 0]


def explicit_lookup_range_from_entry(entry: Optional[Dict[str, object]]) -> Optional[Tuple[float, float]]:
    if not entry:
        return None
    pair = entry.get("diameter_pair_mm")
    if isinstance(pair, dict):
        values = [float(value) for value in pair.values() if float(value) > 0]
        if len(values) >= 2:
            return min(values), max(values)
    return None


def find_diameter_lookup_row(
    standard: Dict[str, object],
    nominal_mm: Optional[float],
    selected_entry: Optional[Dict[str, object]],
    diameter_lookup: Dict[str, object],
) -> Optional[Dict[str, object]]:
    standard_id = str(standard.get("id"))
    standard_lookup = (diameter_lookup.get("standards") or {}).get(standard_id)
    if not standard_lookup:
        return None
    entries = standard_lookup.get("entries") or []
    if not entries or nominal_mm is None or not np.isfinite(float(nominal_mm)):
        return None
    nominal = float(nominal_mm)
    selected_source_row = selected_entry.get("source_row") if isinstance(selected_entry, dict) else None

    def score(entry: Dict[str, object]) -> Tuple[int, float]:
        row_penalty = 0 if selected_source_row is not None and entry.get("source_row") == selected_source_row else 1
        candidates = []
        if entry.get("nominal_mm") is not None:
            candidates.append(abs(float(entry["nominal_mm"]) - nominal))
        if entry.get("secondary_nominal_mm") is not None:
            candidates.append(abs(float(entry["secondary_nominal_mm"]) - nominal))
        return row_penalty, min(candidates) if candidates else float("inf")

    best = min(entries, key=score)
    row_penalty, diff = score(best)
    if diff > max(0.001, abs(nominal) * 0.00001) and row_penalty > 0:
        return None
    if best.get("min_diameter_mm") is None or best.get("max_diameter_mm") is None:
        return None
    return best


def diameter_lookup_source_text(row: Dict[str, object], standard_id: str) -> str:
    source = str(row.get("source", "diameter_lookup"))
    formula = row.get("formula")
    text = (
        f"diameter_lookup.json standard {standard_id}, source row {row.get('source_row', 'unknown')}; "
        f"{source}; allowed diameter {float(row['min_diameter_mm']):.4f}-"
        f"{float(row['max_diameter_mm']):.4f} mm"
    )
    if formula:
        text = f"{text}; formula: {formula}"
    return text


def evaluate_ovality_tolerance(
    standard: Dict[str, object],
    tolerance_row: Optional[Dict[str, object]],
    nominal_mm: Optional[float],
    measured_ovality_mm: Optional[float],
    selected_entry: Optional[Dict[str, object]] = None,
    ovality_lookup: Optional[Dict[str, object]] = None,
    selected_wall_thickness_mm: Optional[float] = None,
    selected_wall_schedule: Optional[str] = None,
) -> Dict[str, object]:
    rule = standard.get("ovality_tolerance")
    if not isinstance(rule, dict):
        return {
            "ovality_evaluated": False,
            "roundness_ok": None,
            "max_ovality_mm": float("nan"),
            "roundness_excess_mm": float("nan"),
            "ovality_source": "not mapped in standards contract",
        }

    mode = str(rule.get("mode", "")).strip()
    source = str(rule.get("source_text", rule.get("note", mode or "not specified")))
    if mode in {"excluded", "requires_source_enrichment"}:
        return {
            "ovality_evaluated": False,
            "roundness_ok": None,
            "max_ovality_mm": float("nan"),
            "roundness_excess_mm": float("nan"),
            "ovality_source": source,
        }

    if measured_ovality_mm is None or not np.isfinite(float(measured_ovality_mm)):
        return {
            "ovality_evaluated": False,
            "roundness_ok": None,
            "max_ovality_mm": float("nan"),
            "roundness_excess_mm": float("nan"),
            "ovality_source": f"{source}; measured MZC roundness not available",
        }

    limit_mm = float("nan")
    if mode == "formula":
        if nominal_mm is not None and np.isfinite(float(nominal_mm)) and "percent_of_od" in rule:
            limit_mm = float(nominal_mm) * float(rule["percent_of_od"]) / 100.0
    elif mode == "diameter_tolerance_range" and tolerance_row is not None:
        max_diameter = float(tolerance_row["max_diameter_mm"])
        min_diameter = float(tolerance_row["min_diameter_mm"])
        limit_mm = max_diameter - min_diameter
        source = (
            f"{source}; formula: ovality_limit_mm = selected allowed max diameter "
            f"({max_diameter:.4f}) - selected allowed min diameter ({min_diameter:.4f}) "
            f"= {limit_mm:.4f} mm"
        )
    elif mode == "explicit" and "limit_mm" in rule:
        limit_mm = float(rule["limit_mm"])
    elif mode == "lookup_table":
        lookup_result = find_ovality_lookup_limit(
            standard,
            nominal_mm,
            selected_entry,
            ovality_lookup or {},
            selected_wall_thickness_mm,
            selected_wall_schedule,
        )
        limit_mm = lookup_result["limit_mm"]
        source = lookup_result["source"]

    if not np.isfinite(limit_mm):
        return {
            "ovality_evaluated": False,
            "roundness_ok": None,
            "max_ovality_mm": float("nan"),
            "roundness_excess_mm": float("nan"),
            "ovality_source": f"{source}; no deterministic ovality limit could be computed",
        }

    measured = float(measured_ovality_mm)
    excess = max(0.0, measured - limit_mm)
    return {
        "ovality_evaluated": True,
        "roundness_ok": bool(measured <= limit_mm),
        "max_ovality_mm": limit_mm,
        "roundness_excess_mm": excess,
        "ovality_source": source,
        "measured_ovality_mm": measured,
    }


def find_ovality_lookup_limit(
    standard: Dict[str, object],
    nominal_mm: Optional[float],
    selected_entry: Optional[Dict[str, object]],
    ovality_lookup: Dict[str, object],
    selected_wall_thickness_mm: Optional[float] = None,
    selected_wall_schedule: Optional[str] = None,
) -> Dict[str, object]:
    standard_id = str(standard.get("id"))
    standard_lookup = (ovality_lookup.get("standards") or {}).get(standard_id)
    if not standard_lookup:
        return {"limit_mm": float("nan"), "source": "ovality lookup table not found"}
    entries = standard_lookup.get("entries") or []
    if not entries or nominal_mm is None or not np.isfinite(float(nominal_mm)):
        return {"limit_mm": float("nan"), "source": "ovality lookup row not available"}

    nominal = float(nominal_mm)
    secondary = None
    if isinstance(selected_entry, dict) and selected_entry.get("secondary_nominal_mm") is not None:
        secondary = float(selected_entry["secondary_nominal_mm"])

    def entry_score(entry: Dict[str, object]) -> float:
        candidates = []
        if entry.get("nominal_mm") is not None:
            candidates.append(abs(float(entry["nominal_mm"]) - nominal))
        if entry.get("secondary_nominal_mm") is not None:
            candidates.append(abs(float(entry["secondary_nominal_mm"]) - nominal))
        if not candidates:
            return float("inf")
        return min(candidates)

    entry = min(entries, key=entry_score)
    if entry_score(entry) > max(0.001, abs(nominal) * 0.00001):
        return {"limit_mm": float("nan"), "source": "no matching ovality lookup row for selected nominal"}

    limit = entry.get("limit_mm")
    if limit is None and entry.get("primary_limit_mm") is not None:
        primary_diff = abs(float(entry.get("nominal_mm", nominal)) - nominal)
        secondary_diff = (
            abs(float(entry["secondary_nominal_mm"]) - nominal)
            if entry.get("secondary_nominal_mm") is not None
            else float("inf")
        )
        if secondary_diff < primary_diff and entry.get("secondary_limit_mm") is not None:
            limit = entry["secondary_limit_mm"]
        else:
            limit = entry["primary_limit_mm"]
    if limit is None and standard_id in {"3", "3.1"}:
        branch = entry.get("a999") or entry.get("a1016")
        if branch:
            wall_thickness = selected_wall_thickness_mm
            if wall_thickness is None and selected_wall_schedule:
                schedule = (entry.get("schedules") or {}).get(selected_wall_schedule)
                if isinstance(schedule, dict) and schedule.get("nominal_mm") not in {None, "N/A"}:
                    wall_thickness = float(schedule["nominal_mm"])
            if wall_thickness is None or not np.isfinite(float(wall_thickness)) or float(wall_thickness) <= 0:
                return {
                    "limit_mm": float("nan"),
                    "source": f"ovality_lookup.json standard {standard_id}, source row {entry.get('source_row', 'unknown')}; wall thickness/schedule required for ASTM thin-wall branch",
                }
            wall_ratio = float(wall_thickness) / nominal
            threshold = float(branch.get("wall_ratio_threshold", 0.03))
            if wall_ratio > threshold:
                limit = branch.get("thick_wall_limit_mm")
                branch_name = f"thick-wall branch, t/D={wall_ratio:.4f}"
            else:
                limit = branch.get("thin_wall_percent_of_od", 2.0) * nominal / 100.0
                branch_name = f"thin-wall branch, t/D={wall_ratio:.4f}"
            return {
                "limit_mm": float(limit) if limit is not None else float("nan"),
                "source": f"ovality_lookup.json standard {standard_id}, source row {entry.get('source_row', 'unknown')}; {branch_name}",
            }

    return {
        "limit_mm": float(limit) if limit is not None else float("nan"),
        "source": f"ovality_lookup.json standard {standard_id}, source row {entry.get('source_row', 'unknown')}",
    }


def build_contract_tolerance_report(
    measurement: PipeMeasurement,
    standard: Dict[str, object],
    nominal_values: List[Tuple[str, float]],
    selected_entry: Optional[Dict[str, object]],
    selected_classes: List[str],
    manual_tolerance_mm: Optional[float] = None,
    compliance_diameter_mm: Optional[float] = None,
    compliance_diameter_label: str = "Fitted diameter",
    measured_ovality_mm: Optional[float] = None,
    ovality_lookup: Optional[Dict[str, object]] = None,
    diameter_lookup: Optional[Dict[str, object]] = None,
    selected_wall_thickness_mm: Optional[float] = None,
    selected_wall_schedule: Optional[str] = None,
) -> Dict[str, object]:
    rows = []
    measured_diameter_mm = float(compliance_diameter_mm if compliance_diameter_mm is not None else measurement.diameter_mm)
    if not np.isfinite(measured_diameter_mm) or measured_diameter_mm <= 0:
        raise ValueError(f"{compliance_diameter_label} is not available for standards compliance.")
    classes = tolerance_classes(standard)
    for nominal_source, nominal_mm in nominal_values:
        for class_name in selected_classes:
            if class_name not in classes:
                continue
            row = compute_contract_diameter_row(nominal_mm, class_name, classes[class_name])
            row["nominal_source"] = nominal_source
            row["diameter_ok"] = bool(row["min_diameter_mm"] <= measured_diameter_mm <= row["max_diameter_mm"])
            ovality = evaluate_ovality_tolerance(standard, row, nominal_mm, measured_ovality_mm, selected_entry, ovality_lookup, selected_wall_thickness_mm, selected_wall_schedule)
            row["max_ovality_mm"] = ovality["max_ovality_mm"]
            row["roundness_ok"] = ovality["roundness_ok"]
            row["ovality_evaluated"] = ovality["ovality_evaluated"]
            row["ovality_source"] = ovality["ovality_source"]
            row["roundness_excess_mm"] = ovality["roundness_excess_mm"]
            rows.append(row)
        if not classes and manual_tolerance_mm is not None:
            abs_tol = float(manual_tolerance_mm)
            row = {
                "class": "MANUAL",
                "nominal_mm": nominal_mm,
                "nominal_source": nominal_source,
                "percent": None,
                "min_mm": None,
                "percent_branch_mm": None,
                "applied_branch": "manual",
                "abs_tol_mm": abs_tol,
                "min_diameter_mm": nominal_mm - abs_tol,
                "max_diameter_mm": nominal_mm + abs_tol,
                "source_text": "Approved manual tolerance, not from the extracted standard",
            }
            row["diameter_ok"] = bool(row["min_diameter_mm"] <= measured_diameter_mm <= row["max_diameter_mm"])
            ovality = evaluate_ovality_tolerance(standard, row, nominal_mm, measured_ovality_mm, selected_entry, ovality_lookup, selected_wall_thickness_mm, selected_wall_schedule)
            row["max_ovality_mm"] = ovality["max_ovality_mm"]
            row["roundness_ok"] = ovality["roundness_ok"]
            row["ovality_evaluated"] = ovality["ovality_evaluated"]
            row["ovality_source"] = ovality["ovality_source"]
            row["roundness_excess_mm"] = ovality["roundness_excess_mm"]
            rows.append(row)
        if not classes and manual_tolerance_mm is None:
            lookup = find_diameter_lookup_row(standard, nominal_mm, selected_entry, diameter_lookup or {})
            if lookup is not None:
                row = {
                    "class": "DIAMETER_LOOKUP",
                    "nominal_mm": float(lookup.get("nominal_mm", nominal_mm)),
                    "nominal_source": nominal_source,
                    "percent": None,
                    "min_mm": None,
                    "percent_branch_mm": None,
                    "applied_branch": "lookup",
                    "abs_tol_mm": float(lookup["diameter_range_mm"]) / 2.0 if lookup.get("diameter_range_mm") is not None else None,
                    "min_diameter_mm": float(lookup["min_diameter_mm"]),
                    "max_diameter_mm": float(lookup["max_diameter_mm"]),
                    "source_text": diameter_lookup_source_text(lookup, str(standard.get("id"))),
                }
                row["diameter_ok"] = bool(row["min_diameter_mm"] <= measured_diameter_mm <= row["max_diameter_mm"])
                ovality = evaluate_ovality_tolerance(standard, row, row["nominal_mm"], measured_ovality_mm, selected_entry, ovality_lookup, selected_wall_thickness_mm, selected_wall_schedule)
                row["max_ovality_mm"] = ovality["max_ovality_mm"]
                row["roundness_ok"] = ovality["roundness_ok"]
                row["ovality_evaluated"] = ovality["ovality_evaluated"]
                row["ovality_source"] = ovality["ovality_source"]
                row["roundness_excess_mm"] = ovality["roundness_excess_mm"]
                rows.append(row)

    selected_entry_summary = selected_entry or {}
    if not rows:
        lookup_range = explicit_lookup_range_from_entry(selected_entry)
        if lookup_range is not None:
            min_diameter_mm, max_diameter_mm = lookup_range
            diameter_ok = bool(min_diameter_mm <= measured_diameter_mm <= max_diameter_mm)
            lookup_row = {
                "min_diameter_mm": min_diameter_mm,
                "max_diameter_mm": max_diameter_mm,
            }
            nominal_for_ovality = nominal_values[0][1] if nominal_values else (min_diameter_mm + max_diameter_mm) / 2.0
            ovality = evaluate_ovality_tolerance(standard, lookup_row, nominal_for_ovality, measured_ovality_mm, selected_entry, ovality_lookup, selected_wall_thickness_mm, selected_wall_schedule)
            roundness_ok = ovality["roundness_ok"]
            overall = bool(diameter_ok and (roundness_ok is not False))
            failure_reason = (
                f"LOOKUP RANGE MATCH: {compliance_diameter_label} {measured_diameter_mm:.3f} mm is inside the "
                f"selected row range {min_diameter_mm:.3f}-{max_diameter_mm:.3f} mm. "
                "This is a row-range check, not a formula tolerance check."
                if diameter_ok
                else f"LOOKUP RANGE MISS: {compliance_diameter_label} {measured_diameter_mm:.3f} mm is outside the "
                f"selected row range {min_diameter_mm:.3f}-{max_diameter_mm:.3f} mm. "
                "This is a row-range check, not a formula tolerance check."
            )
            table = pd.DataFrame(
                [
                    {
                        "class": "LOOKUP_RANGE",
                        "nominal_mm": nominal_values[0][1] if nominal_values else float("nan"),
                        "min_diameter_mm": min_diameter_mm,
                        "max_diameter_mm": max_diameter_mm,
                        "diameter_ok": diameter_ok,
                        "max_ovality_mm": ovality["max_ovality_mm"],
                        "roundness_ok": roundness_ok,
                        "ovality_evaluated": ovality["ovality_evaluated"],
                        "source_text": "Explicit min/max range from the selected lookup row",
                    }
                ]
            )
            return {
                "standard_label": standard_display_name(standard),
                "standard_id": standard["id"],
                "sheet_name": standard["sheet"],
                "mode": standard["mode"],
                "nominal_mm": nominal_values[0][1] if nominal_values else (min_diameter_mm + max_diameter_mm) / 2.0,
                "tolerance_table": table,
                "best_diameter_class": "LOOKUP_RANGE" if diameter_ok else "NONE",
                "best_overall_class": "LOOKUP_RANGE" if overall else "NONE",
                "overall": overall,
                "diameter_ok": diameter_ok,
                "roundness_ok": roundness_ok,
                "ovality_evaluated": ovality["ovality_evaluated"],
                "measured_ovality_mm": ovality.get("measured_ovality_mm", measured_ovality_mm),
                "compliance_diameter_mm": measured_diameter_mm,
                "compliance_diameter_label": compliance_diameter_label,
                "min_diameter_mm": min_diameter_mm,
                "max_diameter_mm": max_diameter_mm,
                "diameter_fail_amount_mm": 0.0 if diameter_ok else min(
                    abs(measured_diameter_mm - min_diameter_mm),
                    abs(measured_diameter_mm - max_diameter_mm),
                ),
                "roundness_excess_mm": ovality["roundness_excess_mm"],
                "max_ovality_mm": ovality["max_ovality_mm"],
                "ovality_source": ovality["ovality_source"],
                "failure_reason": failure_reason,
                "selected_entry": selected_entry_summary,
                "selected_wall_schedule": selected_wall_schedule,
                "selected_wall_thickness_mm": selected_wall_thickness_mm,
                "evaluation_available": True,
                "lookup_range_only": True,
            }
        return {
            "standard_label": standard_display_name(standard),
            "standard_id": standard["id"],
            "sheet_name": standard["sheet"],
            "mode": standard["mode"],
            "nominal_mm": nominal_values[0][1] if nominal_values else None,
            "tolerance_table": pd.DataFrame(),
            "max_ovality_mm": float("nan"),
            "ovality_source": "not evaluated",
            "best_diameter_class": "NONE",
            "best_overall_class": "NONE",
            "overall": False,
            "diameter_ok": False,
            "roundness_ok": False,
            "ovality_evaluated": False,
            "measured_ovality_mm": measured_ovality_mm,
            "compliance_diameter_mm": measured_diameter_mm,
            "compliance_diameter_label": compliance_diameter_label,
            "min_diameter_mm": float("nan"),
            "max_diameter_mm": float("nan"),
            "failure_reason": (
                "This selected lookup row has no explicit min/max diameter range and no extracted diameter tolerance formula. "
                "Use it as reference data only, or enable an approved manual tolerance if the vendor/customer provides one."
            ),
            "selected_entry": selected_entry_summary,
            "selected_wall_schedule": selected_wall_schedule,
            "selected_wall_thickness_mm": selected_wall_thickness_mm,
            "evaluation_available": False,
        }

    table = pd.DataFrame(rows)
    passing_rows = [row for row in rows if row["diameter_ok"]]
    best = passing_rows[0] if passing_rows else min(
        rows,
        key=lambda row: min(
            abs(measured_diameter_mm - row["min_diameter_mm"]),
            abs(measured_diameter_mm - row["max_diameter_mm"]),
        ),
    )
    diameter_ok = bool(passing_rows)
    ovality = evaluate_ovality_tolerance(standard, best, best["nominal_mm"], measured_ovality_mm, selected_entry, ovality_lookup, selected_wall_thickness_mm, selected_wall_schedule)
    roundness_ok = ovality["roundness_ok"]
    overall = bool(diameter_ok and (roundness_ok is not False))
    if diameter_ok and best["class"] == "MANUAL":
        failure_reason = (
            f"PASS: {compliance_diameter_label} {measured_diameter_mm:.3f} mm fits the approved manual "
            f"±{best['abs_tol_mm']:.3f} mm tolerance for {best['nominal_mm']:.3f} mm nominal."
        )
    elif diameter_ok:
        failure_reason = (
            f"PASS: {compliance_diameter_label} {measured_diameter_mm:.3f} mm fits "
            f"{best['class']} for {best['nominal_mm']:.3f} mm nominal."
        )
    elif measured_diameter_mm < best["min_diameter_mm"]:
        failure_reason = (
            f"FAIL: {compliance_diameter_label} {measured_diameter_mm:.3f} mm is below "
            f"the closest allowed range by {best['min_diameter_mm'] - measured_diameter_mm:.4f} mm."
        )
    else:
        failure_reason = (
            f"FAIL: {compliance_diameter_label} {measured_diameter_mm:.3f} mm is above "
            f"the closest allowed range by {measured_diameter_mm - best['max_diameter_mm']:.4f} mm."
        )
    if diameter_ok and roundness_ok is False:
        failure_reason = (
                f"FAIL: measured diameter range {float(measured_ovality_mm):.4f} mm exceeds "
            f"the allowed limit {ovality['max_ovality_mm']:.4f} mm by {ovality['roundness_excess_mm']:.4f} mm."
        )

    return {
        "standard_label": standard_display_name(standard),
        "standard_id": standard["id"],
        "sheet_name": standard["sheet"],
        "mode": standard["mode"],
        "nominal_mm": best["nominal_mm"],
        "tolerance_table": table,
        "max_ovality_mm": ovality["max_ovality_mm"],
        "ovality_source": ovality["ovality_source"],
        "best_diameter_class": best["class"] if diameter_ok else "NONE",
        "best_overall_class": best["class"] if overall else "NONE",
        "overall": overall,
        "diameter_ok": diameter_ok,
        "roundness_ok": roundness_ok,
        "ovality_evaluated": ovality["ovality_evaluated"],
        "measured_ovality_mm": ovality.get("measured_ovality_mm", measured_ovality_mm),
        "compliance_diameter_mm": measured_diameter_mm,
        "compliance_diameter_label": compliance_diameter_label,
        "min_diameter_mm": best["min_diameter_mm"],
        "max_diameter_mm": best["max_diameter_mm"],
        "diameter_fail_amount_mm": 0.0 if diameter_ok else min(
            abs(measured_diameter_mm - best["min_diameter_mm"]),
            abs(measured_diameter_mm - best["max_diameter_mm"]),
        ),
        "roundness_excess_mm": ovality["roundness_excess_mm"],
        "failure_reason": failure_reason,
        "selected_entry": selected_entry_summary,
        "selected_wall_schedule": selected_wall_schedule,
        "selected_wall_thickness_mm": selected_wall_thickness_mm,
        "evaluation_available": True,
    }


def render_contract_standard_grid(standards: List[Dict[str, object]]) -> Dict[str, object]:
    if "selected_contract_standard_id" not in st.session_state:
        st.session_state["selected_contract_standard_id"] = str(standards[0]["id"])

    selected_id = str(st.session_state["selected_contract_standard_id"])
    st.session_state["selected_contract_standard_id"] = selected_id

    st.markdown("#### 🏷️ STEP 1 — Choose a Standard")
    st.caption("Select the applicable industry standard for the pipe or fitting.")

    for row_start in range(0, len(standards), 2):
        cols = st.columns(2, gap="medium")
        for col, standard in zip(cols, standards[row_start : row_start + 2]):
            standard_id = str(standard["id"])
            is_selected = standard_id == selected_id
            std_type = str(standard.get("type", "standard")).title()
            title = STANDARD_UI_TITLE_BY_ID.get(standard_id, str(standard.get("name", "Standard")))
            ot_codes = standard.get("ot_codes", [])
            if ot_codes:
                codes = [str(ot.get("code", "?")) for ot in ot_codes[:3]]
                range_summary = f"OT: {', '.join(codes)}"
                if len(ot_codes) > 3:
                    range_summary = f"{range_summary} +{len(ot_codes) - 3}"
            else:
                range_summary = "No OT range"
            classes = tolerance_classes(standard)
            class_summary = f"{len(classes)} D-classes" if classes else "Lookup table"

            border = "2px solid #1f6feb" if is_selected else "1px solid #dde5ef"
            bg = "#f0f6ff" if is_selected else "#ffffff"
            with col:
                with st.container(border=True):
                    img_col, txt_col = st.columns([0.35, 1.0], gap="small", vertical_alignment="center")
                    with img_col:
                        image_name = STANDARD_IMAGE_BY_ID.get(standard_id)
                        image_path = APP_DIR / "images" / image_name if image_name else None
                        if image_path is not None and image_path.exists():
                            st.image(str(image_path), width=72)
                        else:
                            st.markdown("📋")
                    with txt_col:
                        badge_color = "#0d7d33" if is_selected else "#52616f"
                        badge_bg = "#daf5e1" if is_selected else "#eef2f7"
                        st.markdown(
                            f"""<span style="background:{badge_bg};color:{badge_color};padding:2px 8px;
                            border-radius:8px;font-size:0.72rem;font-weight:700;">{std_type}</span>
                            <span style="margin-left:6px;font-weight:700;font-size:0.95rem;">{standard_id}</span>""",
                            unsafe_allow_html=True,
                        )
                        st.caption(title[:80] + ("…" if len(title) > 80 else ""))
                    info_a, info_b = st.columns(2, gap="small")
                    info_a.caption(f"📏 {range_summary}")
                    info_b.caption(f"📐 {class_summary}")

                    btn_label = "✓ Selected" if is_selected else "Select"
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(btn_label, key=f"std_card_{standard_id}", type=btn_type, use_container_width=True,
                                 help=f"Select {standard_id}: {title}"):
                        st.session_state["selected_contract_standard_id"] = standard_id
                        st.rerun()

    return next(standard for standard in standards if str(standard["id"]) == selected_id)


def render_contract_tolerance_inputs(
    measurement: PipeMeasurement,
    standards_contract: Dict[str, object],
    suggested_nominal_mm: Optional[float] = None,
    feature_type: str = "inner",
    compliance_diameter_mm: Optional[float] = None,
    compliance_diameter_label: str = "Fitted diameter",
    measured_ovality_mm: Optional[float] = None,
    ovality_lookup: Optional[Dict[str, object]] = None,
    diameter_lookup: Optional[Dict[str, object]] = None,
    measured_wall_thickness_mm: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    standards = standards_contract["standards"]
    dia_val = float(compliance_diameter_mm if compliance_diameter_mm is not None else measurement.diameter_mm)
    oval_val = float(measured_ovality_mm) if measured_ovality_mm is not None and np.isfinite(float(measured_ovality_mm)) else float("nan")
    feat_label = "Outer pipe diameter" if feature_type == "outer" else "Inner/opening rim"

    # ── Measurement info bar ──
    st.markdown("### 📋 Measurement Snapshot")
    meas_css = """
    <style>
    .meas-info-card {padding:14px 18px; border-radius:10px; border:1px solid #dde5ef;
         background:#ffffff; box-shadow:0 1px 3px rgba(15,23,42,0.04);}
    .meas-info-card .mi-label {font-size:0.76rem;color:#6b7c93;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.03em;}
    .meas-info-card .mi-value {font-size:1.2rem;font-weight:800;color:#111827;}
    .meas-info-card .mi-sub {font-size:0.78rem;color:#52616f;margin-top:2px;}
    </style>
    """
    st.markdown(meas_css, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3, gap="medium")
    with mc1:
        st.markdown(
            f"""
            <div class="meas-info-card">
                <div class="mi-label">🎯 Diameter Method</div>
                <div class="mi-value">{compliance_diameter_label}</div>
                <div class="mi-sub">Compliance Ø = {dia_val:.3f} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        oval_str = f"{oval_val:.4f} mm" if np.isfinite(oval_val) else "not available"
        st.markdown(
            f"""
            <div class="meas-info-card">
                <div class="mi-label">⭕ Ovality Input</div>
                <div class="mi-value">{oval_str}</div>
                <div class="mi-sub">Measured diameter range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"""
            <div class="meas-info-card">
                <div class="mi-label">📐 Target</div>
                <div class="mi-value">{feat_label}</div>
                <div class="mi-sub">Active measurement rim</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── STEP 1: Standard Grid ──
    selected_standard = render_contract_standard_grid(standards)
    entries = selected_standard.get("entries", [])
    classes = tolerance_classes(selected_standard)

    class_text = ", ".join(sorted(classes)) if classes else "lookup data only"
    selected_title = STANDARD_UI_TITLE_BY_ID.get(str(selected_standard["id"]), str(selected_standard["name"]))
    st.caption(f"✓ Selected: **{selected_standard['id']}** — {selected_title[:100]}")

    nominal_values: List[Tuple[str, float]] = []
    selected_entry = None
    selected_classes = list(classes)
    manual_tolerance_mm: Optional[float] = None
    selected_wall_schedule: Optional[str] = None
    selected_wall_thickness_mm: Optional[float] = None
    has_diameter_lookup = bool((diameter_lookup or {}).get("standards", {}).get(str(selected_standard.get("id")), {}).get("entries"))

    # ── STEP 2: Match the Measurement ──
    st.markdown("#### 📐 STEP 2 — Match the Measurement")
    st.caption("Select the nominal diameter and tolerance class that matches the pipe.")

    input_cols = st.columns([1.0, 1.2, 1.0])
    with input_cols[0]:
        st.markdown("**🔖 OT Range**")
        ot_codes = selected_standard.get("ot_codes", [])
        selected_ot = None
        if ot_codes:
            ot_labels = [
                f"{ot['code']} ({ot.get('system', 'system n/a')}, "
                f"{ot.get('range_mm', ot.get('range_inch', ['?', '?']))[0]}-"
                f"{ot.get('range_mm', ot.get('range_inch', ['?', '?']))[1]}"
                f"{' mm' if 'range_mm' in ot else ' inch label'})"
                for ot in ot_codes
            ]
            ot_index = st.selectbox("OT code / range", range(len(ot_labels)), format_func=lambda idx: ot_labels[idx], label_visibility="collapsed")
            selected_ot = ot_codes[ot_index]["code"]
        else:
            st.info("No OT range defined")

    with input_cols[1]:
        st.markdown("**📏 Nominal Diameter**")
        if entries:
            preferred_nominal = suggested_nominal_mm if suggested_nominal_mm is not None and suggested_nominal_mm > 0 else measurement.diameter_mm
            default_entry_index = 0
            closest_diff = float("inf")
            for idx, entry in enumerate(entries):
                candidates = nominal_candidates_from_entry(entry)
                if not candidates:
                    continue
                diff = min(abs(candidate_value - preferred_nominal) for _, candidate_value in candidates)
                if diff < closest_diff:
                    closest_diff = diff
                    default_entry_index = idx
            entry_index = st.selectbox(
                "Nominal / table entry",
                range(len(entries)),
                index=default_entry_index,
                format_func=lambda idx: entry_display_name(entries[idx]),
                label_visibility="collapsed",
            )
            selected_entry = entries[entry_index]
            nominal_values = nominal_candidates_from_entry(selected_entry)
            thickness_nominal = ((selected_entry.get("thickness") or {}).get("nominal") or {}) if isinstance(selected_entry, dict) else {}
            schedule_options = [
                schedule
                for schedule, payload in thickness_nominal.items()
                if isinstance(payload, dict) and payload.get("nominal_mm") not in {None, "N/A"}
            ]
            if schedule_options:
                selected_wall_schedule = st.selectbox(
                    "Wall schedule",
                    schedule_options,
                    help="Used only when the selected standard has an ovality rule that depends on wall thickness.",
                )
                selected_wall_thickness_mm = float(thickness_nominal[selected_wall_schedule]["nominal_mm"])
                st.caption(f"Wall: {selected_wall_thickness_mm:.3f} mm (schedule {selected_wall_schedule})")
            elif measured_wall_thickness_mm is not None and np.isfinite(float(measured_wall_thickness_mm)):
                selected_wall_thickness_mm = float(measured_wall_thickness_mm)
                st.caption(f"Wall: {selected_wall_thickness_mm:.3f} mm (measured)")

            if nominal_values:
                match_diff = closest_diff
                match_color = "#0d7d33" if match_diff < 2 else ("#d97706" if match_diff < 10 else "#dc2626")
                match_icon = "✓" if match_diff < 2 else "⚠"
                st.caption(f"{match_icon} Closest nominal match: Δ {match_diff:.1f} mm from measured {preferred_nominal:.1f} mm")
            else:
                st.info("No diameter nominal in this row.")
        else:
            preferred_nominal = float(max(suggested_nominal_mm if suggested_nominal_mm is not None else measurement.diameter_mm, 0.001))
            mm_range = standard_range_mm(selected_standard, selected_ot)
            if mm_range is not None:
                nominal = st.slider(
                    "Manual nominal diameter (mm)",
                    min_value=float(mm_range[0]),
                    max_value=float(mm_range[1]),
                    value=float(max(mm_range[0], min(preferred_nominal, mm_range[1]))),
                    step=0.1,
                    help="Limited by the selected OT range.",
                )
            else:
                nominal = st.number_input("Manual nominal diameter (mm)", min_value=0.001, value=preferred_nominal, step=0.1)
            nominal_values = [("manual_nominal_mm", float(nominal))]

    with input_cols[2]:
        st.markdown("**📊 Tolerance Classes**")
        if classes:
            class_options = sorted(classes)
            selected_classes = st.multiselect(
                "Tolerance class",
                class_options,
                default=class_options,
                label_visibility="collapsed",
                help="Select one or more D classes to test.",
            )
            if not selected_classes:
                st.warning("Select at least one class.")
        else:
            if has_diameter_lookup:
                st.info("Uses lookup min/max")
            else:
                st.info("No formula classes")

    # ── STEP 3: Evaluate ──
    st.markdown("#### 🚀 STEP 3 — Evaluate")
    run_both = st.button("🔍 Run Full Standards Check", type="primary", use_container_width=True,
                          help="Evaluates BOTH roundness/ovality and diameter tolerance against the selected standard.")
    if run_both:
        if not nominal_values:
            st.error("No nominal value is available for this selected standard row.")
            return None
        if classes and not selected_classes:
            st.error("Select at least one tolerance class.")
            return None

        # Run diameter check
        dia_report = build_contract_tolerance_report(
            measurement, selected_standard, nominal_values, selected_entry, selected_classes,
            manual_tolerance_mm, compliance_diameter_mm, compliance_diameter_label,
            measured_ovality_mm, ovality_lookup, diameter_lookup,
            selected_wall_thickness_mm, selected_wall_schedule,
        )
        dia_report["evaluation_kind"] = "diameter"
        dia_report["evaluation_label"] = "Diameter tolerance"
        dia_report = focus_contract_report_on_selected_check(dia_report)

        # Run roundness check
        oval_report = build_contract_tolerance_report(
            measurement, selected_standard, nominal_values, selected_entry, selected_classes,
            manual_tolerance_mm, compliance_diameter_mm, compliance_diameter_label,
            measured_ovality_mm, ovality_lookup, diameter_lookup,
            selected_wall_thickness_mm, selected_wall_schedule,
        )
        oval_report["evaluation_kind"] = "roundness"
        oval_report["evaluation_label"] = "Roundness / ovality"
        oval_report = focus_contract_report_on_selected_check(oval_report)

        # Combined report
        dia_ok = dia_report.get("diameter_ok", False)
        oval_ok = oval_report.get("roundness_ok")
        oval_evaluated = oval_report.get("ovality_evaluated", False)

        if oval_evaluated and oval_ok is not None:
            combined_ok = bool(dia_ok and oval_ok)
            if not dia_ok and not oval_ok:
                combined_reason = "FAIL: Both diameter tolerance and roundness/ovality failed."
            elif not dia_ok:
                combined_reason = f"FAIL: Diameter tolerance failed. {standard_check_message(dia_report)}"
            elif not oval_ok:
                combined_reason = f"FAIL: Roundness/ovality failed. {standard_check_message(oval_report)}"
            else:
                combined_reason = "PASS: Both diameter tolerance and roundness/ovality are within limits."
        else:
            combined_ok = bool(dia_ok)
            if not dia_ok:
                combined_reason = f"FAIL: Diameter tolerance failed. Roundness not evaluated for this standard. {standard_check_message(dia_report)}"
            else:
                combined_reason = "PASS: Diameter tolerance is within limits. Roundness not evaluated for this standard."

        combined = {
            "diameter_report": dia_report,
            "oval_report": oval_report,
            "combined_ok": combined_ok,
            "combined_reason": combined_reason,
            "ovality_evaluated": oval_evaluated,
            "standard_label": selected_standard.get("name", ""),
            "standard_id": selected_standard.get("id", ""),
            "nominal_mm": dia_report.get("nominal_mm"),
            "evaluation_available": True,
            "diameter_ok": dia_ok,
            "roundness_ok": oval_ok,
            "overall": combined_ok,
            "failure_reason": combined_reason,
            "compliance_diameter_mm": dia_report.get("compliance_diameter_mm"),
            "compliance_diameter_label": dia_report.get("compliance_diameter_label"),
            "min_diameter_mm": dia_report.get("min_diameter_mm"),
            "max_diameter_mm": dia_report.get("max_diameter_mm"),
            "max_ovality_mm": oval_report.get("max_ovality_mm"),
            "ovality_source": oval_report.get("ovality_source"),
            "diameter_fail_amount_mm": dia_report.get("diameter_fail_amount_mm"),
            "roundness_excess_mm": oval_report.get("roundness_excess_mm"),
            "measured_ovality_mm": oval_report.get("measured_ovality_mm"),
            "selected_entry": selected_entry,
            "selected_wall_schedule": selected_wall_schedule,
            "selected_wall_thickness_mm": selected_wall_thickness_mm,
            "tolerance_table": dia_report.get("tolerance_table"),
            "best_diameter_class": dia_report.get("best_diameter_class"),
            "best_overall_class": dia_report.get("best_overall_class") if combined_ok else "NONE",
        }
        return combined
    return None


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def crop_bgr_to_bbox(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_fraction: float = 0.12,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_fraction)
    pad_y = int((y2 - y1) * padding_fraction)
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)
    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]


def measurement_zoom_bbox(image_bgr: np.ndarray, measurement: PipeMeasurement, radius_multiplier: float = 1.45) -> Tuple[int, int, int, int]:
    height, width = image_bgr.shape[:2]
    half_size = max(40, int(round(measurement.radius_px * radius_multiplier)))
    cx = int(round(measurement.center_x_px))
    cy = int(round(measurement.center_y_px))
    return (
        max(0, cx - half_size),
        max(0, cy - half_size),
        min(width, cx + half_size),
        min(height, cy + half_size),
    )


def draw_outer_circle_overlay(
    image_bgr: np.ndarray,
    measurement: PipeMeasurement,
    processed: Dict[str, np.ndarray],
    tolerance_report: Optional[Dict[str, object]] = None,
    unit: str = "px",
    wall_detection: Optional[PipeWallDetection] = None,
) -> np.ndarray:
    overlay = image_bgr.copy()
    center = np.array([measurement.center_x_px, measurement.center_y_px], dtype=np.float64)
    radius = float(measurement.radius_px)
    value_scale = measurement.mm_per_pixel if unit == "mm" else 1.0

    x1, y1, x2, y2 = processed.get("roi_bbox", (0, 0, image_bgr.shape[1], image_bgr.shape[0]))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

    points = measurement.contour_points.astype(np.int32)
    if len(points) > 1:
        cv2.polylines(overlay, [points.reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(overlay, tuple(np.round(center).astype(int)), int(round(radius)), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(overlay, tuple(np.round(center).astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)

    if wall_detection is not None:
        inner_center = (int(round(wall_detection.inner_center_x_px)), int(round(wall_detection.inner_center_y_px)))
        outer_center = (int(round(wall_detection.outer_center_x_px)), int(round(wall_detection.outer_center_y_px)))
        inner_radius = int(round(wall_detection.inner_radius_px))
        outer_radius = int(round(wall_detection.outer_radius_px))
        cv2.circle(overlay, inner_center, inner_radius, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(overlay, outer_center, outer_radius, (0, 255, 0), 3, cv2.LINE_AA)
        if len(wall_detection.outer_points) > 2:
            outer_points = wall_detection.outer_points.astype(np.int32)
            cv2.polylines(overlay, [outer_points.reshape(-1, 1, 2)], True, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "inner", (inner_center[0] + 10, max(24, inner_center[1] - inner_radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, "inner", (inner_center[0] + 10, max(24, inner_center[1] - inner_radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 120, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, "outer fit", (outer_center[0] + 10, max(24, outer_center[1] - outer_radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, "outer fit", (outer_center[0] + 10, max(24, outer_center[1] - outer_radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 130, 0), 1, cv2.LINE_AA)

    step = max(1, len(measurement.contour_points) // 260)
    for point in measurement.contour_points[::step]:
        vector = point.astype(np.float64) - center
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-6:
            continue
        signed_error = distance - radius
        if signed_error < 0:
            continue
        ideal_point = center + vector / distance * radius
        cv2.line(
            overlay,
            tuple(np.round(ideal_point).astype(int)),
            tuple(np.round(point).astype(int)),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    status = None
    if tolerance_report is not None:
        status = "PASS" if tolerance_report["overall"] else "FAIL"
    lines = [
        f"Fitted dia: {measurement.diameter_px * value_scale:.3f} {unit}",
        f"Real edge PTV: {measurement.roundness_px * value_scale:.4f} {unit}",
        f"Rim points: {len(measurement.contour_points)}",
    ]
    if tolerance_report is not None and tolerance_report.get("evaluation_available", True):
        lines.extend(
            [
                f"Nominal: {tolerance_report['nominal_mm']:.3f} mm",
                f"Allowed dia: {tolerance_report['min_diameter_mm']:.3f}-{tolerance_report['max_diameter_mm']:.3f} mm",
            ]
        )

    y = 34
    for line in lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30
    if status:
        color = (0, 180, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(overlay, status, (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(overlay, status, (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return overlay


def draw_deviation_heat_overlay(
    image_bgr: np.ndarray,
    measurement: PipeMeasurement,
    processed: Dict[str, np.ndarray],
    second_measurement: Optional[PipeMeasurement] = None,
    second_label: str = "",
) -> np.ndarray:
    overlay = image_bgr.copy()

    # ── Draw legend background FIRST ──
    legend_h = 120 if second_measurement is not None else 60
    cv2.rectangle(overlay, (12, 4), (280, 28 + legend_h), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (12, 4), (280, 28 + legend_h), (50, 50, 50), 1, cv2.LINE_AA)

    # Outer rim (or primary) legend — blue-to-red
    cv2.circle(overlay, (28, 26), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(overlay, "Outward dev. (outer)" if second_measurement else "Outward deviation", (44, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(overlay, (28, 55), 6, (255, 80, 0), -1, cv2.LINE_AA)
    cv2.putText(overlay, "Inward dev. (outer)" if second_measurement else "Inward deviation", (44, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

    # Inner rim legend
    if second_measurement is not None:
        cv2.circle(overlay, (28, 86), 6, (0, 220, 80), -1, cv2.LINE_AA)
        cv2.putText(overlay, "Outward dev. (inner)", (44, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.circle(overlay, (28, 112), 6, (220, 60, 220), -1, cv2.LINE_AA)
        cv2.putText(overlay, "Inward dev. (inner)", (44, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

    # ── Draw deviation dots ──
    def draw_rim_deviation(meas: PipeMeasurement, color_outward: Tuple[int, int, int], color_inward: Tuple[int, int, int]):
        center = np.array([meas.center_x_px, meas.center_y_px], dtype=np.float64)
        radius = float(meas.radius_px)
        cv2.circle(overlay, tuple(np.round(center).astype(int)), int(round(radius)), (180, 180, 180), 1, cv2.LINE_AA)

        signed = meas.signed_errors_px.astype(np.float64)
        points = meas.contour_points.astype(np.float64)
        if len(signed) == 0 or len(points) == 0:
            return

        distances = np.linalg.norm(points - center, axis=1)
        radial_error = distances - radius
        display_band_px = max(3.0, radius * 0.035)
        display_mask = np.abs(radial_error) <= display_band_px
        if np.count_nonzero(display_mask) < 20:
            display_mask = np.ones(len(points), dtype=bool)

        display_points = points[display_mask].astype(np.int32)
        display_errors = radial_error[display_mask]
        max_abs_error = max(float(np.max(np.abs(display_errors))), 1e-9)
        point_radius = max(2, int(round(radius / 180.0)))

        for point, error in zip(display_points, display_errors):
            strength = min(1.0, abs(float(error)) / max_abs_error)
            if error >= 0:
                color = (int(40 * (1.0 - strength)), int(80 * (1.0 - strength)), int(255 * strength + 80 * (1.0 - strength)))
            else:
                color = (int(255 * strength + 80 * (1.0 - strength)), int(120 * (1.0 - strength)), int(30 * (1.0 - strength)))
            cv2.circle(overlay, tuple(point), point_radius, tuple(min(255, max(0, int(c))) for c in color), -1, cv2.LINE_AA)

    draw_rim_deviation(measurement, (0, 0, 255), (255, 80, 0))

    # Inner rim dots — magenta-to-green (smaller)
    if second_measurement is not None:
        inner_center = np.array([second_measurement.center_x_px, second_measurement.center_y_px], dtype=np.float64)
        inner_radius = float(second_measurement.radius_px)
        cv2.circle(overlay, tuple(np.round(inner_center).astype(int)), int(round(inner_radius)), (200, 180, 200), 1, cv2.LINE_AA)

        signed = second_measurement.signed_errors_px.astype(np.float64)
        points = second_measurement.contour_points.astype(np.float64)
        if len(signed) > 0 and len(points) > 0:
            distances = np.linalg.norm(points - inner_center, axis=1)
            radial_error = distances - inner_radius
            display_band_px = max(3.0, inner_radius * 0.035)
            display_mask = np.abs(radial_error) <= display_band_px
            if np.count_nonzero(display_mask) < 20:
                display_mask = np.ones(len(points), dtype=bool)
            display_points = points[display_mask].astype(np.int32)
            display_errors = radial_error[display_mask]
            max_abs_error = max(float(np.max(np.abs(display_errors))), 1e-9)
            point_radius = max(1, int(round(inner_radius / 250.0)))

            for point, error in zip(display_points, display_errors):
                strength = min(1.0, abs(float(error)) / max_abs_error)
                if error >= 0:
                    color = (int(60 * (1.0 - strength)), int(200 * strength + 80 * (1.0 - strength)), int(80 * (1.0 - strength)))
                else:
                    color = (int(200 * strength + 80 * (1.0 - strength)), int(60 * (1.0 - strength)), int(200 * strength + 80 * (1.0 - strength)))
                cv2.circle(overlay, tuple(point), point_radius, tuple(min(255, max(0, int(c))) for c in color), -1, cv2.LINE_AA)

    x1, y1, x2, y2 = processed.get("roi_bbox", (0, 0, image_bgr.shape[1], image_bgr.shape[0]))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)
    return overlay


def draw_pipe_wall_overlay(
    image_bgr: np.ndarray,
    measurement: PipeMeasurement,
    wall_detection: Optional[PipeWallDetection],
    processed: Dict[str, np.ndarray],
) -> np.ndarray:
    overlay = image_bgr.copy()
    x1, y1, x2, y2 = processed.get("roi_bbox", (0, 0, image_bgr.shape[1], image_bgr.shape[0]))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

    if wall_detection is not None:
        inner_center = (int(round(wall_detection.inner_center_x_px)), int(round(wall_detection.inner_center_y_px)))
        inner_radius = int(round(wall_detection.inner_radius_px))
        cv2.circle(overlay, inner_center, inner_radius, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(overlay, inner_center, 5, (0, 0, 255), -1, cv2.LINE_AA)
        if len(wall_detection.inner_points) > 2:
            inner_points = wall_detection.inner_points.astype(np.int32)
            cv2.polylines(overlay, [inner_points.reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
            fit_center = np.array([wall_detection.inner_center_x_px, wall_detection.inner_center_y_px], dtype=np.float64)
            fit_radius = float(wall_detection.inner_radius_px)
            step = max(1, len(wall_detection.inner_points) // 160)
            for point in wall_detection.inner_points[::step]:
                vector = point.astype(np.float64) - fit_center
                distance = float(np.linalg.norm(vector))
                if distance <= 1e-6:
                    continue
                ideal_point = fit_center + vector / distance * fit_radius
                cv2.line(
                    overlay,
                    tuple(np.round(ideal_point).astype(int)),
                    tuple(np.round(point).astype(int)),
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
        inner_left = (inner_center[0] - inner_radius, inner_center[1])
        inner_right = (inner_center[0] + inner_radius, inner_center[1])
        cv2.line(overlay, inner_left, inner_right, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(overlay, inner_left, 4, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, inner_right, 4, (255, 255, 0), -1, cv2.LINE_AA)
        inner_label_pos = (max(10, inner_left[0]), max(24, inner_center[1] - 14))
        cv2.putText(overlay, f"Inner dia: {wall_detection.inner_radius_px * 2.0:.1f} px", inner_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Inner dia: {wall_detection.inner_radius_px * 2.0:.1f} px", inner_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 120, 0), 1, cv2.LINE_AA)

        outer_center = (int(round(wall_detection.outer_center_x_px)), int(round(wall_detection.outer_center_y_px)))
        outer_radius = int(round(wall_detection.outer_radius_px))
        cv2.circle(overlay, outer_center, outer_radius, (0, 255, 0), 3, cv2.LINE_AA)
        if len(wall_detection.outer_points) > 2:
            points = wall_detection.outer_points.astype(np.int32)
            cv2.polylines(overlay, [points.reshape(-1, 1, 2)], True, (0, 165, 255), 2, cv2.LINE_AA)
            fit_center = np.array([wall_detection.outer_center_x_px, wall_detection.outer_center_y_px], dtype=np.float64)
            fit_radius = float(wall_detection.outer_radius_px)
            step = max(1, len(wall_detection.outer_points) // 160)
            for point in wall_detection.outer_points[::step]:
                vector = point.astype(np.float64) - fit_center
                distance = float(np.linalg.norm(vector))
                if distance <= 1e-6:
                    continue
                ideal_point = fit_center + vector / distance * fit_radius
                cv2.line(
                    overlay,
                    tuple(np.round(ideal_point).astype(int)),
                    tuple(np.round(point).astype(int)),
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
        outer_top = (outer_center[0], outer_center[1] - outer_radius)
        outer_bottom = (outer_center[0], outer_center[1] + outer_radius)
        cv2.line(overlay, outer_top, outer_bottom, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(overlay, outer_top, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, outer_bottom, 4, (0, 255, 0), -1, cv2.LINE_AA)
        outer_label_pos = (min(max(10, outer_center[0] + 10), image_bgr.shape[1] - 260), max(24, outer_top[1] + 24))
        cv2.putText(overlay, f"Outer fit dia: {wall_detection.outer_radius_px * 2.0:.1f} px", outer_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Outer fit dia: {wall_detection.outer_radius_px * 2.0:.1f} px", outer_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 130, 0), 1, cv2.LINE_AA)
        text_lines = [
            "Outer/inner rim view",
            "Magenta/Cyan: sampled/fitted inner",
            "Orange/Green: sampled/fitted outer",
            f"Inner fitted diameter: {wall_detection.inner_radius_px * 2.0:.1f} px",
            f"Outer fitted diameter: {wall_detection.outer_radius_px * 2.0:.1f} px",
            f"Wall thickness: {wall_detection.wall_thickness_px:.1f} px",
            f"Range: {wall_detection.wall_thickness_min_px:.1f}-{wall_detection.wall_thickness_max_px:.1f} px",
        ]
    else:
        detected_center = (int(round(measurement.center_x_px)), int(round(measurement.center_y_px)))
        detected_radius = int(round(measurement.radius_px))
        cv2.circle(overlay, detected_center, detected_radius, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(overlay, detected_center, 5, (0, 0, 255), -1, cv2.LINE_AA)
        detected_left = (detected_center[0] - detected_radius, detected_center[1])
        detected_right = (detected_center[0] + detected_radius, detected_center[1])
        cv2.line(overlay, detected_left, detected_right, (255, 255, 0), 2, cv2.LINE_AA)
        label_pos = (max(10, detected_left[0]), max(24, detected_center[1] - 14))
        cv2.putText(overlay, f"Detected dia: {measurement.radius_px * 2.0:.1f} px", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Detected dia: {measurement.radius_px * 2.0:.1f} px", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 120, 0), 1, cv2.LINE_AA)
        text_lines = [
            "Outer/inner rim view",
            "Second rim not found clearly",
            "Review ROI and edge map",
        ]

    y = 34
    for line in text_lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30
    return overlay


def build_polar_figure(measurement: PipeMeasurement, unit: str = "mm") -> go.Figure:
    scale = measurement.mm_per_pixel if unit == "mm" else 1.0
    points = measurement.contour_points
    signed = measurement.signed_errors_px * scale
    angles = np.arctan2(points[:, 1] - measurement.center_y_px, points[:, 0] - measurement.center_x_px)
    order = np.argsort(angles)
    angles_closed = np.append(angles[order], angles[order][0] + 2 * np.pi)
    signed_closed = np.append(signed[order], signed[order][0])
    positive = np.where(signed_closed >= 0, signed_closed, np.nan)
    negative = np.where(signed_closed <= 0, signed_closed, np.nan)
    zero = np.zeros_like(angles_closed)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=zero,
            theta=angles_closed * 180.0 / np.pi,
            mode="lines",
            name="Ideal circle",
            line=dict(color="black", dash="dash", width=2),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=positive,
            theta=angles_closed * 180.0 / np.pi,
            mode="lines",
            fill="toself",
            name="Outward deviation",
            line=dict(color="#00bcd4", width=2),
            fillcolor="rgba(0, 188, 212, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=negative,
            theta=angles_closed * 180.0 / np.pi,
            mode="lines",
            fill="toself",
            name="Inward deviation",
            line=dict(color="#f97316", width=2),
            fillcolor="rgba(249, 115, 22, 0.22)",
        )
    )
    max_abs = max(1e-6, float(np.nanmax(np.abs(signed_closed))))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=52, b=20),
        polar=dict(radialaxis=dict(range=[-max_abs * 1.15, max_abs * 1.15], title=f"Deviation ({unit})")),
        title=f"Radial deviation ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def build_deviation_profile_figure(measurement: PipeMeasurement, unit: str = "px") -> go.Figure:
    scale = measurement.mm_per_pixel if unit == "mm" else 1.0
    points = measurement.contour_points
    signed = measurement.signed_errors_px * scale
    angles = np.degrees(np.arctan2(points[:, 1] - measurement.center_y_px, points[:, 0] - measurement.center_x_px))
    angles = (angles + 360.0) % 360.0
    order = np.argsort(angles)
    max_abs = max(1e-6, float(np.max(np.abs(signed))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=angles[order],
            y=signed[order],
            mode="lines",
            name="Signed radial deviation",
            line=dict(color="#1f6feb", width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 111, 235, 0.18)",
        )
    )
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=48, b=40),
        title=f"Deviation by angle ({unit})",
        xaxis_title="Angle around rim (degrees)",
        yaxis_title=f"Signed deviation ({unit})",
        yaxis=dict(range=[-max_abs * 1.15, max_abs * 1.15]),
        showlegend=False,
    )
    return fig


def build_deviation_histogram(measurement: PipeMeasurement, unit: str = "px") -> go.Figure:
    scale = measurement.mm_per_pixel if unit == "mm" else 1.0
    signed = measurement.signed_errors_px * scale
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=signed,
            nbinsx=50,
            marker=dict(color="#2f80ed"),
            name="Signed radial error",
        )
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=48, b=30),
        title=f"Signed radial deviation distribution ({unit})",
        xaxis_title=f"Deviation ({unit})",
        yaxis_title="Point count",
        showlegend=False,
    )
    return fig


def build_result_row(
    test_measurement: PipeMeasurement,
    tolerance_report: Optional[Dict[str, object]],
    standard_label: Optional[str],
    scale_source: str,
    measurement_target: Optional[str] = None,
) -> pd.DataFrame:
    diameter_stats = measurement_diameter_stats(test_measurement)
    roundness_methods = compute_roundness_method_stats(test_measurement)

    diameter_method = st.session_state.get("diameter_ref_method", "lsc")
    ovality_method = st.session_state.get("ovality_ref_method", "ostb")

    row = {
        # ── File & Scale ──
        "test_file": test_measurement.filename,
        "measurement_target": measurement_target or "",
        "scale_source": scale_source,
        "mm_per_pixel": test_measurement.mm_per_pixel,
        # ── Methods Selected ──
        "diameter_method": "LSC (robust)" if diameter_method == "lsc" else "MCC/MIC (strict)",
        "ovality_method": "OSTB measured range" if ovality_method == "ostb" else "MZC roundness (ISO)",
    }

    if scale_source != "pixel_only":
        mcc_dia = roundness_methods["mcc_radius_px"] * 2.0 * test_measurement.mm_per_pixel
        mic_dia = roundness_methods["mic_diameter_mm"]
        lsc_dia = test_measurement.diameter_mm
        lsc_round = roundness_methods["lsc_roundness_mm"]
        mzc_round = roundness_methods["mzc_roundness_mm"]
        edge_min = diameter_stats["real_edge_min_diameter_px"] * test_measurement.mm_per_pixel
        edge_max = diameter_stats["real_edge_max_diameter_px"] * test_measurement.mm_per_pixel
        edge_range = diameter_stats["real_edge_diameter_range_px"] * test_measurement.mm_per_pixel

        row.update({
            # ── Diameters (mm) ──
            "lsc_fitted_diameter_mm": round(lsc_dia, 4),
            "mcc_diameter_mm": round(mcc_dia, 4),
            "mic_diameter_mm": round(mic_dia, 4),
            "compliance_diameter_used_mm": round(lsc_dia if diameter_method == "lsc" else (mcc_dia if measurement_target == "Outer pipe diameter" else mic_dia), 4),
            # ── Rim Diameter Range ──
            "real_edge_min_diameter_mm": round(edge_min, 4),
            "real_edge_max_diameter_mm": round(edge_max, 4),
            "real_edge_diameter_range_mm": round(edge_range, 4),
            # ── Roundness ──
            "lsc_roundness_mm": round(lsc_round, 4),
            "mzc_roundness_mm": round(mzc_round, 4),
            "ovality_input_used_mm": round(edge_range if ovality_method == "ostb" else mzc_round, 4),
        })
    else:
        row.update({
            "test_fitted_diameter_px": round(diameter_stats["fitted_diameter_px"], 2),
            "real_edge_min_diameter_px": round(diameter_stats["real_edge_min_diameter_px"], 2),
            "real_edge_max_diameter_px": round(diameter_stats["real_edge_max_diameter_px"], 2),
            "real_edge_diameter_range_px": round(diameter_stats["real_edge_diameter_range_px"], 2),
            "roundness_px": round(test_measurement.roundness_px, 3),
        })

    # ── Standards Check Results ──
    if tolerance_report is not None:
        row.update({
            "standard": tolerance_report.get("standard_label", standard_label or ""),
            "standard_id": tolerance_report.get("standard_id", ""),
            "nominal_mm": round(float(tolerance_report.get("nominal_mm", float("nan"))), 4),
            "min_allowed_diameter_mm": round(float(tolerance_report.get("min_diameter_mm", float("nan"))), 4),
            "max_allowed_diameter_mm": round(float(tolerance_report.get("max_diameter_mm", float("nan"))), 4),
            "checked_diameter_mm": round(float(tolerance_report.get("compliance_diameter_mm", float("nan"))), 4),
            "diameter_OK": tolerance_report.get("diameter_ok", False),
            "ovality_limit_mm": round(float(tolerance_report.get("max_ovality_mm", float("nan"))), 4),
            "measured_ovality_mm": round(float(tolerance_report.get("measured_ovality_mm", float("nan"))), 4),
            "ovality_evaluated": tolerance_report.get("ovality_evaluated", False),
            "roundness_OK": tolerance_report.get("roundness_ok"),
            "overall_PASS": tolerance_report.get("overall", False),
            "failure_reason": tolerance_report.get("failure_reason", ""),
            "best_class": tolerance_report.get("best_overall_class", ""),
            "wall_schedule": tolerance_report.get("selected_wall_schedule", ""),
            "wall_thickness_mm": round(float(tolerance_report.get("selected_wall_thickness_mm", float("nan"))), 4) if tolerance_report.get("selected_wall_thickness_mm") is not None else "",
        })
    return pd.DataFrame([row])


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #f7f9fc;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde5ef;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        h2, h3 {
            letter-spacing: 0;
        }
        .visual-note {
            color: #52616f;
            font-size: 0.95rem;
            margin-top: -0.25rem;
            margin-bottom: 0.75rem;
        }
        .brand-caption {
            color: #52616f;
            font-size: 0.98rem;
            margin-top: -0.35rem;
        }
        .page-footer {
            border-top: 1px solid #dde5ef;
            margin-top: 2.25rem;
            padding-top: 1.1rem;
            color: #52616f;
            font-size: 0.9rem;
        }
        .sidebar-footer {
            border-top: 1px solid #dde5ef;
            margin-top: 1.75rem;
            padding-top: 1rem;
            color: #52616f;
            font-size: 0.82rem;
        }
        .circle-summary {
            border: 1px solid #dde5ef;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.85rem 0 0.35rem;
            background: #ffffff;
        }
        .circle-summary.good {
            border-left: 6px solid #16a34a;
            background: #f3fbf6;
        }
        .circle-summary.warn {
            border-left: 6px solid #d97706;
            background: #fff8eb;
        }
        .circle-summary.bad {
            border-left: 6px solid #dc2626;
            background: #fff3f3;
        }
        .summary-symbol {
            font-size: 2.2rem;
            line-height: 1;
            margin-right: 0.85rem;
        }
        .summary-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #2f3340;
            margin-bottom: 0.15rem;
        }
        .summary-message {
            color: #4b5563;
            font-size: 0.96rem;
        }
        .summary-row {
            display: flex;
            align-items: center;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.9rem;
        }
        .summary-label {
            color: #52616f;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .summary-value {
            color: #2f3340;
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }
        .meter {
            height: 8px;
            border-radius: 8px;
            background: #dfe7f1;
            overflow: hidden;
        }
        .meter-fill {
            height: 100%;
            border-radius: 8px;
        }
        @media (max-width: 700px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    title_col, logo_col = st.columns([4.5, 1.1], vertical_alignment="center")
    with title_col:
        st.title("Industrial Pipe Roundness Inspector")
    with logo_col:
        if OSTP_LOGO_PATH.exists():
            st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
            st.image(str(OSTP_LOGO_PATH), width=170)


def render_system_status(
    calibration: Optional[CameraCalibration],
    standards_contract: Optional[Dict[str, object]],
    saved_scale: Optional[Dict[str, object]],
) -> None:
    st.sidebar.subheader("System Status")
    if saved_scale is None:
        st.sidebar.info("No saved millimeter scale.")
    else:
        st.sidebar.success(f"Saved scale: {float(saved_scale['mm_per_pixel']):.6f} mm/px.")


def render_sidebar_brand() -> None:
    if NOVIA_LOGO_PATH.exists():
        st.sidebar.image(str(NOVIA_LOGO_PATH), width="stretch")


def render_sidebar_footer() -> None:
    st.sidebar.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    if NOVIA_LOGO_PATH.exists():
        st.sidebar.image(str(NOVIA_LOGO_PATH), width=130)
    st.sidebar.caption("Applied computer vision inspection support")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


def render_page_footer() -> None:
    st.markdown('<div class="page-footer">', unsafe_allow_html=True)
    footer_text, footer_logo = st.columns([4.5, 1.0], vertical_alignment="center")
    with footer_text:
        st.caption("Pipe Roundness Inspector | Vision-based measurement review")
    with footer_logo:
        if OSTP_LOGO_PATH.exists():
            st.image(str(OSTP_LOGO_PATH), width=92)
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_row(measurement: PipeMeasurement) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Fitted diameter",
        f"{measurement.diameter_mm:.3f} mm",
        help="Stable reference diameter from the fitted circle after applying the active millimeter scale. This is the diameter used for nominal tolerance comparison.",
    )
    c2.metric(
        "Real edge PTV",
        f"{measurement.roundness_mm:.4f} mm",
        help="Peak-to-valley radial variation of the sampled real edge against the fitted reference circle. This is where outer/inner rim defects show up.",
    )
    c3.metric(
        "Max abs deviation",
        f"{measurement.max_abs_deviation_mm:.4f} mm",
        help="Largest single radial error from the fitted ideal circle, regardless of direction. Lower is better; ideal is 0.0000 mm.",
    )
    c4.metric(
        "Scale",
        f"{measurement.mm_per_pixel:.6f} mm/px",
        help="Conversion factor used to turn pixel measurements into millimeters. It must come from the same camera distance, zoom, resolution, and pipe plane to be reliable.",
    )


def render_pixel_metric_row(measurement: PipeMeasurement) -> None:
    relative_roundness = measurement.roundness_px / measurement.diameter_px * 100.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Fitted diameter",
        f"{measurement.diameter_px:.2f} px",
        help="Stable reference diameter from the fitted circle in pixels. Raw sampled edge min/max diameter is included in the export.",
    )
    c2.metric(
        "Real edge PTV",
        f"{measurement.roundness_px:.3f} px",
        help="Peak-to-valley radial variation in pixels from the sampled real edge against the fitted reference circle. Ideal is 0 px; lower is better.",
    )
    c3.metric(
        "Max abs deviation",
        f"{measurement.max_abs_deviation_px:.3f} px",
        help="Largest single distance from the detected rim to the fitted ideal circle. Ideal is 0 px; high values can indicate ovality, edge noise, glare, or detection errors.",
    )
    c4.metric(
        "Relative roundness",
        f"{relative_roundness:.3f}%",
        help="Roundness PTV divided by measured diameter, expressed as a percent. It normalizes roundness for pipe size. Ideal is 0%; smaller is better. For pass/fail, use the applicable standard or ovality limit.",
    )


@st.cache_data
def load_gif_data_uri(gif_path: str) -> str:
    path = Path(gif_path)
    return f"data:image/gif;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def selected_standard_check_passed(tolerance_report: Dict[str, object]) -> Optional[bool]:
    kind = tolerance_report.get("evaluation_kind", "combined")
    if kind == "diameter":
        return bool(tolerance_report.get("diameter_ok"))
    if kind == "roundness":
        if not tolerance_report.get("ovality_evaluated"):
            return None
        return bool(tolerance_report.get("roundness_ok"))
    return bool(tolerance_report.get("overall"))


def focus_contract_report_on_selected_check(report: Dict[str, object]) -> Dict[str, object]:
    kind = report.get("evaluation_kind", "combined")
    if not report.get("evaluation_available", True):
        return report
    table = report.get("tolerance_table")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return report

    if kind == "roundness" and "ovality_evaluated" in table.columns:
        evaluated = table[table["ovality_evaluated"].fillna(False).astype(bool)]
        if evaluated.empty:
            report["roundness_ok"] = None
            return report
        passing = evaluated[evaluated["roundness_ok"].fillna(False).astype(bool)] if "roundness_ok" in evaluated.columns else pd.DataFrame()
        selected = passing.iloc[0] if not passing.empty else evaluated.iloc[0]
        measured = float(report.get("measured_ovality_mm", float("nan")))
        limit = float(selected.get("max_ovality_mm", float("nan")))
        report["nominal_mm"] = float(selected.get("nominal_mm", report.get("nominal_mm", float("nan"))))
        if "min_diameter_mm" in selected:
            report["min_diameter_mm"] = float(selected["min_diameter_mm"])
        if "max_diameter_mm" in selected:
            report["max_diameter_mm"] = float(selected["max_diameter_mm"])
        report["max_ovality_mm"] = limit
        if "ovality_source" in selected:
            report["ovality_source"] = str(selected["ovality_source"])
        report["roundness_ok"] = bool(measured <= limit) if np.isfinite(measured) and np.isfinite(limit) else None
        report["roundness_excess_mm"] = max(0.0, measured - limit) if np.isfinite(measured) and np.isfinite(limit) else float("nan")
        report["best_overall_class"] = str(selected.get("class", report.get("best_overall_class", "NONE")))
    return report


def render_standard_check_overlay(tolerance_report: Dict[str, object]) -> None:
    if not tolerance_report.get("evaluation_available", True):
        return
    if tolerance_report.get("lookup_range_only", False) and tolerance_report.get("evaluation_kind") != "diameter":
        return

    passed_value = selected_standard_check_passed(tolerance_report)
    if passed_value is None:
        return
    passed = bool(passed_value)
    gif_path = SUCCESS_GIF_PATH if passed else FAIL_GIF_PATH
    if not gif_path.exists():
        return

    status = "PASS" if passed else "FAIL"
    result_class = "pass" if passed else "fail"
    reason = html.escape(standard_check_message(tolerance_report))
    gif_data_uri = load_gif_data_uri(str(gif_path))
    accent = "#11823b" if passed else "#b42318"
    background = "rgba(6, 31, 18, 0.52)" if passed else "rgba(44, 12, 10, 0.52)"

    st.markdown(
        f"""
        <style>
            @keyframes ostb-result-overlay-hide {{
                0%, 82% {{
                    opacity: 1;
                    visibility: visible;
                }}
                100% {{
                    opacity: 0;
                    visibility: hidden;
                }}
            }}
            .ostb-result-overlay {{
                position: fixed;
                inset: 0;
                z-index: 999999;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 24px;
                background: {background};
                animation: ostb-result-overlay-hide 4.2s ease forwards;
            }}
            .ostb-result-modal {{
                width: min(720px, calc(100vw - 32px));
                border: 1px solid rgba(255, 255, 255, 0.55);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.96);
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
                padding: 24px;
                text-align: center;
                color: #111827;
            }}
            .ostb-result-modal img {{
                width: min(680px, 88vw);
                max-height: 68vh;
                object-fit: contain;
                margin-bottom: 16px;
            }}
            .ostb-result-status {{
                color: {accent};
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                margin: 0 0 8px 0;
                letter-spacing: 0;
            }}
            .ostb-result-reason {{
                margin: 0;
                font-size: 0.98rem;
                line-height: 1.35;
            }}
        </style>
        <div class="ostb-result-overlay" aria-live="polite">
            <div class="ostb-result-modal {result_class}">
                <img src="{gif_data_uri}" alt="{status} result">
                <p class="ostb-result-status">{status}</p>
                <p class="ostb-result-reason">{reason}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def standard_check_message(tolerance_report: Dict[str, object]) -> str:
    kind = tolerance_report.get("evaluation_kind", "combined")
    diameter_label = str(tolerance_report.get("compliance_diameter_label", "Compliance diameter"))
    diameter = float(tolerance_report.get("compliance_diameter_mm", float("nan")))
    nominal = float(tolerance_report.get("nominal_mm", float("nan")))
    min_dia = float(tolerance_report.get("min_diameter_mm", float("nan")))
    max_dia = float(tolerance_report.get("max_diameter_mm", float("nan")))
    if kind == "diameter":
        if tolerance_report.get("diameter_ok"):
            return (
                f"PASS: {diameter_label} {diameter:.3f} mm is inside "
                f"{min_dia:.3f}-{max_dia:.3f} mm for nominal {nominal:.3f} mm."
            )
        if np.isfinite(diameter) and np.isfinite(min_dia) and diameter < min_dia:
            return f"FAIL: {diameter_label} {diameter:.3f} mm is below the allowed minimum by {min_dia - diameter:.4f} mm."
        if np.isfinite(diameter) and np.isfinite(max_dia):
            return f"FAIL: {diameter_label} {diameter:.3f} mm is above the allowed maximum by {diameter - max_dia:.4f} mm."
        return "FAIL: diameter tolerance could not be evaluated from the selected standard inputs."
    if kind == "roundness":
        if not tolerance_report.get("ovality_evaluated"):
            return f"Roundness check not evaluated: {tolerance_report.get('ovality_source', 'not mapped')}."
        measured = float(tolerance_report.get("measured_ovality_mm", float("nan")))
        limit = float(tolerance_report.get("max_ovality_mm", float("nan")))
        if tolerance_report.get("roundness_ok"):
            return f"PASS: measured diameter range {measured:.4f} mm is within the ovality limit {limit:.4f} mm."
        return (
            f"FAIL: measured diameter range {measured:.4f} mm exceeds the ovality limit "
            f"{limit:.4f} mm by {float(tolerance_report.get('roundness_excess_mm', 0.0)):.4f} mm."
        )
    return str(tolerance_report.get("failure_reason", "Standards check completed."))


def render_tolerance_summary(tolerance_report: Dict[str, object]) -> None:
    if not tolerance_report.get("evaluation_available", True):
        st.warning(tolerance_report["failure_reason"])
        st.caption(f"Selected standard: {tolerance_report['standard_label']}")
        return

    selected_pass = selected_standard_check_passed(tolerance_report)
    status = "NOT EVALUATED" if selected_pass is None else ("PASS" if selected_pass else "FAIL")
    message = standard_check_message(tolerance_report)
    if selected_pass is None:
        st.warning(message)
    elif selected_pass:
        st.success(message)
    else:
        st.error(message)

    st.caption(f"Selected standard: {tolerance_report['standard_label']} | Check: {tolerance_report.get('evaluation_label', 'Combined')}")
    if tolerance_report.get("selected_entry"):
        st.caption(f"Selected table entry: {entry_display_name(tolerance_report['selected_entry'])}")

    kind = tolerance_report.get("evaluation_kind", "combined")
    if kind == "roundness":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Measured dia range", f"{float(tolerance_report.get('measured_ovality_mm', float('nan'))):.4f} mm")
        c2.metric("Allowed ovality", f"{float(tolerance_report.get('max_ovality_mm', float('nan'))):.4f} mm")
        c3.metric("Excess", f"{float(tolerance_report.get('roundness_excess_mm', float('nan'))):.4f} mm")
        c4.metric("Roundness result", status)
        st.caption(f"Rule/source: {tolerance_report.get('ovality_source', 'not mapped')}")
        if tolerance_report.get("selected_wall_schedule") or tolerance_report.get("selected_wall_thickness_mm"):
            st.caption(
                f"Wall input: schedule {tolerance_report.get('selected_wall_schedule') or 'measured'}; "
                f"thickness {float(tolerance_report.get('selected_wall_thickness_mm', float('nan'))):.3f} mm."
            )
    elif kind == "diameter":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Allowed diameter min", f"{tolerance_report['min_diameter_mm']:.3f} mm")
        c2.metric("Allowed diameter max", f"{tolerance_report['max_diameter_mm']:.3f} mm")
        c3.metric("Nominal diameter", f"{tolerance_report['nominal_mm']:.3f} mm")
        c4.metric(
            str(tolerance_report.get("compliance_diameter_label", "Compliance diameter")),
            f"{float(tolerance_report.get('compliance_diameter_mm', float('nan'))):.3f} mm",
        )
        c5.metric("Diameter result", status)
        if tolerance_report["best_diameter_class"] != "NONE":
            st.caption(f"Diameter tolerance class/method: {tolerance_report['best_diameter_class']}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Allowed diameter min", f"{tolerance_report['min_diameter_mm']:.3f} mm")
        c2.metric("Allowed diameter max", f"{tolerance_report['max_diameter_mm']:.3f} mm")
        c3.metric("Nominal diameter", f"{tolerance_report['nominal_mm']:.3f} mm")
        c4.metric(
            str(tolerance_report.get("compliance_diameter_label", "Compliance diameter")),
            f"{float(tolerance_report.get('compliance_diameter_mm', float('nan'))):.3f} mm",
        )
        c5.metric("Overall", status)

    if tolerance_report["best_diameter_class"] != "NONE":
        if tolerance_report.get("lookup_range_only", False):
            st.caption("This result used an explicit lookup row range, not a formula tolerance class.")
        elif kind != "diameter":
            st.caption(f"Passing tolerance class: {tolerance_report['best_diameter_class']}")
    table = tolerance_report["tolerance_table"].copy()
    if kind == "diameter":
        keep_columns = [
            "class",
            "nominal_mm",
            "nominal_source",
            "min_diameter_mm",
            "max_diameter_mm",
            "diameter_ok",
            "applied_branch",
            "abs_tol_mm",
            "source_text",
        ]
        table = table[[col for col in keep_columns if col in table.columns]]
    elif kind == "roundness":
        keep_columns = [
            "class",
            "nominal_mm",
            "nominal_source",
            "max_ovality_mm",
            "roundness_ok",
            "roundness_excess_mm",
            "ovality_evaluated",
            "ovality_source",
            "source_text",
        ]
        table = table[[col for col in keep_columns if col in table.columns]]
    for bool_column in ["diameter_ok", "roundness_ok", "ovality_evaluated"]:
        if bool_column in table.columns:
            table[bool_column] = table[bool_column].map(
                lambda value: "n/a" if pd.isna(value) else ("✓" if bool(value) else "✕")
            )
    status_columns = [col for col in ["diameter_ok", "roundness_ok"] if col in table.columns]
    if status_columns:
        styled_table = table.style.map(
            lambda value: "color: #15803d; font-weight: 700;" if value == "✓" else "color: #b91c1c; font-weight: 700;",
            subset=status_columns,
        )
        st.dataframe(styled_table, width="stretch")
    else:
        st.dataframe(table, width="stretch")


def render_combined_tolerance_results(combined: Dict[str, object]) -> None:
    """Renders a rich combined report with both diameter and roundness/ovality results."""
    dia_report = combined["diameter_report"]
    oval_report = combined["oval_report"]
    combined_ok = bool(combined["combined_ok"])
    oval_evaluated = bool(combined.get("ovality_evaluated", False))

    # ── Combined result banner ──
    if combined_ok:
        st.success(f"### ✅ {combined['combined_reason']}")
    else:
        st.error(f"### ❌ {combined['combined_reason']}")

    # ── Summary cards row ──
    sc1, sc2, sc3, sc4, sc5 = st.columns(5, gap="small")
    dia_val = float(combined.get("compliance_diameter_mm", float("nan")))
    oval_val = float(combined.get("measured_ovality_mm", float("nan")))
    dia_ok = bool(combined["diameter_ok"])
    oval_ok = bool(combined["roundness_ok"]) if combined["roundness_ok"] is not None else None

    card_html = """
    <style>
    .sc-card {padding:10px 12px;border-radius:8px;border:1px solid #dde5ef;background:#fff;
              box-shadow:0 1px 2px rgba(15,23,42,0.04);text-align:center;}
    .sc-card .sc-label {font-size:0.7rem;color:#6b7c93;text-transform:uppercase;letter-spacing:0.03em;}
    .sc-card .sc-value {font-size:1.05rem;font-weight:800;margin-top:2px;}
    .sc-pass {color:#0d7d33;} .sc-fail {color:#b91c1c;} .sc-na {color:#6b7c93;}
    </style>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    with sc1:
        st.markdown(f"""<div class="sc-card"><div class="sc-label">Diameter</div>
            <div class="sc-value {'sc-pass' if dia_ok else 'sc-fail'}">{'✅ PASS' if dia_ok else '❌ FAIL'}</div></div>""", unsafe_allow_html=True)
    with sc2:
        if oval_evaluated and oval_ok is not None:
            st.markdown(f"""<div class="sc-card"><div class="sc-label">Roundness</div>
                <div class="sc-value {'sc-pass' if oval_ok else 'sc-fail'}">{'✅ PASS' if oval_ok else '❌ FAIL'}</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="sc-card"><div class="sc-label">Roundness</div>
                <div class="sc-value sc-na">— not evaluated</div></div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""<div class="sc-card"><div class="sc-label">Overall</div>
            <div class="sc-value {'sc-pass' if combined_ok else 'sc-fail'}">{'✅ PASS' if combined_ok else '❌ FAIL'}</div></div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""<div class="sc-card"><div class="sc-label">Measured Ø</div>
            <div class="sc-value" style="color:#111827;">{dia_val:.3f} mm</div></div>""", unsafe_allow_html=True)
    with sc5:
        st.markdown(f"""<div class="sc-card"><div class="sc-label">Measured Ovality</div>
            <div class="sc-value" style="color:#111827;">{oval_val:.4f} mm</div></div>""", unsafe_allow_html=True)

    # ── Standard reference card ──
    std_id = str(combined.get("standard_id", ""))
    std_title = STANDARD_UI_TITLE_BY_ID.get(std_id, combined.get("standard_label", ""))
    image_name = STANDARD_IMAGE_BY_ID.get(std_id)
    image_path = APP_DIR / "images" / image_name if image_name else None

    std_card_cols = st.columns([0.15, 1.0], gap="small", vertical_alignment="center")
    with std_card_cols[0]:
        if image_path is not None and image_path.exists():
            st.image(str(image_path), width=72)
        else:
            st.markdown("📋")
    with std_card_cols[1]:
        st.markdown(f"**Standard {std_id}** — {combined.get('standard_label', '')}")
        st.caption(std_title)

    # ── Two report panels side by side ──
    st.markdown("---")
    left_panel, right_panel = st.columns(2, gap="large")

    with left_panel:
        st.markdown("#### 📏 Diameter Tolerance")
        dia_ok = bool(combined["diameter_ok"])
        color = "#0d7d33" if dia_ok else "#b91c1c"
        st.markdown(f"**Status:** <span style='color:{color};font-size:1.1rem;'>{'✅ PASS' if dia_ok else '❌ FAIL'}</span>", unsafe_allow_html=True)

        min_dia = float(combined.get("min_diameter_mm", float("nan")))
        max_dia = float(combined.get("max_diameter_mm", float("nan")))
        nom = float(combined.get("nominal_mm", float("nan")))
        if np.isfinite(min_dia) and np.isfinite(max_dia) and np.isfinite(dia_val):
            fail_amt = float(combined.get("diameter_fail_amount_mm", 0))
            # Mini range bar
            range_span = max_dia - min_dia
            if range_span > 0:
                bar_margin = range_span * 0.15
                bar_min = min_dia - bar_margin
                bar_max = max_dia + bar_margin
                dia_pct = max(0, min(100, (dia_val - bar_min) / (bar_max - bar_min) * 100))
                min_pct = max(0, min(100, (min_dia - bar_min) / (bar_max - bar_min) * 100))
                max_pct = max(0, min(100, (max_dia - bar_min) / (bar_max - bar_min) * 100))
                in_range = min_dia <= dia_val <= max_dia
                marker_color = "#0d7d33" if in_range else "#dc2626"
                st.markdown(
                    f"""
                    <div style="margin:6px 0 2px 0;">
                        <div style="font-size:0.78rem;color:#52616f;margin-bottom:2px;">
                            {dia_val:.4f} mm  &nbsp;|&nbsp;  Allowed: {min_dia:.4f} – {max_dia:.4f} mm
                            &nbsp;|&nbsp; Nominal: {nom:.3f} mm
                        </div>
                        <div style="height:14px;border-radius:7px;background:#e5e7eb;position:relative;overflow:hidden;">
                            <div style="position:absolute;left:{min_pct:.1f}%;width:{max_pct - min_pct:.1f}%;height:100%;
                                 background:rgba(22,163,74,0.30);border-radius:7px;"></div>
                            <div style="position:absolute;left:{dia_pct:.1f}%;top:-2px;width:4px;height:18px;
                                 background:{marker_color};border-radius:2px;"
                                 title="Measured: {dia_val:.4f} mm"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        dia_table = dia_report.get("tolerance_table")
        if isinstance(dia_table, pd.DataFrame) and not dia_table.empty:
            keep = ["class", "diameter_ok", "nominal_mm", "min_diameter_mm", "max_diameter_mm", "source_text"]
            tbl = dia_table[[c for c in keep if c in dia_table.columns]].copy()
            if "diameter_ok" in tbl.columns:
                tbl["diameter_ok"] = tbl["diameter_ok"].map(lambda v: "✓" if v is True else ("✕" if v is False else "n/a"))
            row_h = min(len(tbl), 3) * 38 + 38
            st.dataframe(tbl, width="stretch", height=row_h)

    with right_panel:
        st.markdown("#### ⭕ Roundness / Ovality")
        if not oval_evaluated:
            st.info("This standard does not have a mapped ovality rule. Only diameter tolerance was checked.")
            st.caption(f"Source: {oval_report.get('ovality_source', 'not mapped')}")
        else:
            oval_ok = bool(combined["roundness_ok"])
            color = "#0d7d33" if oval_ok else "#b91c1c"
            st.markdown(f"**Status:** <span style='color:{color};font-size:1.1rem;'>{'✅ PASS' if oval_ok else '❌ FAIL'}</span>", unsafe_allow_html=True)

            oval_limit = float(combined.get("max_ovality_mm", float("nan")))
            oval_excess = float(combined.get("roundness_excess_mm", 0))
            if np.isfinite(oval_val) and np.isfinite(oval_limit):
                bar_max_val = max(oval_val, oval_limit) * 1.25
                oval_pct = min(100, oval_val / bar_max_val * 100)
                limit_pct = min(100, oval_limit / bar_max_val * 100)
                marker_color = "#0d7d33" if oval_ok else "#dc2626"
                st.markdown(
                    f"""
                    <div style="margin:6px 0 2px 0;">
                        <div style="font-size:0.78rem;color:#52616f;margin-bottom:2px;">
                            Measured range: {oval_val:.4f} mm  &nbsp;|&nbsp;  Limit: {oval_limit:.4f} mm
                            &nbsp;|&nbsp; Excess: {oval_excess:.4f} mm
                        </div>
                        <div style="height:14px;border-radius:7px;background:#e5e7eb;position:relative;overflow:hidden;">
                            <div style="position:absolute;left:0;width:{limit_pct:.1f}%;height:100%;
                                 background:rgba(22,163,74,0.30);border-radius:7px;"></div>
                            <div style="position:absolute;left:{limit_pct:.1f}%;width:{max(0, oval_pct - limit_pct):.1f}%;height:100%;
                                 background:rgba(220,38,38,0.25);border-radius:0 7px 7px 0;"></div>
                            <div style="position:absolute;left:{oval_pct:.1f}%;top:-2px;width:4px;height:18px;
                                 background:{marker_color};border-radius:2px;"
                                 title="Measured: {oval_val:.4f} mm"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"Rule: {combined.get('ovality_source', '')}")
            if combined.get("selected_wall_schedule") or combined.get("selected_wall_thickness_mm"):
                st.caption(
                    f"Wall input: schedule {combined.get('selected_wall_schedule') or 'measured'}; "
                    f"thickness {float(combined.get('selected_wall_thickness_mm', float('nan'))):.3f} mm."
                )

        oval_table = oval_report.get("tolerance_table")
        if isinstance(oval_table, pd.DataFrame) and not oval_table.empty and oval_evaluated:
            keep = ["class", "roundness_ok", "nominal_mm", "max_ovality_mm", "roundness_excess_mm", "ovality_source"]
            tbl = oval_table[[c for c in keep if c in oval_table.columns]].copy()
            if "roundness_ok" in tbl.columns:
                tbl["roundness_ok"] = tbl["roundness_ok"].map(lambda v: "✓" if v is True else ("✕" if v is False else "n/a"))
            row_h = min(len(tbl), 3) * 38 + 38
            st.dataframe(tbl, width="stretch", height=row_h)

    # ── Formula explanation ──
    st.markdown("---")
    with st.expander("📐 How these checks are computed — formulas & derivation", expanded=False):
        st.markdown(
            """
            ### Diameter Tolerance Check
            """
        )
        st.latex(r"\text{abs\_tol} = \max\left(\text{nominal} \times \frac{\text{percent}}{100},\; \text{min\_mm}\right)")
        st.latex(r"\text{min\_allowed} = \text{nominal} - \text{abs\_tol}")
        st.latex(r"\text{max\_allowed} = \text{nominal} + \text{abs\_tol}")

        dia_val = float(combined.get("compliance_diameter_mm", 0))
        nom_val = float(combined.get("nominal_mm", 0))
        min_allowed = float(combined.get("min_diameter_mm", 0))
        max_allowed = float(combined.get("max_diameter_mm", 0))
        st.markdown(
            f"""
            | Variable | Computed value |
            |---|---|
            | Measured diameter | **{dia_val:.4f} mm** |
            | Nominal diameter | **{nom_val:.3f} mm** |
            | Allowed range | **{min_allowed:.4f} – {max_allowed:.4f} mm** |
            | Result | **{'✅ PASS' if combined['diameter_ok'] else '❌ FAIL'}** |
            """
        )

        st.markdown("---")
        st.markdown(
            """
            ### Ovality / Roundness Check
            """
        )
        if oval_evaluated:
            st.latex(r"\text{ovality\_limit} = \begin{cases} \text{max\_dia} - \text{min\_dia} & \text{(diameter\_tolerance\_range mode)} \\ \text{lookup\_table\_value} & \text{(lookup\_table mode)} \\ \text{nominal} \times \text{percent\_of\_od} / 100 & \text{(formula mode)} \end{cases}")
            st.markdown(
                f"""
                | Variable | Computed value |
                |---|---|
                | Measured diameter range (ovality) | **{float(combined.get('measured_ovality_mm', 0)):.4f} mm** |
                | Allowed ovality limit | **{float(combined.get('max_ovality_mm', 0)):.4f} mm** |
                | Result | **{'✅ PASS' if combined['roundness_ok'] else '❌ FAIL'}** |
                """
            )
        else:
            st.info("This standard does not have a mapped ovality rule in the OSTB contract.")


def render_measurement_quality(measurement: PipeMeasurement, detection_note: str, scale_source: str) -> None:
    inlier_ratio = len(measurement.fitting_inliers) / max(len(measurement.contour_points), 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Contour points",
        f"{len(measurement.contour_points)}",
        help="Number of edge points selected on the pipe rim. More usable points usually means a more stable fit. There is no fixed ideal, but hundreds to thousands of points is generally strong; very low counts need visual review.",
    )
    c2.metric(
        "Fitting inliers",
        f"{len(measurement.fitting_inliers)}",
        help="Number of contour points accepted by RANSAC as belonging to the fitted circle. More inliers usually means the detected rim is consistent and less affected by noise or stray edges.",
    )
    c3.metric(
        "Inlier ratio",
        f"{inlier_ratio * 100.0:.1f}%",
        help="Fitting inliers divided by contour points. Higher is better. 70%+ is usually strong, 50-70% is usable but worth reviewing, and below 50% often means the edge map or ROI should be checked.",
    )
    c4.metric(
        "Scale source",
        scale_source.replace("_", " "),
        help="Shows how the measurement is being scaled. Pixel-only gives no real-world millimeter result; known diameter or a reference image enables millimeter tolerance checks.",
    )
    st.caption(detection_note.replace("ROI", "ROI (Region of Interest)"))

    with st.expander("🔍 How the circle is fitted — RANSAC outlier rejection", expanded=False):
        st.markdown(
            f"""
            **RANSAC (RANdom SAmple Consensus)** is a robust fitting algorithm that ignores outliers.

            1. Randomly picks 3 edge points → fits a circle  
            2. Counts how many other points agree (**inliers**)  
            3. Repeats 1,000 times; keeps the circle with most inliers  
            4. Only the inliers are then used for a precision **least-squares** refinement

            **Measurement:** {len(measurement.contour_points)} contour points → **{len(measurement.fitting_inliers)}** RANSAC inliers ({inlier_ratio * 100.0:.1f}%).

            > Without RANSAC, a single glare spot or shadow edge could shift the entire fit.
            """
        )


def render_pixel_interpretation(measurement: PipeMeasurement) -> None:
    relative_roundness = measurement.roundness_px / measurement.diameter_px * 100.0
    inlier_ratio = len(measurement.fitting_inliers) / max(len(measurement.contour_points), 1)

    if relative_roundness <= 0.5 and inlier_ratio >= 0.70:
        status = "Excellent Circle"
        symbol = "🟢"
        message = "Very close to the ideal circle."
        summary_class = "good"
        meter_color = "#16a34a"
    elif relative_roundness <= 1.0 and inlier_ratio >= 0.60:
        status = "Good Circle"
        symbol = "🟢"
        message = "Small deviation from the ideal circle."
        summary_class = "good"
        meter_color = "#16a34a"
    elif relative_roundness <= 2.0 or inlier_ratio >= 0.50:
        status = "Check Circle"
        symbol = "🟡"
        message = "Usable, but review the edge map."
        summary_class = "warn"
        meter_color = "#d97706"
    else:
        status = "Needs Review"
        symbol = "🔴"
        message = "High deviation or weak circle fit."
        summary_class = "bad"
        meter_color = "#dc2626"

    deviation_quality = max(0.0, min(100.0, 100.0 - (relative_roundness / 3.0 * 100.0)))
    fit_quality = max(0.0, min(100.0, inlier_ratio * 100.0))
    st.markdown(
        f"""
        <div class="circle-summary {summary_class}">
            <div class="summary-row">
                <div class="summary-symbol">{symbol}</div>
                <div>
                    <div class="summary-title">{status}</div>
                    <div class="summary-message">{message}</div>
                </div>
            </div>
            <div class="summary-grid">
                <div>
                    <div class="summary-label">Circle shape</div>
                    <div class="summary-value">Relative roundness {relative_roundness:.3f}%</div>
                    <div class="meter"><div class="meter-fill" style="width: {deviation_quality:.1f}%; background: {meter_color};"></div></div>
                </div>
                <div>
                    <div class="summary-label">Fit confidence</div>
                    <div class="summary-value">{fit_quality:.1f}% of rim points support the fit</div>
                    <div class="meter"><div class="meter-fill" style="width: {fit_quality:.1f}%; background: {meter_color};"></div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Roundness {measurement.roundness_px:.3f} px | Max deviation {measurement.max_abs_deviation_px:.3f} px")
    with st.expander("How to read this"):
        st.markdown(
            """
            - **Circle shape**: farther to the right is better. It means the rim is closer to the ideal fitted circle.
            - **Relative roundness**: smaller is better. It compares rim variation with the measured diameter.
            - **Fit confidence**: farther to the right is better. It means more rim points agree with the fitted circle.
            - **Roundness**: total high-to-low rim variation in pixels.
            - **Max deviation**: biggest single distance from the ideal circle.

            Use this as a quick shape and detection-quality check. For final acceptance, add a millimeter scale and run the tolerance check.
            """
        )


def roundness_quality_label(relative_roundness: float) -> Tuple[str, str]:
    if relative_roundness <= 0.5:
        return "Nearly round", "The detected edge stays very close to the fitted circle."
    if relative_roundness <= 1.0:
        return "Small deviation", "The edge has visible variation, but it is still compact relative to diameter."
    if relative_roundness <= 2.0:
        return "Noticeable deviation", "Review the rim overlay and deviation plots before accepting the result."
    return "High deviation", "The detected edge varies strongly from the fitted circle or includes noisy rim points."


def render_roundness_evaluation(measurement: PipeMeasurement, unit: str, feature_type: str = "inner") -> Dict[str, object]:
    """
    Educational, interactive roundness evaluation based on ISO 12181 concepts.
    Explains LSC, MZC, MCC, and MIC on the actual measurement data.
    Includes method comparison, formulas, computed values, and focused radius-line charts.
    """
    scale = measurement.mm_per_pixel if unit == "mm" else 1.0
    roundness_methods = compute_roundness_method_stats(measurement)

    lsc_center = np.array([measurement.center_x_px, measurement.center_y_px], dtype=np.float64)
    lsc_radius = float(measurement.radius_px)

    mzc_center = np.array(
        [roundness_methods["mzc_center_x_px"], roundness_methods["mzc_center_y_px"]],
        dtype=np.float64,
    )
    mzc_inner = float(roundness_methods["mzc_inner_radius_px"])
    mzc_outer = float(roundness_methods["mzc_outer_radius_px"])

    mcc_center = np.array(
        [roundness_methods["mcc_center_x_px"], roundness_methods["mcc_center_y_px"]],
        dtype=np.float64,
    )
    mcc_radius = float(roundness_methods["mcc_radius_px"])

    mic_center = np.array(
        [roundness_methods["mic_center_x_px"], roundness_methods["mic_center_y_px"]],
        dtype=np.float64,
    )
    mic_radius = float(roundness_methods["mic_radius_px"])

    roundness_lsc = float(roundness_methods["lsc_roundness_px"] * scale)
    roundness_mzc = float(roundness_methods["mzc_roundness_px"] * scale)
    roundness_mcc = float(roundness_methods["mcc_roundness_px"] * scale)
    mcc_diameter = float(mcc_radius * 2.0 * scale)
    mic_diameter = float(roundness_methods["mic_diameter_px"] * scale)
    value = measurement.roundness_mm if unit == "mm" else measurement.roundness_px
    points = measurement.contour_points.astype(np.float64)

    def format_value(value_to_format: float, suffix: str = unit) -> str:
        if not np.isfinite(value_to_format):
            return "not available"
        return f"{value_to_format:.4f} {suffix}"

    def format_center(center_xy: np.ndarray) -> str:
        if not np.all(np.isfinite(center_xy)):
            return "not available"
        return f"({center_xy[0]:.1f}, {center_xy[1]:.1f}) px"

    st.markdown("## Roundness Evaluation (ISO 12181 Concept)")
    st.markdown(
        """
        Roundness quantifies how much a real profile deviates from a perfect circle.
        This section explains the four main evaluation methods and shows how they apply to the measurement.
        """
    )

    st.markdown("### 🔍 Understanding the four methods")
    st.caption("Each method fits a circle differently. Expand the analogy to grasp the concept intuitively.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### LSC - Least Squares Circle")
            st.caption("**Definition** - Minimizes the sum of squared radial distances from the circle to the measured points.")
            st.markdown(
                f"""
                - **Center** `{format_center(lsc_center)}`
                - **Radius** `{format_value(lsc_radius * scale)}`
                - **Roundness error** `{format_value(roundness_lsc)}`
                """
            )
            with st.expander("🔎 Real-world analogy"):
                st.markdown(
                    """
                    Imagine pebbles scattered roughly in a circle.
                    You want to draw **one perfect circle** that passes as close as possible to **all** pebbles,
                    balancing the distances so no single pebble dominates.

                    LSC finds that **best compromise** circle, which makes it useful for noisy vision measurements.
                    """
                )
            st.markdown("✅ Best for noisy vision data and quick screening.")

        with st.container(border=True):
            st.markdown("#### MCC - Minimum Circumscribed Circle")
            st.caption("**Definition** - Smallest circle that encloses all measured points.")
            st.markdown(
                f"""
                - **Center** `{format_center(mcc_center)}`
                - **Radius** `{format_value(mcc_radius * scale)}`
                - **Enclosing-circle deviation** `{format_value(roundness_mcc)}`
                - **Interpretation** enclosing radius minus nearest inner point
                """
            )
            with st.expander("🔎 Real-world analogy"):
                st.markdown(
                    """
                    Think of putting a **rubber band** around nails on a board.
                    The rubber band stretches to touch the **outermost nails**, forming the smallest circle that contains everything.

                    MCC is that enclosing circle. It helps answer what minimum tube size would **fit over** the part.
                    """
                )
            st.markdown("⚠️ Sensitive to outliers; useful for checking maximum outward deviation.")

    with col2:
        with st.container(border=True):
            st.markdown("#### MZC - Minimum Zone Circle")
            st.caption("**Definition** - Two concentric circles that contain all points with the smallest radial separation.")
            st.markdown(
                f"""
                - **Center** `{format_center(mzc_center)}`
                - **Inner / outer radii** `{format_value(mzc_inner * scale)}` / `{format_value(mzc_outer * scale)}`
                - **Roundness error** `{format_value(roundness_mzc)}`
                """
            )
            with st.expander("🔎 Real-world analogy"):
                st.markdown(
                    """
                    Imagine placing the same pebbles between **two concentric rings**, like a doughnut-shaped zone.
                    You can slide the rings around and adjust their size until the **gap between them is as small as possible**
                    while still containing all pebbles.

                    The width of that gap is the roundness error. MZC finds the **tightest possible zone**.
                    """
                )
            st.markdown("🎯 Closest to strict ISO roundness evaluation.")

        with st.container(border=True):
            st.markdown("#### MIC - Maximum Inscribed Circle")
            st.caption("**Definition** - Largest circle that fits entirely inside the measured profile.")
            st.markdown(
                f"""
                - **Center** `{format_center(mic_center)}`
                - **Radius** `{format_value(mic_radius * scale)}`
                - **Diameter** `{format_value(mic_diameter)}` (clearance value)
                """
            )
            with st.expander("🔎 Real-world analogy"):
                st.markdown(
                    """
                    Picture a **coin** dropped into a slightly irregular hole.
                    The coin can only be as large as the **narrowest part** of the hole allows.

                    MIC finds that largest coin. It tells you the maximum **rod or pin** that would **fit inside** the opening.
                    """
                )
            st.markdown("📏 Not a roundness error - used for go/no-go gauging and clearance review.")

    def add_circle_trace(
        fig: go.Figure,
        center_xy: np.ndarray,
        radius_px: float,
        name: str,
        color: str,
        dash: Optional[str] = None,
        visible: object = "legendonly",
        extra: str = "",
    ) -> None:
        if not np.all(np.isfinite(center_xy)) or not np.isfinite(radius_px) or radius_px <= 0:
            return
        theta = np.linspace(0, 2.0 * math.pi, 240)
        line = dict(color=color, width=2)
        if dash is not None:
            line["dash"] = dash
        hover = (
            f"<b>{name}</b><br>"
            f"Center: ({center_xy[0]:.2f}, {center_xy[1]:.2f}) px<br>"
            f"Radius: {radius_px * scale:.4f} {unit}<br>"
            f"{extra}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=center_xy[0] + radius_px * np.cos(theta),
                y=center_xy[1] + radius_px * np.sin(theta),
                mode="lines",
                name=name,
                line=line,
                visible=visible,
                hoverinfo="text",
                hovertext=hover,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[center_xy[0]],
                y=[center_xy[1]],
                mode="markers",
                marker=dict(color=color, size=8, symbol="x"),
                showlegend=False,
                visible=visible,
                hoverinfo="text",
                hovertext=f"<b>{name} center</b><br>({center_xy[0]:.2f}, {center_xy[1]:.2f}) px<extra></extra>",
            )
        )

    with st.expander("Compare all methods", expanded=False):
        st.caption("Toggle circles in the legend. Hover to see center, radius, and computed value.")
        fig_all = go.Figure()
        fig_all.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                name="Measured points",
                marker=dict(color="#6b7280", size=3, opacity=0.42),
                hoverinfo="skip",
            )
        )
        add_circle_trace(fig_all, lsc_center, lsc_radius, "LSC", "#1f77b4", extra=f"Roundness error: {format_value(roundness_lsc)}")
        add_circle_trace(fig_all, mzc_center, mzc_inner, "MZC inner", "#ff7f0e", dash="dot", extra=f"Roundness error: {format_value(roundness_mzc)}")
        add_circle_trace(fig_all, mzc_center, mzc_outer, "MZC outer", "#ff7f0e", extra=f"Roundness error: {format_value(roundness_mzc)}")
        add_circle_trace(fig_all, mcc_center, mcc_radius, "MCC", "#2ca02c", extra=f"Roundness error: {format_value(roundness_mcc)}")
        add_circle_trace(fig_all, mic_center, mic_radius, "MIC", "#d62728", dash="dash", extra=f"Inner diameter: {format_value(mic_diameter)}")
        fig_all.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(scaleanchor="y", title="x (px)"),
            yaxis=dict(title="y (px)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="closest",
        )
        st.plotly_chart(fig_all, use_container_width=True)
        st.info(
            "Each method can have a different center. LSC is a best fit, MZC minimizes the radial zone, "
            "and MCC/MIC are extremal fits."
        )

    st.markdown("### Explore one method in detail")
    selected_method = st.radio(
        "Select a method to see its formula, values, and radius line",
        ["None", "LSC", "MZC", "MCC", "MIC"],
        horizontal=True,
        index=0,
        key="roundness_method_detail_selector",
    )
    method_data = {
        "LSC": {
            "center": lsc_center,
            "radius": lsc_radius,
            "color": "#1f77b4",
            "roundness": roundness_lsc,
            "formula": r"\min \sum_{i=1}^{n} \left( \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2} - R \right)^2",
            "description": "Least Squares Circle: minimizes squared radial distances.",
        },
        "MZC": {
            "center": mzc_center,
            "inner": mzc_inner,
            "outer": mzc_outer,
            "color": "#ff7f0e",
            "roundness": roundness_mzc,
            "formula": r"\min_{x_c,y_c} \left( \max_i d_i - \min_i d_i \right)",
            "description": "Minimum Zone Circle: two concentric circles with minimal radial separation.",
        },
        "MCC": {
            "center": mcc_center,
            "radius": mcc_radius,
            "color": "#2ca02c",
            "roundness": roundness_mcc,
            "formula": r"\min R \quad \text{s.t.} \quad \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2} \le R",
            "description": "Minimum Circumscribed Circle: smallest circle enclosing all points.",
        },
        "MIC": {
            "center": mic_center,
            "radius": mic_radius,
            "color": "#d62728",
            "diameter": mic_diameter,
            "formula": r"\max R \quad \text{s.t.} \quad \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2} \ge R",
            "description": "Maximum Inscribed Circle: largest circle fitting inside the profile.",
        },
    }

    data = method_data.get(selected_method)
    detail_left, detail_right = st.columns([1.2, 1.0], gap="large")
    with detail_left:
        if data is None:
            st.caption("No method selected. The chart shows only the measured edge points.")
        else:
            st.latex(data["formula"])
            st.caption(data["description"])
        fig_focus = go.Figure()
        fig_focus.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                name="Measured points",
                marker=dict(color="#6b7280", size=3, opacity=0.5),
                hoverinfo="skip",
            )
        )
        theta = np.linspace(0, 2.0 * math.pi, 240)
        if data is None:
            pass
        elif selected_method == "MZC" and np.all(np.isfinite(data["center"])) and np.isfinite(data["inner"]) and np.isfinite(data["outer"]):
            center = data["center"]
            color = data["color"]
            fig_focus.add_trace(
                go.Scatter(
                    x=center[0] + data["inner"] * np.cos(theta),
                    y=center[1] + data["inner"] * np.sin(theta),
                    mode="lines",
                    name="MZC inner",
                    line=dict(color=color, width=2, dash="dot"),
                )
            )
            fig_focus.add_trace(
                go.Scatter(
                    x=center[0] + data["outer"] * np.cos(theta),
                    y=center[1] + data["outer"] * np.sin(theta),
                    mode="lines",
                    name="MZC outer",
                    line=dict(color=color, width=2),
                )
            )
            for radius_key, label, y_offset, dash in [
                ("outer", "outer R", -35, "solid"),
                ("inner", "inner R", 35, "dot"),
            ]:
                radius = data[radius_key]
                end_x = center[0] + radius
                end_y = center[1]
                fig_focus.add_shape(
                    type="line",
                    x0=center[0],
                    y0=center[1],
                    x1=end_x,
                    y1=end_y,
                    line=dict(color=color, width=2, dash=dash),
                )
                fig_focus.add_annotation(
                    x=end_x,
                    y=end_y,
                    text=f"{label} = {radius * scale:.3f} {unit}",
                    showarrow=True,
                    arrowhead=2,
                    ax=35,
                    ay=y_offset,
                )
            fig_focus.add_trace(
                go.Scatter(
                    x=[center[0]],
                    y=[center[1]],
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="x"),
                    name="MZC center",
                )
            )
        elif np.all(np.isfinite(data["center"])) and np.isfinite(data["radius"]) and data["radius"] > 0:
            center = data["center"]
            color = data["color"]
            radius = data["radius"]
            fig_focus.add_trace(
                go.Scatter(
                    x=center[0] + radius * np.cos(theta),
                    y=center[1] + radius * np.sin(theta),
                    mode="lines",
                    name=selected_method,
                    line=dict(color=color, width=2),
                )
            )
            end_x = center[0] + radius
            end_y = center[1]
            fig_focus.add_shape(
                type="line",
                x0=center[0],
                y0=center[1],
                x1=end_x,
                y1=end_y,
                line=dict(color=color, width=2),
            )
            label_text = f"R = {radius * scale:.4f} {unit}"
            if selected_method == "MIC":
                label_text = f"R = {radius * scale:.4f} {unit}<br>D = {format_value(data['diameter'])}"
            fig_focus.add_annotation(
                x=end_x,
                y=end_y,
                text=label_text,
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-40,
            )
            fig_focus.add_trace(
                go.Scatter(
                    x=[center[0]],
                    y=[center[1]],
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="x"),
                    name=f"{selected_method} center",
                )
            )
        else:
            st.warning(f"{selected_method} geometry is not available for this measurement.")

        fig_focus.update_layout(
            height=410,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(scaleanchor="y", title="x (px)"),
            yaxis=dict(title="y (px)"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_focus, use_container_width=True)

    with detail_right:
        st.markdown("#### Computed values")
        if data is None:
            st.metric("Selected method", "None")
            st.caption("Select LSC, MZC, MCC, or MIC to view the formula, fitted geometry, and computed values.")
        else:
            center = data["center"]
            st.metric("Center (px)", f"({center[0]:.2f}, {center[1]:.2f})" if np.all(np.isfinite(center)) else "not available")
        if selected_method == "MZC" and data is not None:
            st.metric("Inner radius", format_value(data["inner"] * scale))
            st.metric("Outer radius", format_value(data["outer"] * scale))
            st.metric("Roundness error", format_value(data["roundness"]))
        elif selected_method == "MIC" and data is not None:
            st.metric("Radius", format_value(data["radius"] * scale))
            st.metric("Inner diameter", format_value(data["diameter"]))
            st.caption("MIC is a size/clearance value, not a roundness error.")
        elif data is not None:
            st.metric("Radius", format_value(data["radius"] * scale))
            st.metric("Roundness error", format_value(data["roundness"]))
        if selected_method not in {"None", "MIC"}:
            st.caption("Roundness error = maximum radial distance - minimum radial distance.")

    reference_diameter = measurement.diameter_mm if unit == "mm" else measurement.diameter_px
    diameter_stats = measurement_diameter_stats(measurement)
    diameter_scale = measurement.mm_per_pixel if unit == "mm" else 1.0
    real_edge_min_diameter = diameter_stats["real_edge_min_diameter_px"] * diameter_scale
    real_edge_max_diameter = diameter_stats["real_edge_max_diameter_px"] * diameter_scale
    real_edge_diameter_range = diameter_stats["real_edge_diameter_range_px"] * diameter_scale

    def format_error_with_percent(error_value: float) -> str:
        if not np.isfinite(error_value):
            return format_value(error_value)
        if not np.isfinite(reference_diameter) or reference_diameter <= 0:
            return format_value(error_value)
        percent_value = error_value / reference_diameter * 100.0
        return f"{format_value(error_value)} ({percent_value:.2f}%)"

    def render_value_card(column, label: str, primary_value: str, secondary_value: str | None = None) -> None:
        secondary_html = f"<div class='ostb-method-card-secondary'>{secondary_value}</div>" if secondary_value else ""
        column.markdown(
            f"""
            <div class="ostb-method-card">
                <div class="ostb-method-card-label">{label}</div>
                <div class="ostb-method-card-value">{primary_value}</div>
                {secondary_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <style>
        .ostb-method-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin: 0 0 14px 0;
        }
        .ostb-method-chip {
            display:inline-flex;
            align-items:center;
            padding: 4px 10px;
            border-radius:999px;
            font-size:0.82rem;
            font-weight:600;
            border:1px solid #d6e4f0;
            background:#f7fafc;
            color:#1f2937;
        }
        .ostb-method-card {
            min-height: 132px;
            padding: 16px 18px;
            border: 1px solid rgba(49, 83, 120, 0.22);
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .ostb-method-card-label {
            font-size: 0.95rem;
            color: #1f2937;
            line-height: 1.35;
            margin-bottom: 12px;
        }
        .ostb-method-card-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
            word-break: break-word;
        }
        .ostb-method-card-secondary {
            margin-top: 6px;
            font-size: 0.96rem;
            font-weight: 600;
            color: #475569;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ISO method values at a glance")
    cols = st.columns(4)
    render_value_card(cols[0], "LSC roundness error", format_value(roundness_lsc), f"{roundness_lsc / reference_diameter * 100.0:.2f}%" if np.isfinite(roundness_lsc) and np.isfinite(reference_diameter) and reference_diameter > 0 else None)
    render_value_card(cols[1], "MZC roundness error", format_value(roundness_mzc), f"{roundness_mzc / reference_diameter * 100.0:.2f}%" if np.isfinite(roundness_mzc) and np.isfinite(reference_diameter) and reference_diameter > 0 else None)
    render_value_card(cols[2], "MCC diameter", format_value(mcc_diameter), f"Enclosing deviation: {format_value(roundness_mcc)}" if np.isfinite(roundness_mcc) else None)
    render_value_card(cols[3], "MIC clearance diameter", format_value(mic_diameter))

    # ── Diameter Reference Method Selector ──
    is_outer = feature_type == "outer"
    feature_heading = "Outer pipe diameter" if is_outer else "Inner/opening rim"
    lsc_diameter = measurement.diameter_mm if unit == "mm" else measurement.diameter_px
    strict_diameter = mcc_diameter if is_outer else mic_diameter
    strict_label = "MCC diameter" if is_outer else "MIC diameter"
    strict_subtitle = "Smallest enclosing circle" if is_outer else "Largest inscribed circle"
    strict_analogy = "Smallest tube that fits over" if is_outer else "Largest rod that fits inside"

    if "diameter_ref_method" not in st.session_state:
        st.session_state["diameter_ref_method"] = "lsc"

    st.markdown("### 📏 Diameter Reference Method")
    st.caption("Choose which diameter value is used for the standards tolerance check below.")

    card_css = """
    <style>
    .drm-card-row {
        display: flex;
        gap: 16px;
        margin: 12px 0 8px 0;
    }
    .drm-card {
        flex: 1;
        border: 2px solid #dde5ef;
        border-radius: 12px;
        padding: 20px 22px;
        cursor: pointer;
        transition: all 0.2s ease;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        position: relative;
    }
    .drm-card:hover {
        border-color: #93b5e0;
        box-shadow: 0 4px 14px rgba(31,111,235,0.10);
    }
    .drm-card.active {
        border-color: #1f6feb;
        background: #f6faff;
        box-shadow: 0 2px 12px rgba(31,111,235,0.12);
    }
    .drm-card-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }
    .drm-badge-recommended {
        background: #daf5e1;
        color: #0d7d33;
    }
    .drm-badge-precision {
        background: #fff0d9;
        color: #b45f06;
    }
    .drm-card-icon {
        font-size: 1.4rem;
        margin-right: 6px;
        vertical-align: middle;
    }
    .drm-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin: 4px 0 2px 0;
    }
    .drm-card-subtitle {
        font-size: 0.82rem;
        color: #52616f;
        margin-bottom: 10px;
    }
    .drm-card-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1f6feb;
        line-height: 1.15;
    }
    .drm-card-unit {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7c93;
    }
    .drm-card-bullets {
        margin-top: 12px;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.55;
    }
    .drm-card-bullets span {
        display: block;
    }
    .drm-bullet-good { color: #0d7d33; }
    .drm-bullet-warn { color: #b45f06; }
    .drm-selected-bar {
        margin-top: 6px;
        padding: 8px 14px;
        background: #f0f6ff;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #1e3a5f;
        font-weight: 600;
        border-left: 4px solid #1f6feb;
    }
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    lsc_active = st.session_state["diameter_ref_method"] == "lsc"
    strict_active = st.session_state["diameter_ref_method"] == "strict"

    card_left, card_right = st.columns(2, gap="medium")

    with card_left:
        lsc_border = "2px solid #1f6feb" if lsc_active else "2px solid #dde5ef"
        lsc_bg = "#f6faff" if lsc_active else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{lsc_border};border-radius:12px;padding:20px 22px;background:{lsc_bg};box-shadow:0 1px 3px rgba(15,23,42,0.04);cursor:pointer;min-height:260px;"
                 onclick="document.getElementById('drm-btn-lsc').click();">
                <span class="drm-card-badge drm-badge-recommended">✓ RECOMMENDED</span>
                <div class="drm-card-title">🏭 LSC Fitted Diameter</div>
                <div class="drm-card-subtitle">Least-Squares Circle — best-fit average</div>
                <div class="drm-card-value">{format_value(lsc_diameter)}</div>
                <div class="drm-card-unit">Robust reference diameter</div>
                <div class="drm-card-bullets">
                    <span>✅ <b>Stable with noise</b> — blur, glare, shadows barely affect it</span>
                    <span>✅ <b>Matches caliper / pi-tape</b> — what shop-floor inspectors recognize</span>
                    <span>✅ <b>Real-world ready</b> — honest &amp; repeatable for industrial cameras</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Hidden button for the onclick handler
        if st.button("Select LSC", key="drm-btn-lsc", type="primary" if not lsc_active else "secondary",
                     use_container_width=True, disabled=lsc_active,
                     help="Robust: stable with noise, matches caliper readings, best for industrial vision."):
            st.session_state["diameter_ref_method"] = "lsc"
            st.rerun()

    with card_right:
        strict_border = "2px solid #1f6feb" if strict_active else "2px solid #dde5ef"
        strict_bg = "#f6faff" if strict_active else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{strict_border};border-radius:12px;padding:20px 22px;background:{strict_bg};box-shadow:0 1px 3px rgba(15,23,42,0.04);cursor:pointer;min-height:260px;"
                 onclick="document.getElementById('drm-btn-strict').click();">
                <span class="drm-card-badge drm-badge-precision">🔬 PRECISION</span>
                <div class="drm-card-title">🎯 {strict_label}</div>
                <div class="drm-card-subtitle">{strict_subtitle} — envelope fit</div>
                <div class="drm-card-value">{format_value(strict_diameter)}</div>
                <div class="drm-card-unit">{strict_analogy} the part</div>
                <div class="drm-card-bullets">
                    <span>⚠️ <b>Single outlier</b> can change the result</span>
                    <span>⚠️ <b>Needs near-perfect edges</b> — sensitive to glare &amp; dust</span>
                    <span>💡 <b>Best for lab CMM</b> — not ideal for variable camera conditions</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Select Strict", key="drm-btn-strict", type="primary" if not strict_active else "secondary",
                     use_container_width=True, disabled=strict_active,
                     help="Strict: envelope fit (MCC/MIC). Sensitive to outliers; best for clean lab setups."):
            st.session_state["diameter_ref_method"] = "strict"
            st.rerun()

    if lsc_active:
        size_metric_label = "LSC fitted diameter"
        size_metric_value = lsc_diameter
        diameter_method = "lsc"
        selected_note = "🏭 LSC Fitted Diameter — Closest to pi-tape / caliper reading; most stable; what shop-floor inspectors recognize."
    else:
        size_metric_label = strict_label
        size_metric_value = strict_diameter
        diameter_method = "mcc_mic"
        selected_note = f"🎯 {strict_label} — Envelope fit. Smallest enclosing (outer) or largest inscribed (inner). Sensitive to outliers; best for lab CMM."

    st.markdown(
        f"""<div class="drm-selected-bar">Selected: {selected_note}</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Why this matters — real-world vs. lab measurement"):
        st.markdown(
            f"""
            **In an industrial vision system** with variable lighting, pipe surface finish, and camera noise, the
            **LSC fitted diameter** is the honest, repeatable choice. A few noisy edge pixels from glare, shadows,
            or slight blur barely move the LSC fit. It matches what a physical caliper or pi-tape would read.

            **MCC/MIC envelope fits** are extremal: a **single** outlier point (dust, reflection artifact, tiny
            chip on the rim) can inflate MCC or shrink MIC and give a wrong pass/fail result. They need a
            near-perfect contour — better suited for a **lab CMM** than a camera.

            The standards (EN 10217-7, ASTM A312, EN 10253-4) define diameter tolerance **limits** but do not
            mandate a specific measurement methodology. This tool lets us choose the approach that fits the
            inspection context.
            """
        )
    with st.expander("🔍 When to use each method"):
        compare_rows = f"""
        |  | 🏭 LSC Fitted (Robust) | 🎯 MCC/MIC (Strict) |
        |---|---|---|
        | **Measurement** | Best-fit average circle | Envelope (min enclosing / max inscribed) |
        | **Noise resistance** | ✅ High — outliers barely affect fit | ⚠️ Low — one bad pixel shifts result |
        | **Matches physical tools** | ✅ Caliper, pi-tape, micrometer | ❌ No physical equivalent |
        | **ISO GPS alignment** | LS circle per ISO 12181 | Envelope per ISO 14405-1 |
        | **Best for** | Shop-floor vision inspection | Lab CMM / metrology lab |
        | **Risk** | May accept a marginally oval part | May reject a good part with edge noise |
        | **Computed value** | `{format_value(lsc_diameter)}` | `{format_value(strict_diameter)}` |
        """
        st.markdown(compare_rows)

    # ── Ovality Input Method Selector ──
    mzc_roundness_val = roundness_mzc if np.isfinite(roundness_mzc) else 0.0
    mzc_roundness_pct = (
        f"{mzc_roundness_val / reference_diameter * 100.0:.2f}%"
        if np.isfinite(mzc_roundness_val) and np.isfinite(reference_diameter) and reference_diameter > 0
        else "not available"
    )

    if "ovality_ref_method" not in st.session_state:
        st.session_state["ovality_ref_method"] = "ostb"

    st.markdown("### ⭕ Ovality / Roundness Input Method")
    st.caption("Ovality in pipe standards is defined as the difference between maximum and minimum diameter. Choose whether to use the standard definition or the ISO Minimum Zone Circle concept.")

    oval_ostb_active = st.session_state["ovality_ref_method"] == "ostb"
    oval_mzc_active = st.session_state["ovality_ref_method"] == "mzc"

    oval_left, oval_right = st.columns(2, gap="medium")

    with oval_left:
        ostb_border = "2px solid #1f6feb" if oval_ostb_active else "2px solid #dde5ef"
        ostb_bg = "#f6faff" if oval_ostb_active else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{ostb_border};border-radius:12px;padding:20px 22px;background:{ostb_bg};box-shadow:0 1px 3px rgba(15,23,42,0.04);cursor:pointer;min-height:250px;"
                 onclick="document.getElementById('ovm-btn-ostb').click();">
                <span class="drm-card-badge drm-badge-recommended">✓ RECOMMENDED</span>
                <div class="drm-card-title">📐 OSTB Standard — Measured Diameter Range</div>
                <div class="drm-card-subtitle">Max diameter − Min diameter across real rim points</div>
                <div class="drm-card-value">{format_value(real_edge_diameter_range)}</div>
                <div class="drm-card-unit">D<sub>max</sub> − D<sub>min</sub> = {format_value(real_edge_max_diameter)} − {format_value(real_edge_min_diameter)}</div>
                <div class="drm-card-bullets">
                    <span>✅ <b>Matches Excel / standard definition</b> — exactly what the spec asks for</span>
                    <span>✅ <b>Direct from measured points</b> — no floating center, fixed to fitted axis</span>
                    <span>✅ <b>What calipers would read</b> — major vs. minor diameter</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Select OSTB Standard", key="ovm-btn-ostb",
                     type="primary" if not oval_ostb_active else "secondary",
                     use_container_width=True, disabled=oval_ostb_active,
                     help="Recommended: measured diameter range per the Excel/standard definition. Dmax − Dmin."):
            st.session_state["ovality_ref_method"] = "ostb"
            st.rerun()

    with oval_right:
        mzc_border = "2px solid #1f6feb" if oval_mzc_active else "2px solid #dde5ef"
        mzc_bg = "#f6faff" if oval_mzc_active else "#ffffff"
        st.markdown(
            f"""
            <div style="border:{mzc_border};border-radius:12px;padding:20px 22px;background:{mzc_bg};box-shadow:0 1px 3px rgba(15,23,42,0.04);cursor:pointer;min-height:250px;"
                 onclick="document.getElementById('ovm-btn-mzc').click();">
                <span class="drm-card-badge" style="background:#e8e0f0;color:#5b21b6;">🎓 ISO 12181</span>
                <div class="drm-card-title">🎯 MZC Roundness Error</div>
                <div class="drm-card-subtitle">Minimum Zone Circle — tightest 2-ring zone</div>
                <div class="drm-card-value">{format_value(mzc_roundness_val)}</div>
                <div class="drm-card-unit">Radial zone width ({mzc_roundness_pct} of diameter)</div>
                <div class="drm-card-bullets">
                    <span>💡 <b>ISO 12181 formal definition</b> — floating center minimizes zone</span>
                    <span>⚠️ <b>Not what the standard asks for</b> — MZC can under-report ovality</span>
                    <span>📚 <b>For academic comparison</b> — useful alongside the standard method</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Select MZC", key="ovm-btn-mzc",
                     type="primary" if not oval_mzc_active else "secondary",
                     use_container_width=True, disabled=oval_mzc_active,
                     help="ISO 12181 Minimum Zone Circle. Floating center minimizes radial zone; useful for academic comparison."):
            st.session_state["ovality_ref_method"] = "mzc"
            st.rerun()

    if oval_ostb_active:
        ovality_input_value = real_edge_diameter_range
        ovality_method = "ostb_measured_range"
        ovality_label = "Measured diameter range (OSTB standard)"
        ovality_note = "📐 OSTB Standard — Measured Diameter Range. Max − Min diameter across real rim points, per the Excel definition."
    else:
        ovality_input_value = mzc_roundness_val
        ovality_method = "mzc_roundness"
        ovality_label = "MZC roundness error (ISO 12181)"
        ovality_note = "🎯 MZC Roundness Error — ISO 12181 Minimum Zone. Floating-center radial zone width; for academic comparison."

    st.markdown(
        f"""<div class="drm-selected-bar">Selected: {ovality_note}</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 OSTB standard vs. MZC — why the difference matters"):
        st.markdown(
            f"""
            **Pipe standards (EN 10217-7, ASTM A312, EN 10253-4)** define ovality as the difference
            between the maximum and minimum measured diameter — essentially what a pair of calipers
            reads at different angles around the pipe. Calipers measure through the **pipe axis**,
            not a floating optimal center.

            **MZC (Minimum Zone Circle, ISO 12181)** finds the two concentric circles with the
            smallest radial gap that contain all points — but the center is allowed to **float**
            to minimize that gap. A perfectly oval (elliptical) pipe can have a very small MZC
            error even though calipers would measure a clear diameter difference.

            **Example:** An ellipse with 21.4 mm major and 19.9 mm minor axis has:
            - **Measured diameter range**: 1.5 mm (what the standard sees)
            - **MZC roundness error**: potentially much smaller (floating center hides the ovality)

            **Computed values:**
            - OSTB standard: `{format_value(real_edge_diameter_range)}` (Dmax {format_value(real_edge_max_diameter)} − Dmin {format_value(real_edge_min_diameter)})
            - MZC roundness: `{format_value(mzc_roundness_val)}`
            """
        )

    with st.expander("🔍 When to use each method"):
        st.markdown(
            f"""
            |  | 📐 OSTB Standard (Recommended) | 🎯 MZC Roundness (ISO) |
            |---|---|---|
            | **Definition** | D<sub>max</sub> − D<sub>min</sub> | Min. radial zone width |
            | **Center** | Fixed to fitted circle axis | Optimized (floating) |
            | **Matches Excel/standard** | ✅ Yes — exact spec definition | ❌ No — different concept |
            | **Matches caliper** | ✅ Yes | ❌ No |
            | **ISO 12181 alignment** | ❌ Not an ISO roundness metric | ✅ Formal ISO roundness |
            | **Best for** | Pass/fail per the standard | Academic / geometric analysis |
            | **Computed value** | `{format_value(real_edge_diameter_range)}` | `{format_value(mzc_roundness_val)}` |
            """
        )

    # ── OSTB evaluation inputs ──
    st.markdown("### OSTB evaluation inputs")
    normalized_diameter_range = (
        f"{real_edge_diameter_range / reference_diameter * 100.0:.2f}%"
        if np.isfinite(real_edge_diameter_range) and np.isfinite(reference_diameter) and reference_diameter > 0
        else "not available"
    )
    st.markdown(
        f"""
        <div class="ostb-method-chip-row">
            <span class="ostb-method-chip">{feature_heading}</span>
            <span class="ostb-method-chip">Diameter method: {diameter_method.upper()}</span>
            <span class="ostb-method-chip">OSTB size input: {size_metric_label}</span>
            <span class="ostb-method-chip">Ovality input: {ovality_label}</span>
            <span class="ostb-method-chip">Checked when selected standard maps ovality</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    value_cols = st.columns(4, gap="medium")
    render_value_card(value_cols[0], "OSTB size input", format_value(size_metric_value))
    render_value_card(
        value_cols[1],
        "OSTB ovality check input",
        format_value(ovality_input_value),
        f"{ovality_label} | {ovality_input_value / reference_diameter * 100.0:.2f}% of dia" if np.isfinite(ovality_input_value) and np.isfinite(reference_diameter) and reference_diameter > 0 else ovality_label,
    )
    render_value_card(value_cols[2], "Measured min diameter", format_value(real_edge_min_diameter))
    render_value_card(value_cols[3], "Measured max diameter", format_value(real_edge_max_diameter))
    eval_left, eval_right = st.columns(2, gap="large")
    with eval_left:
        st.markdown(f"#### {feature_heading}")
        st.markdown(
            f"""
            - **Diameter method**: `{diameter_method.upper()}` ({size_metric_label})
            - **OSTB size input**: `{format_value(size_metric_value)}`
            - **Ovality method**: `{ovality_method.upper()}` ({ovality_label})
            - **OSTB ovality check input**: `{format_value(ovality_input_value)}`
            - **Measured diameter range**: `{format_value(real_edge_diameter_range)}`
            - **Measured min/max diameter**: `{format_value(real_edge_min_diameter)}` / `{format_value(real_edge_max_diameter)}`
            - **MZC roundness error**: `{format_value(mzc_roundness_val)}`
            """
        )
    with eval_right:
        if is_outer:
            st.markdown(
                f"""
                - **OSTB applied flow**
                - Use `{format_value(size_metric_value)}` from **{size_metric_label}** for size/fit comparison.
                - Use `{format_value(ovality_input_value)}` from **{ovality_label}** as the ovality input when the selected standard has an ovality limit.
                - ISO display values still shown for review: `LSC {format_value(roundness_lsc)}` and `MCC enclosing deviation {format_value(roundness_mcc)}`.
                - LSC diameter available: `{format_value(lsc_diameter)}` | MCC diameter: `{format_value(mcc_diameter)}`.
                - MZC roundness: `{format_value(mzc_roundness_val)}` | Measured range: `{format_value(real_edge_diameter_range)}`.
                """
            )
        else:
            st.markdown(
                f"""
                - **OSTB applied flow**
                - Use `{format_value(size_metric_value)}` from **{size_metric_label}** for size/clearance comparison.
                - Use `{format_value(ovality_input_value)}` from **{ovality_label}** as the ovality input when the selected standard has an ovality limit.
                - ISO display values still shown for review: `LSC {format_value(roundness_lsc)}` and `MCC enclosing deviation {format_value(roundness_mcc)}`.
                - LSC diameter available: `{format_value(lsc_diameter)}` | MIC diameter: `{format_value(mic_diameter)}`.
                - MZC roundness: `{format_value(mzc_roundness_val)}` | Measured range: `{format_value(real_edge_diameter_range)}`.
                """
            )

    with st.expander("How to interpret these numbers"):
        st.markdown(
            f"""
            - **LSC roundness error** gives a stable, repeatable screening value for vision data.
            - **MZC roundness error** is closest to the strict ISO roundness-error concept in this app.
            - **Measured diameter range** is `real-edge max diameter - real-edge min diameter`; the OSTB ovality check uses this value because ovality is usually defined as major diameter minus minor diameter.
            - **MCC enclosing deviation** helps answer: *What is the smallest tube that would fit over this part?* A large value often indicates local outward bumps.
            - **MIC clearance diameter** helps answer: *What is the largest rod that would fit inside?* It is a clearance value, not a roundness error.
            - **Diameter method** you selected above determines which value is compared against the tolerance limits. LSC is recommended for camera-based inspection.

            The ISO method values above are for geometric interpretation. The OSTB section below uses the selected OSTB inputs for pass/fail. All values are shown in **{unit}**.
            """
        )

    return {
        "method": "lsc_mzc_mcc_mic",
        "standard_reference": "ISO 12181 concept",
        "roundness_px": measurement.roundness_px,
        "roundness_value": value,
        "lsc_roundness_px": roundness_methods["lsc_roundness_px"],
        "mzc_roundness_px": roundness_methods["mzc_roundness_px"],
        "mcc_roundness_px": roundness_methods["mcc_roundness_px"],
        "mic_diameter_px": roundness_methods["mic_diameter_px"],
        "lsc_roundness": roundness_lsc,
        "mzc_roundness": roundness_mzc,
        "mcc_roundness": roundness_mcc,
        "mcc_diameter": mcc_diameter,
        "lsc_diameter": lsc_diameter,
        "lsc_roundness_error": roundness_lsc,
        "mzc_roundness_error": roundness_mzc,
        "real_edge_min_diameter": real_edge_min_diameter,
        "real_edge_max_diameter": real_edge_max_diameter,
        "real_edge_diameter_range": real_edge_diameter_range,
        "ostb_ovality_input": ovality_input_value,
        "ovality_method": ovality_method,
        "ovality_label": ovality_label,
        "mcc_enclosing_deviation": roundness_mcc,
        "mic_diameter": mic_diameter,
        "feature_type": feature_type,
        "diameter_method": diameter_method,
        "size_metric_label": size_metric_label,
        "size_metric_value": size_metric_value,
        "unit": unit,
    }


def render_chart_explanation(chart_name: str, unit: str) -> None:
    if chart_name == "polar":
        with st.expander("How to read radial deviation"):
            st.markdown(
                f"""
                - The dashed circle is the **ideal fitted circle**.
                - Blue areas show points outside the fitted circle.
                - Orange areas show points inside the fitted circle.
                - The radial scale is in **{unit}**.
                - A smaller, more even shape means the rim is closer to round.
                - Large peaks in either direction mean that part of the rim is farther from the fitted circle.
                """
            )
    elif chart_name == "histogram":
        with st.expander("How to read signed radial deviation distribution"):
            st.markdown(
                f"""
                - This chart counts how many rim points have each deviation value.
                - The center line at **0 {unit}** means the ideal fitted circle.
                - Values to the right are points outside the fitted circle.
                - Values to the left are points inside the fitted circle.
                - A narrow group close to zero means a more consistent circular rim.
                - A wide spread means more roundness variation or edge noise.
                """
            )
    elif chart_name == "profile":
        with st.expander("How to read deviation by angle"):
            st.markdown(
                f"""
                - This unwraps the rim into a line from **0 to 360 degrees**.
                - The horizontal zero line is the ideal fitted circle.
                - Peaks above zero are outward deviations.
                - Dips below zero are inward deviations.
                - Repeating waves can indicate ovality or shape distortion.
                - Sharp spikes can indicate edge noise, glare, scratches, or a local defect.
                """
            )


def render_pipe_wall_summary(wall_detection: Optional[PipeWallDetection], unit: str, mm_per_pixel: float) -> None:
    """Minimal summary shown inside visual analysis. Full detail is in render_wall_thickness_review."""
    if wall_detection is None:
        return
    scale = mm_per_pixel if unit == "mm" else 1.0
    st.caption(
        f"Inner Ø {wall_detection.inner_radius_px * 2.0 * scale:.2f} {unit} | "
        f"Outer Ø {wall_detection.outer_radius_px * 2.0 * scale:.2f} {unit} | "
        f"Wall {wall_detection.wall_thickness_px * scale:.2f} {unit} "
        f"(min {wall_detection.wall_thickness_min_px * scale:.2f}, max {wall_detection.wall_thickness_max_px * scale:.2f})"
    )


def render_wall_thickness_review(wall_detection: Optional[PipeWallDetection], unit: str, mm_per_pixel: float) -> None:
    st.markdown("## 🔬 Wall / Thickness Review")

    if wall_detection is None:
        st.warning("Inner/outer wall pair was not detected clearly enough for thickness review.")
        with st.expander("📖 How wall detection works"):
            st.markdown(
                """
                The wall detector scans **720 radial rays** outward from the fitted inner circle center.
                Along each ray it looks for a **dark→bright transition** (from pipe interior to wall material)
                at the inner edge and a **bright→dark transition** (from wall to background) at the outer edge.

                The distance between these two edges along each ray is the **per-ray wall thickness**.

                **Why it might fail:**
                - Low contrast between pipe and background
                - Glare or reflections washing out the edge
                - Very thin walls (< 12 px) below the detection threshold
                - Pipe surface texture confusing the gradient detector
                """
            )
        return

    scale = mm_per_pixel if unit == "mm" else 1.0

    # ── Derived values ──
    inner_dia = wall_detection.inner_radius_px * 2.0 * scale
    outer_dia = wall_detection.outer_radius_px * 2.0 * scale
    wall_from_dia = (outer_dia - inner_dia) / 2.0
    center_offset = wall_detection.center_offset_px * scale
    wall_min = wall_detection.wall_thickness_min_px * scale
    wall_median = wall_detection.wall_thickness_px * scale
    wall_max = wall_detection.wall_thickness_max_px * scale
    wall_range = wall_max - wall_min
    wall_range_pct = (wall_range / wall_median * 100.0) if wall_median > 0 else 0.0

    # ── How it works expander ──
    with st.expander("📖 How wall detection works — radial ray scanning", expanded=False):
        st.markdown(
            f"""
            ### The radial ray-scan method

            The detector fires **720 rays** outward from the fitted inner-circle center, one every 0.5°:

            1. For each angle θ, it samples pixel intensities along the ray from `inner_radius × 1.08` out to `inner_radius × 2.4`
            2. It looks for the **strongest dark→bright gradient** — this is the **outer wall edge** (pipe material → background)
            3. If outer edge is found on ≥ 40 rays, the outer points are circle-fitted to get **outer center & radius**
            4. **Wall thickness per ray** = `outer_radius_at_angle − inner_radius`
            5. **Wall median** = median of all per-ray thicknesses (robust against outliers)

            **Detection:**
            - Inner center: `({wall_detection.inner_center_x_px:.1f}, {wall_detection.inner_center_y_px:.1f}) px`
            - Outer center: `({wall_detection.outer_center_x_px:.1f}, {wall_detection.outer_center_y_px:.1f}) px`
            - Center offset: `{center_offset:.3f} {unit}` (inner vs outer center misalignment)
            - Detected via: **{wall_detection.method}**
            """
        )

    # ── Fitted Diameters ──
    st.markdown("### 📐 Fitted Diameters & Wall")
    st.caption("Circle-fitted values from detected inner and outer rim points.")

    dia_css = """
    <style>
    .wall-formula-card {
        padding: 16px 20px;
        border: 1px solid #dde5ef;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        min-height: 140px;
    }
    .wall-formula-card .wf-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.15;
    }
    .wall-formula-card .wf-unit {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7c93;
    }
    .wall-formula-card .wf-formula {
        margin-top: 8px;
        font-size: 0.78rem;
        color: #52616f;
        font-family: 'SF Mono', 'Cascadia Code', monospace;
        background: #f7f9fc;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .wall-formula-card .wf-label {
        font-size: 0.88rem;
        font-weight: 650;
        color: #334155;
        margin-bottom: 4px;
    }
    </style>
    """
    st.markdown(dia_css, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4, gap="medium")
    with d1:
        st.markdown(
            f"""
            <div class="wall-formula-card">
                <div class="wf-label">🔵 Inner Fitted Diameter</div>
                <div class="wf-value">{inner_dia:.3f}</div>
                <div class="wf-unit">{unit}</div>
                <div class="wf-formula">2 × inner_radius</div>
                <div style="margin-top:6px;font-size:0.76rem;color:#6b7c93;">Fitted to inner edge points</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div class="wall-formula-card">
                <div class="wf-label">🟢 Outer Fitted Diameter</div>
                <div class="wf-value">{outer_dia:.3f}</div>
                <div class="wf-unit">{unit}</div>
                <div class="wf-formula">2 × outer_radius</div>
                <div style="margin-top:6px;font-size:0.76rem;color:#6b7c93;">Fitted to outer edge points</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            f"""
            <div class="wall-formula-card">
                <div class="wf-label">🟠 Wall (from diameters)</div>
                <div class="wf-value">{wall_from_dia:.3f}</div>
                <div class="wf-unit">{unit}</div>
                <div class="wf-formula">(OD − ID) ÷ 2</div>
                <div style="margin-top:6px;font-size:0.76rem;color:#6b7c93;">Average wall from fitted circles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d4:
        offset_color = "#0d7d33" if center_offset < max(0.01, outer_dia * 0.005) else "#b45f06"
        st.markdown(
            f"""
            <div class="wall-formula-card">
                <div class="wf-label">📍 Center Offset</div>
                <div class="wf-value" style="color:{offset_color};">{center_offset:.3f}</div>
                <div class="wf-unit">{unit}</div>
                <div class="wf-formula">‖ inner_center − outer_center ‖</div>
                <div style="margin-top:6px;font-size:0.76rem;color:#6b7c93;">{'✅ Concentric' if center_offset < max(0.01, outer_dia * 0.005) else '⚠️ Eccentric — wall may vary'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Per-Ray Wall Variation ──
    st.markdown("### 📏 Per-Ray Wall Variation")
    st.caption("Wall thickness measured individually along each of the 720 radial rays — not from fitted circles.")

    # Range bar visualization
    range_pct_fill = min(100.0, wall_range_pct)
    range_color = "#0d7d33" if wall_range_pct < 10 else ("#d97706" if wall_range_pct < 25 else "#dc2626")

    st.markdown(
        f"""
        <div style="margin:12px 0 16px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.82rem;color:#334155;">
                <span>Min: <b>{wall_min:.3f} {unit}</b></span>
                <span>Median: <b>{wall_median:.3f} {unit}</b></span>
                <span>Max: <b>{wall_max:.3f} {unit}</b></span>
            </div>
            <div style="height:18px;border-radius:9px;background:#e5e7eb;position:relative;overflow:hidden;">
                <div style="position:absolute;left:0;top:0;height:100%;border-radius:9px;
                     background:linear-gradient(90deg,#16a34a,#d97706,#dc2626);width:100%;opacity:0.35;"></div>
                <div style="position:absolute;left:{max(0.0, (wall_median - wall_min) / max(wall_range, 0.001) * 100.0 - 1.5):.1f}%;top:2px;
                     width:3px;height:14px;background:#1f6feb;border-radius:2px;" title="Median {wall_median:.3f}"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:0.76rem;color:#6b7c93;">
                <span>Range: {wall_range:.3f} {unit}</span>
                <span style="color:{range_color};font-weight:650;">{wall_range_pct:.1f}% of median</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    var1, var2, var3, var4 = st.columns(4, gap="medium")
    var1.metric("Wall min", f"{wall_min:.3f} {unit}", help="Smallest per-ray wall thickness — thinnest spot.")
    var2.metric("Wall median", f"{wall_median:.3f} {unit}", help="Median of all 720 per-ray measurements. Robust against outliers.")
    var3.metric("Wall max", f"{wall_max:.3f} {unit}", help="Largest per-ray wall thickness — thickest spot.")
    var4.metric("Wall range", f"{wall_range:.3f} {unit}", delta=f"{wall_range_pct:.1f}% of median", help="Max − Min. Lower is more uniform.")

    # ── Inner & Outer Rim Diameter Range ──
    st.markdown("### 📐 Rim Diameter Range (Dmax − Dmin)")
    st.caption("Per-point diameter = 2 × distance from each edge point to its fitted circle center. Dmax and Dmin are the max/min of those per-point diameters.")

    # Compute inner rim diameter stats
    inner_pts = wall_detection.inner_points.astype(np.float64)
    inner_ctr = np.array([wall_detection.inner_center_x_px, wall_detection.inner_center_y_px], dtype=np.float64)
    inner_diameters = 2.0 * np.linalg.norm(inner_pts - inner_ctr, axis=1)
    inner_dmax = float(np.max(inner_diameters)) * scale
    inner_dmin = float(np.min(inner_diameters)) * scale
    inner_drange = inner_dmax - inner_dmin
    inner_drange_pct = (inner_drange / (inner_dia) * 100.0) if inner_dia > 0 else 0.0

    # Compute outer rim diameter stats
    outer_pts = wall_detection.outer_points.astype(np.float64)
    outer_ctr = np.array([wall_detection.outer_center_x_px, wall_detection.outer_center_y_px], dtype=np.float64)
    outer_diameters = 2.0 * np.linalg.norm(outer_pts - outer_ctr, axis=1)
    outer_dmax = float(np.max(outer_diameters)) * scale
    outer_dmin = float(np.min(outer_diameters)) * scale
    outer_drange = outer_dmax - outer_dmin
    outer_drange_pct = (outer_drange / outer_dia * 100.0) if outer_dia > 0 else 0.0

    rim_range_css = """
    <style>
    .rim-range-card {
        padding: 16px 20px;
        border: 1px solid #dde5ef;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        margin-bottom: 8px;
    }
    .rim-range-card .rr-label {
        font-size: 0.88rem;
        font-weight: 650;
        color: #334155;
        margin-bottom: 8px;
    }
    .rim-range-card .rr-formula {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        font-family: 'SF Mono', 'Cascadia Code', monospace;
    }
    .rim-range-card .rr-values {
        font-size: 0.82rem;
        color: #52616f;
        margin-top: 4px;
    }
    </style>
    """
    st.markdown(rim_range_css, unsafe_allow_html=True)

    rim_left, rim_right = st.columns(2, gap="medium")
    with rim_left:
        inner_color = "#0d7d33" if inner_drange_pct < 3 else ("#d97706" if inner_drange_pct < 8 else "#dc2626")
        st.markdown(
            f"""
            <div class="rim-range-card">
                <div class="rr-label">🔵 Inner Rim Diameter Range</div>
                <div class="rr-formula">D<sub>max</sub> − D<sub>min</sub> = {inner_dmax:.4f} − {inner_dmin:.4f}</div>
                <div class="rr-values">
                    Range: <b style="color:{inner_color};">{inner_drange:.4f} {unit}</b> 
                    ({inner_drange_pct:.2f}% of inner Ø)
                </div>
                <div style="margin-top:8px;font-size:0.75rem;color:#6b7c93;">
                    Formula: 2 × ‖point<sub>i</sub> − inner_center‖ for each of {len(inner_pts)} inner rim points
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rim_right:
        outer_color = "#0d7d33" if outer_drange_pct < 3 else ("#d97706" if outer_drange_pct < 8 else "#dc2626")
        st.markdown(
            f"""
            <div class="rim-range-card">
                <div class="rr-label">🟢 Outer Rim Diameter Range</div>
                <div class="rr-formula">D<sub>max</sub> − D<sub>min</sub> = {outer_dmax:.4f} − {outer_dmin:.4f}</div>
                <div class="rr-values">
                    Range: <b style="color:{outer_color};">{outer_drange:.4f} {unit}</b> 
                    ({outer_drange_pct:.2f}% of outer Ø)
                </div>
                <div style="margin-top:8px;font-size:0.75rem;color:#6b7c93;">
                    Formula: 2 × ‖point<sub>i</sub> − outer_center‖ for each of {len(outer_pts)} outer rim points
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📖 How Dmax and Dmin are derived", expanded=False):
        st.latex(r"D_i = 2 \cdot \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2}")
        st.caption("Per-point diameter: twice the Euclidean distance from each rim point to its fitted circle center.")
        st.markdown(
            """
            | Symbol | Meaning |
            |---|---|
            | **Dᵢ** | Diameter passing through the i-th rim point |
            | **xᵢ, yᵢ** | Pixel coordinates of the i-th detected rim point |
            | **x_c, y_c** | Pixel coordinates of the fitted circle center |
            | **√((xᵢ−x_c)² + (yᵢ−y_c)²)** | Radial distance from center to point = radius at that angle |
            """
        )
        st.markdown(
            f"""
            Then:
            - **Dmax** = max(D₁, D₂, …, Dₙ) — the largest per-point diameter
            - **Dmin** = min(D₁, D₂, …, Dₙ) — the smallest per-point diameter  
            - **Diameter range** = Dmax − Dmin (**ovality input** for the OSTB check)

            **Inner rim:** {len(inner_pts)} points, fitted center at ({wall_detection.inner_center_x_px:.1f}, {wall_detection.inner_center_y_px:.1f}) px  
            **Outer rim:** {len(outer_pts)} points, fitted center at ({wall_detection.outer_center_x_px:.1f}, {wall_detection.outer_center_y_px:.1f}) px

            The OSTB ovality check uses the range from the **active measurement target** 
            (inner rim if "Inner/opening rim" is selected, outer rim if "Outer pipe diameter" is selected).
            """
        )

    # ── Wall & OSTB Standards ──
    st.markdown("### 🔗 Wall & OSTB Standards")
    with st.expander("How wall thickness affects standards evaluation", expanded=False):
        st.markdown(
            f"""
            ### Wall thickness in the OSTB tolerance flow

            For most standards, wall thickness is **informational only** — it does not directly affect
            the diameter tolerance or ovality pass/fail. However, it matters for:

            **ASTM thin-wall branch (Standards 3 & 3.1):**
            When the ratio **t/D < 3%** (wall thickness / outer diameter), ASTM A999 / A1016 applies a
            **different ovality rule** because thin-wall pipes are more flexible and allowed more ovality.

            | Condition | Ovality limit |
            |---|---|
            | t/D > 3% (thick wall) | Fixed value from lookup table |
            | t/D ≤ 3% (thin wall) | 1.5% or 2.0% of nominal OD |

            **Computed values:**
            - Wall thickness: `{wall_median:.3f} {unit}`
            - If outer diameter ≈ `{outer_dia:.3f} {unit}`, then t/D = `{wall_median / outer_dia * 100.0:.2f}%`
            - This branch selection happens automatically when the OSTB standards check is run.

            **For all other standards (EN 10217-7, EN 10253-4, etc.):**
            Wall thickness is displayed for review but does not change the pass/fail calculation.
            """
        )

    with st.expander("❓ FAQ — Wall thickness accuracy", expanded=False):
        st.markdown(
            f"""
            **Q: Why are "wall from diameters" and "wall median" different?**
            A: "Wall from diameters" is `(OD − ID) / 2` using the **fitted circle** diameters. "Wall median"
            is the median of 720 **per-ray** measurements. They differ when the inner and outer circles
            aren't perfectly concentric or when the wall isn't perfectly uniform.

            **Q: Which value should I trust?**
            A: **Wall median** is more honest — it captures real variation around the circumference.
            "Wall from diameters" is a quick single-number average.

            **Q: What center offset is acceptable?**
            A: Ideally < 0.5% of outer diameter (here: `{center_offset / outer_dia * 100.0:.2f}%`). 
            Larger offsets mean the inner and outer circles are misaligned, which can indicate:
            - The pipe is genuinely eccentric
            - Camera was not perpendicular to the pipe face
            - Edge detection found a false rim on one side

            **Q: Why is the wall chart not perfectly round?**
            A: Real pipes have wall thickness variation. Manufacturing processes (extrusion, welding,
            bending) create natural variation. The chart helps you see if the variation is
            **systematic** (e.g., always thicker on one side → eccentricity) or **random** (normal
            manufacturing tolerance).
            """
        )


def render_visual_analysis(
    processed: Dict[str, np.ndarray],
    measurement: PipeMeasurement,
    tolerance_report: Optional[Dict[str, object]],
    unit: str,
    wall_reference_measurement: Optional[PipeMeasurement] = None,
) -> None:
    st.subheader("Visual Analysis")
    st.markdown(
        '<div class="visual-note">Review the fitted rim, ROI (Region of Interest) crop, edge extraction, inner/outer wall guide, and radial deviation from multiple views.</div>',
        unsafe_allow_html=True,
    )

    wall_reference = wall_reference_measurement or measurement
    wall_detection = detect_pipe_wall_rims(processed, wall_reference)
    full_overlay = draw_outer_circle_overlay(
        processed["full_image_bgr"],
        measurement,
        processed,
        tolerance_report,
        unit=unit,
        wall_detection=wall_detection,
    )
    wall_overlay = draw_pipe_wall_overlay(processed["full_image_bgr"], wall_reference, wall_detection, processed)
    # Show both inner & outer deviation on the heat overlay when wall detection is available
    inner_for_heat = wall_reference_measurement if (wall_detection is not None and wall_reference_measurement is not None and measurement is not wall_reference_measurement) else None
    heat_overlay = draw_deviation_heat_overlay(processed["full_image_bgr"], measurement, processed, second_measurement=inner_for_heat)
    zoom_bbox = measurement_zoom_bbox(processed["full_image_bgr"], measurement)
    zoom_overlay = crop_bgr_to_bbox(full_overlay, zoom_bbox, padding_fraction=0.04)
    zoom_wall = crop_bgr_to_bbox(wall_overlay, zoom_bbox, padding_fraction=0.04)
    zoom_heat = crop_bgr_to_bbox(heat_overlay, zoom_bbox, padding_fraction=0.04)

    overview_col, detail_col = st.columns([1.45, 1.0], gap="large")
    with overview_col:
        st.markdown("#### Full measurement overlay")
        st.image(bgr_to_rgb(full_overlay), width="stretch")
    with detail_col:
        st.markdown("#### Zoomed rim detail")
        st.image(bgr_to_rgb(zoom_overlay), width="stretch")
        st.markdown("#### Inner / outer rim guide")
        st.image(bgr_to_rgb(zoom_wall), width="stretch")
        st.markdown("#### Signed deviation heat overlay")
        st.image(bgr_to_rgb(zoom_heat), width="stretch")

    render_pipe_wall_summary(wall_detection, unit, measurement.mm_per_pixel)

    diag_a, diag_b, diag_c = st.columns(3, gap="large")
    with diag_a:
        st.markdown("#### Undistorted ROI (Region of Interest)")
        st.image(bgr_to_rgb(processed["image_bgr"]), width="stretch")
    with diag_b:
        st.markdown("#### Grayscale ROI (Region of Interest)")
        st.image(processed["gray"], width="stretch", clamp=True)
    with diag_c:
        st.markdown("#### Edge map")
        st.image(processed["clean_edges"], width="stretch", clamp=True)

    chart_left, chart_right = st.columns([1.05, 1.0], gap="large")
    with chart_left:
        st.plotly_chart(build_polar_figure(measurement, unit=unit), use_container_width=True)
        render_chart_explanation("polar", unit)
    with chart_right:
        st.plotly_chart(build_deviation_histogram(measurement, unit=unit), use_container_width=True)
        render_chart_explanation("histogram", unit)
    st.plotly_chart(build_deviation_profile_figure(measurement, unit=unit), use_container_width=True)
    render_chart_explanation("profile", unit)


def main() -> None:
    page_icon = str(NOVIA_LOGO_PATH) if NOVIA_LOGO_PATH.exists() else None
    st.set_page_config(page_title="Pipe Roundness Inspector", page_icon=page_icon, layout="wide")
    inject_app_css()
    render_page_header()

    calibration = load_static_calibration()
    saved_scale = load_saved_scale_config()
    try:
        standards_contract, _execution_contract = load_standards_contract()
    except Exception as exc:
        standards_contract = None
        _execution_contract = None
        st.error(f"Could not load OSTB standards data: {exc}")
    try:
        ovality_lookup = load_ovality_lookup()
    except Exception as exc:
        ovality_lookup = {}
        st.error(f"Could not load ovality lookup data: {exc}")
    try:
        diameter_lookup = load_diameter_lookup()
    except Exception as exc:
        diameter_lookup = {}
        st.error(f"Could not load diameter lookup data: {exc}")

    with st.sidebar:
        render_sidebar_brand()
        render_system_status(calibration, standards_contract, saved_scale)
        st.header("Upload Test Image")
        test_file = st.file_uploader("Test pipe image", type=["bmp", "png", "jpg", "jpeg"], key="test")

        st.header("Processing Settings")
        blur_kernel = 5
        canny_sigma = 0.33
        morphology_kernel = 3
        st.caption("Production defaults are locked for repeatable inspection.")
        show_advanced_settings = st.toggle(
            "Show advanced calibration controls",
            value=False,
            help="Only enable this when recalibrating the image pipeline or debugging unusual images.",
        )
        if show_advanced_settings:
            st.warning("Changing these values changes the measurement pipeline. Keep them fixed during production inspection.")
            with st.expander("Image processing parameters", expanded=True):
                blur_kernel = st.slider(
                    "Gaussian blur kernel",
                    3,
                    15,
                    blur_kernel,
                    step=2,
                    help="Smooths image noise before edge detection. Increase if the edge map has many tiny noisy edges. Decrease if the pipe rim becomes too soft or loses detail.",
                )
                st.caption("Higher = smoother image and less noise. Lower = sharper rim but more noise.")
                canny_sigma = st.slider(
                    "Adaptive Canny sigma",
                    0.10,
                    0.80,
                    canny_sigma,
                    step=0.01,
                    help="Controls edge detector sensitivity. Increase if the rim edge is broken or missing. Decrease if too many background or reflection edges appear.",
                )
                st.caption("Higher = more sensitive edge detection. Lower = stricter edge detection.")
                morphology_kernel = st.slider(
                    "Morphology kernel",
                    1,
                    9,
                    morphology_kernel,
                    step=2,
                    help="Connects small gaps in detected edges. Increase if the rim contour is fragmented. Decrease if separate edges merge together or the rim looks distorted.",
                )
                st.caption("Higher = connects larger gaps. Lower = preserves original edge shape.")
        else:
            st.caption("Using fixed defaults: blur 5, Canny sigma 0.33, morphology 3.")
        measurement_target = st.radio(
            "Measurement target",
            ["Inner/opening rim", "Outer pipe diameter"],
            index=0,
            help=(
                "Choose explicitly whether you want the inner/opening rim or the outer pipe diameter. "
                "Outer pipe diameter uses the wall guide and falls back if that edge is not clear."
            ),
        )
        st.caption("Pixel inspection runs automatically after upload.")
        render_sidebar_footer()

    st.info(
        "Upload one pipe image to get an immediate pixel/relative roundness result. "
        "Millimeter values and OSTB standards checks are enabled after you add a scale."
    )

    if test_file is None:
        st.write("Upload a test image to start inspection.")
        return

    preprocess_config = PreprocessConfig(
        blur_kernel=blur_kernel,
        canny_sigma=canny_sigma,
        morphology_kernel=morphology_kernel,
    )

    try:
        test_bytes = test_file.getvalue()
        with st.spinner("Detecting outer rim and fitting circle..."):
            raw_test, test_processed, test_note = measure_pipe_roundness_pixels_robust(
                test_file.name,
                test_bytes,
                calibration,
                preprocess_config,
            )
    except Exception as exc:
        st.error(f"Inspection failed: {exc}")
        return

    inner_reference_measurement = raw_test
    target_warning = None
    wall_target_detection = detect_pipe_wall_rims(test_processed, inner_reference_measurement)
    resolved_feature_type = "inner"
    if measurement_target == "Inner/opening rim":
        if wall_target_detection is not None:
            raw_test = build_inner_wall_measurement(inner_reference_measurement, wall_target_detection)
            test_note = f"{test_note}; target inner/opening rim from wall guide"
        else:
            test_note = f"{test_note}; target inner/opening rim"
        resolved_feature_type = "inner"
    elif measurement_target == "Outer pipe diameter":
        if wall_target_detection is not None:
            raw_test = build_outer_wall_measurement(inner_reference_measurement, wall_target_detection)
            test_note = f"{test_note}; target outer pipe diameter from wall guide"
            resolved_feature_type = "outer"
        else:
            target_warning = "Outer pipe diameter was requested, but the outer wall edge was not found clearly. Using inner/opening rim result."
            test_note = f"{test_note}; requested outer pipe diameter but outer wall was not found; using inner/opening rim"
            resolved_feature_type = "inner"

    st.subheader("Pixel Inspection")
    if target_warning is not None:
        st.warning(target_warning)
    st.caption(f"Active rim for ISO values and OSTB inputs: {'outer pipe diameter' if resolved_feature_type == 'outer' else 'inner/opening rim'}")
    render_pixel_metric_row(raw_test)
    render_measurement_quality(raw_test, test_note, "pixel_only")
    render_pixel_interpretation(raw_test)

    st.subheader("Scale To Millimeters")
    scale_options = ["Pixel only"]
    if saved_scale is not None:
        scale_options.append("Use saved calibrated scale")
    scale_options.append("Calibrate from this image")
    scale_mode = st.radio(
        "Scale mode",
        scale_options,
        horizontal=True,
    )
    if saved_scale is None:
        st.caption("No saved calibrated scale yet. Calibrate from this image once, then save it for later inspections.")

    scale_source = "pixel_only"
    mm_per_pixel = None
    calibration_details: Optional[Dict[str, object]] = None

    if scale_mode == "Use saved calibrated scale" and saved_scale is not None:
        mm_per_pixel = float(saved_scale["mm_per_pixel"])
        scale_source = "saved_calibrated_scale"
        calibration_details = saved_scale
        st.caption(
            "Saved scale is valid only when camera distance, zoom, resolution, angle, and pipe plane are unchanged."
        )
    elif scale_mode == "Calibrate from this image":
        outer_reference_diameter_px = (
            wall_target_detection.outer_radius_px * 2.0
            if wall_target_detection is not None
            else float("nan")
        )
        cal_col_a, cal_col_b = st.columns([1.0, 1.0])
        with cal_col_a:
            known_test_diameter_mm = st.number_input(
                "Known real outer diameter of this test pipe (mm)",
                min_value=0.001,
                value=60.5,
                step=0.1,
            )
        with cal_col_b:
            if np.isfinite(outer_reference_diameter_px) and outer_reference_diameter_px > 0:
                st.metric("Detected outer diameter", f"{outer_reference_diameter_px:.2f} px")
            else:
                st.metric("Detected outer diameter", "not available")
        if not np.isfinite(outer_reference_diameter_px) or outer_reference_diameter_px <= 0:
            st.error("Outer wall diameter was not detected clearly enough for outer-diameter calibration.")
        else:
            mm_per_pixel = compute_mm_per_pixel_from_known_diameter(outer_reference_diameter_px, known_test_diameter_mm)
        scale_source = "current_image_known_outer_diameter"
        calibration_details = {
            "mm_per_pixel": mm_per_pixel,
            "source": scale_source,
            "reference_diameter_mm": known_test_diameter_mm,
            "reference_diameter_px": outer_reference_diameter_px,
            "reference_filename": test_file.name,
            "measurement_target": "Outer pipe diameter calibration reference",
        }
        save_label = "Update saved calibrated scale" if saved_scale is not None else "Save calibrated scale"
        if st.button(save_label, type="secondary", use_container_width=True, disabled=mm_per_pixel is None):
            save_scale_config(
                mm_per_pixel,
                scale_source,
                known_test_diameter_mm,
                outer_reference_diameter_px,
                test_file.name,
                "Outer pipe diameter calibration reference",
            )
            st.success(f"Saved calibrated scale to {SAVED_SCALE_PATH.name}.")
            st.rerun()
    if mm_per_pixel is not None:
        test_measurement = convert_measurement_scale(raw_test, mm_per_pixel)
        st.success(f"Scale active: {mm_per_pixel:.6f} mm/px")
        render_metric_row(test_measurement)
    else:
        test_measurement = raw_test
        st.info("No scale active. Showing pixel-only measurement; tolerance check requires millimeters.")

    tolerance_report = None
    standard_label = None
    unit = "mm" if mm_per_pixel is not None else "px"
    measured_wall_thickness_mm = (
        wall_target_detection.wall_thickness_px * mm_per_pixel
        if mm_per_pixel is not None and wall_target_detection is not None
        else None
    )

    if mm_per_pixel is not None:
        render_wall_thickness_review(wall_target_detection, unit, mm_per_pixel)
        roundness_summary = render_roundness_evaluation(test_measurement, unit, resolved_feature_type)
        st.subheader("OSTB Standards")
        compliance_diameter_mm = float(roundness_summary["size_metric_value"]) if roundness_summary is not None else float("nan")
        compliance_diameter_label = str(roundness_summary["size_metric_label"]) if roundness_summary is not None else "Fitted diameter"
        measured_ovality_mm = float(roundness_summary["ostb_ovality_input"]) if roundness_summary is not None else float("nan")
        if standards_contract is None:
            st.error("The OSTB standards data package is missing or invalid, so tolerance comparison is unavailable.")
        elif not np.isfinite(compliance_diameter_mm) or compliance_diameter_mm <= 0:
            st.error(f"{compliance_diameter_label} is not available, so standards compliance cannot be checked.")
        else:
            with st.expander("🛡️ OSTB Standards Compliance", expanded=True):
                st.caption("Match the measurement against mapped industry tolerances.")
                try:
                    combined_report = render_contract_tolerance_inputs(
                        test_measurement,
                        standards_contract,
                        suggested_nominal_mm=compliance_diameter_mm,
                        feature_type=resolved_feature_type,
                        compliance_diameter_mm=compliance_diameter_mm,
                        compliance_diameter_label=compliance_diameter_label,
                        measured_ovality_mm=measured_ovality_mm,
                        ovality_lookup=ovality_lookup,
                        diameter_lookup=diameter_lookup,
                        measured_wall_thickness_mm=measured_wall_thickness_mm,
                    )
                    if combined_report is not None:
                        standard_label = combined_report.get("standard_label", "")
                        render_standard_check_overlay(combined_report)
                        render_combined_tolerance_results(combined_report)
                        # Convert to legacy format for build_result_row / visual analysis compatibility
                        tolerance_report = combined_report
                except Exception as exc:
                    st.error(f"OSTB standards check failed: {exc}")

    render_visual_analysis(
        test_processed,
        test_measurement,
        tolerance_report,
        unit,
        wall_reference_measurement=inner_reference_measurement,
    )

    result_df = build_result_row(test_measurement, tolerance_report, standard_label, scale_source, measurement_target)
    st.subheader("Export")
    st.dataframe(result_df, width="stretch")
    st.download_button(
        "Download CSV result",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="roundness_result.csv",
        mime="text/csv",
        width="stretch",
    )
    render_page_footer()


if __name__ == "__main__":
    main()
