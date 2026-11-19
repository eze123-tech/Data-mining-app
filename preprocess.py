"""Document-photo clean-up: loading, page detection, rotation, de-skew, shadow removal, binarisation.

Typical pipeline for a phone photo of a page::

    load_images -> limit size -> find page quad + perspective warp -> (auto-rotate 90°) ->
    de-skew -> upscale small images -> grayscale enhance (shadow removal / threshold)
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
METHODS = {
    "shadow": "Shadow removal + Otsu (best for phone photos)",
    "adaptive": "Adaptive threshold (uneven light)",
    "otsu": "Otsu threshold (clean scans)",
    "clahe": "Contrast boost, grayscale (faint print)",
    "none": "Grayscale only",
}


@dataclass
class PreprocessOptions:
    """Switches for :func:`preprocess`."""

    auto_crop: bool = True      # detect the page in a photo and flatten its perspective
    auto_rotate: bool = False   # fix 90°/180° orientation with Tesseract OSD
    deskew: bool = True         # straighten slightly tilted text
    denoise: bool = True
    method: str = "shadow"      # key of METHODS
    max_side: int = 3000        # shrink huge photos to this long side (speed / memory)
    min_side: int = 1600        # enlarge small images to at least this long side


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess`."""

    aligned: np.ndarray         # RGB, cropped / rotated / de-skewed, before thresholding
    processed: np.ndarray       # single-channel image fed to OCR
    steps: List[str] = field(default_factory=list)
    skew_deg: float = 0.0
    cropped: bool = False


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_images(data: bytes, filename: str = "", dpi: int = 200, max_pages: int = 10) -> List[Image.Image]:
    """Decode an uploaded image / multi-frame TIFF / PDF into RGB PIL images (one per page)."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        from pdf2image import convert_from_bytes
        return [p.convert("RGB") for p in convert_from_bytes(data, dpi=dpi, first_page=1, last_page=max_pages)]
    img = Image.open(io.BytesIO(data))
    frames = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    return frames[:max_pages]


def pdf_text_layer(data: bytes, max_pages: int = 10) -> List[str]:
    """Embedded text of each PDF page (empty strings for scanned pages)."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        return [(p.extract_text() or "").strip() for p in reader.pages[:max_pages]]
    except Exception:
        return []


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_document_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Corners (tl, tr, br, bl) of the largest page-like quadrilateral, or ``None``.

    Returns ``None`` when the page already fills the frame (flat scan) or nothing convincing is found.
    """
    h, w = bgr.shape[:2]
    scale = 700.0 / max(h, w) if max(h, w) > 700 else 1.0
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else bgr
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 130), np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and cv2.isContourConvex(approx) and 0.25 * area_img < area < 0.97 * area_img:
            return _order_points(approx.reshape(4, 2) / scale)
    return None


def four_point_transform(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp ``quad`` (tl, tr, br, bl) to a flat rectangle."""
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (width, height), flags=cv2.INTER_CUBIC)


def rotate(img: np.ndarray, angle: float, border=(255, 255, 255)) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees keeping the canvas size."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Correction angle (degrees, pass to :func:`rotate`) that makes text lines horizontal.

    Uses the projection-profile method: the right angle maximises the sharpness of the row histogram.
    """
    h, w = gray.shape[:2]
    s = 700.0 / max(h, w) if max(h, w) > 700 else 1.0
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else gray
    bw = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    if bw.sum() / 255 < 200:
        return 0.0
    hh, ww = bw.shape

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
        prof = cv2.warpAffine(bw, m, (ww, hh), flags=cv2.INTER_NEAREST).sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = float(coarse[int(np.argmax([score(a) for a in coarse]))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    best = float(fine[int(np.argmax([score(a) for a in fine]))])
    return 0.0 if abs(best) < 0.2 else best


def detect_orientation(gray: np.ndarray) -> int:
    """Degrees (0/90/180/270) to rotate *clockwise* to make text upright (Tesseract OSD); 0 on failure."""
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0)) % 360
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Enhancement
# --------------------------------------------------------------------------- #
def enhance(gray: np.ndarray, method: str = "shadow", denoise: bool = True) -> np.ndarray:
    """Grayscale -> OCR-ready image according to ``method`` (see :data:`METHODS`)."""
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21) if gray.size < 1_500_000 else cv2.medianBlur(gray, 3)
    if method == "none":
        return gray
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        block = max(15, (min(gray.shape[:2]) // 30) | 1)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block | 1, 12)
    # "shadow": estimate the illumination by closing away the dark text on a reduced copy, then divide it out
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR), (0, 0), 5)
    flat = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# --------------------------------------------------------------------------- #
def preprocess(img: Image.Image, opts: Optional[PreprocessOptions] = None) -> PreprocessResult:
    """Run the full clean-up pipeline on one page image."""
    o = opts or PreprocessOptions()
    steps: List[str] = []
    bgr = pil_to_bgr(img)
    h, w = bgr.shape[:2]
    if max(h, w) > o.max_side:
        f = o.max_side / max(h, w)
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        steps.append(f"downscaled to {bgr.shape[1]}×{bgr.shape[0]}")

    cropped = False
    if o.auto_crop:
        quad = find_document_quad(bgr)
        if quad is not None:
            bgr, cropped = four_point_transform(bgr, quad), True
            steps.append("page detected and perspective corrected")

    if o.auto_rotate:
        rot = detect_orientation(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if rot:
            code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
            bgr = cv2.rotate(bgr, code)
            steps.append(f"rotated {rot}° to upright")

    skew = 0.0
    if o.deskew:
        skew = estimate_skew(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if skew:
            bgr = rotate(bgr, skew)
            steps.append(f"de-skewed by {skew:+.1f}°")

    longest = max(bgr.shape[:2])
    if longest < o.min_side:
        f = min(3.0, o.min_side / longest)
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        steps.append(f"upscaled ×{f:.1f}")

    processed = enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), o.method, o.denoise)
    steps.append(f"enhanced: {o.method}" + (" + denoise" if o.denoise else ""))
    return PreprocessResult(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), processed, steps, skew, cropped)
