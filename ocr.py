"""Tesseract OCR wrapper returning text plus word-level boxes and confidences."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Sequence

import cv2
import numpy as np
import pandas as pd
import pytesseract

PSM_MODES = {
    3: "Fully automatic page segmentation",
    4: "Single column of variable-size text",
    6: "Single uniform block of text",
    11: "Sparse text (find as much as possible)",
    12: "Sparse text with orientation detection",
}
WORD_COLUMNS = ["block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"]


@dataclass
class OCRResult:
    """Text, word boxes and quality of one OCR run."""

    text: str = ""
    words: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=WORD_COLUMNS))
    mean_conf: float = float("nan")
    lang: str = "eng"
    psm: int = 3


def available_languages() -> List[str]:
    """Tesseract language packs installed on this machine."""
    try:
        return sorted(l for l in pytesseract.get_languages(config="") if l != "osd") or ["eng"]
    except Exception:
        return ["eng"]


def run_ocr(img: np.ndarray, lang: str = "eng", psm: int = 3, min_conf: float = 0,
            layout: str = "rows") -> OCRResult:
    """OCR a grayscale / RGB image.

    Args:
        img: ``uint8`` image array.
        lang: Tesseract language(s), e.g. ``"eng"`` or ``"eng+fra"``.
        psm: Page segmentation mode (see :data:`PSM_MODES`).
        min_conf: Drop words with confidence below this (0-100).
        layout: ``"rows"`` reads straight across the page line by line (best for invoices, forms, tables);
            ``"blocks"`` keeps Tesseract's own block order (best for multi-column prose).
    """
    d = pytesseract.image_to_data(img, lang=lang, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT)
    df = pd.DataFrame({k: d[k] for k in WORD_COLUMNS})
    df["text"] = df["text"].astype(str).str.strip()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
    df = df[(df["text"] != "") & (df["conf"] >= 0) & (df["conf"] >= min_conf)].reset_index(drop=True)
    text = words_to_rows_text(df) if layout == "rows" else words_to_text(df)
    return OCRResult(text, df, float(df["conf"].mean()) if len(df) else float("nan"), lang, psm)


def words_to_text(df: pd.DataFrame) -> str:
    """Rebuild reading-order text from word rows (blank line between paragraphs / blocks)."""
    if df.empty:
        return ""
    out, prev = [], None
    for key, g in df.sort_values(["block_num", "par_num", "line_num", "word_num"]).groupby(
            ["block_num", "par_num", "line_num"], sort=False):
        if prev is not None and key[:2] != prev[:2]:
            out.append("")
        out.append(" ".join(g["text"]))
        prev = key
    return "\n".join(out)


def group_lines(words: pd.DataFrame, tol: float = 0.6) -> List[pd.DataFrame]:
    """Group word boxes into visual text rows (by vertical centre), each sorted left→right."""
    if words.empty:
        return []
    w = words.assign(_cy=words["top"] + words["height"] / 2.0).sort_values("_cy")
    mh = float(w["height"].median())
    lines, cur, cy = [], [], 0.0
    for idx, c in zip(w.index, w["_cy"]):
        if cur and abs(c - cy) > tol * mh:
            lines.append(cur)
            cur = []
        cur.append(idx)
        cy = float(w.loc[cur, "_cy"].mean())
    lines.append(cur)
    return [w.loc[ix].sort_values("left").drop(columns="_cy") for ix in lines]


def line_cells(line: pd.DataFrame, gap: float) -> List[tuple]:
    """Split one row into cells wherever the horizontal gap exceeds ``gap`` pixels: ``(x0, x1, text)``."""
    cells, cur = [], None
    for r in line.itertuples():
        if cur is not None and r.left - cur[1] <= gap:
            cur = [cur[0], r.left + r.width, cur[2] + " " + r.text]
        else:
            if cur is not None:
                cells.append(tuple(cur))
            cur = [r.left, r.left + r.width, r.text]
    if cur is not None:
        cells.append(tuple(cur))
    return cells


def words_to_rows_text(df: pd.DataFrame, gap_factor: float = 1.5) -> str:
    """Row-wise text: cells separated by three spaces, blank line at larger vertical gaps."""
    lines = group_lines(df)
    if not lines:
        return ""
    mh = float(df["height"].median())
    out, prev_cy = [], None
    for ln in lines:
        cy = float((ln["top"] + ln["height"] / 2).mean())
        if prev_cy is not None and cy - prev_cy > 3.2 * mh:
            out.append("")
        out.append("   ".join(c[2] for c in line_cells(ln, gap_factor * mh)))
        prev_cy = cy
    return "\n".join(out)


def draw_boxes(img: np.ndarray, words: pd.DataFrame) -> np.ndarray:
    """RGB copy of ``img`` with word boxes coloured by confidence (green high → red low)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    for _, r in words.iterrows():
        t = min(max(r["conf"] / 100.0, 0), 1)
        color = (int(255 * (1 - t)), int(200 * t), 40)
        cv2.rectangle(rgb, (int(r.left), int(r.top)), (int(r.left + r.width), int(r.top + r.height)), color, 2)
    return rgb


def searchable_pdf(images: Sequence[np.ndarray], lang: str = "eng", psm: int = 3) -> bytes:
    """Build a PDF whose pages show the images with an invisible, selectable OCR text layer."""
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for im in images:
        page_pdf = pytesseract.image_to_pdf_or_hocr(im, lang=lang, extension="pdf", config=f"--oem 3 --psm {psm}")
        for p in PdfReader(io.BytesIO(page_pdf)).pages:
            writer.add_page(p)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
