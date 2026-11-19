"""End-to-end processing: file bytes -> pages -> clean-up -> OCR -> mining results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from . import mining as M
from .ocr import OCRResult, WORD_COLUMNS, run_ocr
from .preprocess import PreprocessOptions, load_images, pdf_text_layer, preprocess


@dataclass
class PipelineOptions:
    """All user-tunable settings of the pipeline."""

    lang: str = "eng"
    psm: int = 3
    min_conf: int = 0
    layout: str = "rows"            # "rows" | "blocks"
    dpi: int = 200                  # PDF rasterisation resolution
    max_pages: int = 10
    use_pdf_text: bool = True       # use a PDF's embedded text instead of OCR when it has some
    pre: PreprocessOptions = field(default_factory=PreprocessOptions)


@dataclass
class PageResult:
    """One processed page."""

    index: int
    original: np.ndarray            # RGB preview (≤ 1600 px)
    aligned: np.ndarray             # RGB after crop / rotate / de-skew
    processed: np.ndarray           # OCR input
    ocr: OCRResult
    source: str                     # "ocr" | "pdf-text"
    steps: List[str]
    skew_deg: float
    cropped: bool


@dataclass
class DocResult:
    """A processed document (one or more pages)."""

    name: str
    pages: List[PageResult]
    elapsed: float

    @property
    def text(self) -> str:
        return "\n\n".join(p.ocr.text for p in self.pages).strip()

    @property
    def words(self) -> pd.DataFrame:
        frames = [p.ocr.words.assign(page=p.index) for p in self.pages if len(p.ocr.words)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=WORD_COLUMNS + ["page"])

    @property
    def mean_conf(self) -> float:
        w = self.words
        return float(w["conf"].mean()) if len(w) else float("nan")


def _preview(rgb: np.ndarray, side: int = 1600) -> np.ndarray:
    f = side / max(rgb.shape[:2])
    return cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else rgb


def process_document(data: bytes, name: str, opts: Optional[PipelineOptions] = None) -> DocResult:
    """Run clean-up and OCR over every page of an image / PDF."""
    o = opts or PipelineOptions()
    t0 = time.time()
    images = load_images(data, name, o.dpi, o.max_pages)
    layer = pdf_text_layer(data, o.max_pages) if (o.use_pdf_text and (name.lower().endswith(".pdf") or data[:4] == b"%PDF")) else []
    pages = []
    for i, img in enumerate(images):
        pre = preprocess(img, o.pre)
        if i < len(layer) and len(layer[i]) > 20:
            ocr, src = OCRResult(layer[i], pd.DataFrame(columns=WORD_COLUMNS), float("nan"), "pdf", o.psm), "pdf-text"
        else:
            ocr, src = run_ocr(pre.processed, o.lang, o.psm, o.min_conf, o.layout), "ocr"
        pages.append(PageResult(i + 1, _preview(np.array(img.convert("RGB"))), pre.aligned, pre.processed, ocr, src,
                                pre.steps, pre.skew_deg, pre.cropped))
    return DocResult(name, pages, time.time() - t0)


def mine_document(text: str, words: pd.DataFrame, mean_conf: float = float("nan"), dayfirst: bool = True,
                  top_n: int = 15) -> Dict[str, object]:
    """Run every miner on a document's text (and word boxes for tables)."""
    ents = M.extract_entities(text, dayfirst)
    kv = M.extract_key_values(text)
    tables = []
    if len(words):
        for pg, w in words.groupby("page") if "page" in words.columns else [(1, words)]:
            tables += [(int(pg), t) for t in M.detect_tables(w)]
    money = ents.loc[ents["type"] == "money", "normalized"].dropna().astype(float).tolist()
    scores = M.classify_document(text)
    return {"entities": ents, "key_values": kv, "tables": tables, "keywords": M.keywords(text, top_n),
            "word_freq": M.word_frequencies(text, top_n), "stats": M.text_stats(text, mean_conf),
            "money_summary": M.describe_numbers(money).drop(labels=["sum"], errors="ignore"), "classification": scores, "doc_type": M.best_type(scores)}
