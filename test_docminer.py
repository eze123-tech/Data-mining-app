"""Tests: preprocessing geometry, OCR on synthetic documents, mining logic, exports."""
import io
import json

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import mining as M
from src.export import mining_sheets, report_markdown, to_excel_bytes, to_json_bytes
from src.ocr import run_ocr, searchable_pdf
from src.pipeline import PipelineOptions, mine_document, process_document
from src.preprocess import (PreprocessOptions, estimate_skew, find_document_quad, load_images, pil_to_bgr, preprocess,
                            rotate)
from src.sample import image_bytes, make_photo, make_sample_invoice


@pytest.fixture(scope="module")
def invoice():
    return make_sample_invoice()


@pytest.fixture(scope="module")
def clean_doc(invoice):
    return process_document(image_bytes(invoice), "inv.png")


@pytest.fixture(scope="module")
def photo_doc(invoice):
    return process_document(image_bytes(make_photo(invoice, angle=3.0)), "photo.png")


# ------------------------------------------------------------------ preprocessing
@pytest.mark.parametrize("tilt", [-5.0, 3.0, 7.5])
def test_deskew_recovers_tilt(invoice, tilt):
    gray = cv2.cvtColor(pil_to_bgr(invoice), cv2.COLOR_BGR2GRAY)
    tilted = rotate(gray, tilt, border=255)
    assert estimate_skew(tilted) == pytest.approx(-tilt, abs=0.8)


def test_flat_scan_is_not_cropped_but_photo_is(invoice):
    assert find_document_quad(pil_to_bgr(invoice)) is None
    q = find_document_quad(pil_to_bgr(make_photo(invoice)))
    assert q is not None and q.shape == (4, 2)


def test_photo_is_flattened_to_page_shape(invoice):
    r = preprocess(make_photo(invoice))
    assert r.cropped
    h, w = r.aligned.shape[:2]
    assert (w / h) == pytest.approx(1240 / 1754, rel=0.08)
    assert r.processed.ndim == 2 and set(np.unique(r.processed)) <= {0, 255}


@pytest.mark.parametrize("method", ["none", "clahe", "otsu", "adaptive", "shadow"])
def test_all_enhancement_methods_run(invoice, method):
    r = preprocess(invoice, PreprocessOptions(method=method))
    assert r.processed.shape[:2] == r.aligned.shape[:2]


def test_load_images_handles_png_multipage_tiff_and_pdf(invoice):
    assert len(load_images(image_bytes(invoice), "a.png")) == 1
    buf = io.BytesIO()
    invoice.save(buf, format="TIFF", save_all=True, append_images=[invoice, invoice])
    assert len(load_images(buf.getvalue(), "a.tif")) == 3
    buf = io.BytesIO()
    invoice.save(buf, format="PDF")
    assert len(load_images(buf.getvalue(), "a.pdf", dpi=72)) == 1


# ------------------------------------------------------------------ OCR + pipeline
@pytest.mark.parametrize("fixture", ["clean_doc", "photo_doc"])
def test_ocr_reads_key_fields(fixture, request):
    d = request.getfixturevalue(fixture)
    for needle in ("INV-2024-0457", "15 March 2024", "$354.75", "Total:", "Jane Doe"):
        assert needle in d.text, needle
    assert d.mean_conf > 80


def test_photo_ocr_is_close_to_clean(clean_doc, photo_doc):
    a, b = set(clean_doc.text.lower().split()), set(photo_doc.text.lower().split())
    assert len(a & b) / len(a) > 0.85


def test_rows_layout_keeps_label_and_value_together(invoice):
    pre = preprocess(invoice)
    assert "Total:   $354.75" in run_ocr(pre.processed, layout="rows").text


def test_pdf_with_text_layer_skips_ocr(invoice, clean_doc):
    pdf = searchable_pdf([preprocess(invoice).aligned])
    d = process_document(pdf, "s.pdf")
    assert d.pages[0].source == "pdf-text" and "INV-2024-0457" in d.text
    d2 = process_document(pdf, "s.pdf", PipelineOptions(use_pdf_text=False))
    assert d2.pages[0].source == "ocr" and "INV-2024-0457" in d2.text


# ------------------------------------------------------------------ mining
def test_entities_on_ocr_text(clean_doc):
    e = M.extract_entities(clean_doc.text)
    got = {(r.type, r.value) for r in e.itertuples()}
    assert ("email", "billing@acme-supplies.com") in got and ("id", "INV-2024-0457") in got
    assert ("date", "15 March 2024") in got and ("money", "$354.75") in got and ("url", "www.acme-supplies.com") in got
    assert e.loc[e.value == "15 March 2024", "normalized"].iloc[0] == "2024-03-15"
    assert e.loc[e.value == "14/04/2024", "normalized"].iloc[0] == "2024-04-14"          # day-first default


def test_dayfirst_switch():
    assert M.extract_entities("on 03/04/2024", dayfirst=True)["normalized"].iloc[0] == "2024-04-03"
    assert M.extract_entities("on 03/04/2024", dayfirst=False)["normalized"].iloc[0] == "2024-03-04"


@pytest.mark.parametrize("s,v", [("$1,234.50", 1234.5), ("1.234,50 EUR", 1234.5), ("12%", 12.0), ("€5,5", 5.5), ("abc", None), ("₦2,000", 2000.0)])
def test_parse_amount(s, v):
    assert M.parse_amount(s) == v


def test_no_false_positives_in_plain_numbers():
    e = M.extract_entities("Page 12345678 of 3. Room 42.")
    assert e[e.type == "phone"].empty and e[e.type == "money"].empty


def test_key_values(clean_doc):
    kv = M.extract_key_values(clean_doc.text).set_index("key")
    assert kv.loc["Total", "number"] == 354.75 and kv.loc["Subtotal", "number"] == 322.5
    assert pd.isna(kv.loc["Due Date", "number"])            # dates are not numbers
    assert kv.loc["Email", "value"] == "billing@acme-supplies.com"
    dotted = M.extract_key_values("Balance ........ $15.00")
    assert dotted.iloc[0]["number"] == 15.0


@pytest.mark.parametrize("fixture", ["clean_doc", "photo_doc"])
def test_table_detection(fixture, request):
    d = request.getfixturevalue(fixture)
    tables = M.detect_tables(d.words)
    items = max(tables, key=len)
    assert list(items.columns)[0] == "Item" and len(items) == 3 and items.shape[1] == 4
    assert "$125.00" in items.iloc[0].tolist()
    num = M.tables_to_numeric(items)
    assert num["Amount"].tolist() == [125.0, 180.0, 17.5] and not pd.api.types.is_numeric_dtype(num["Item"])


def test_describe_numbers_matches_numpy():
    v = [1, 2, 3, 4, 5, 100]
    s = M.describe_numbers(v)
    assert s["median"] == np.median(v) and s["q1"] == np.percentile(v, 25) and s["p90"] == np.percentile(v, 90)
    assert M.describe_numbers([]).empty and M.describe_numbers([float("nan")]).empty


def test_keywords_ignore_numbers_and_stopwords():
    k = M.keywords("The quick brown fox jumps over the lazy dog. Invoice INV-2024 total 354.75. The quick brown fox again.")
    phrases = k["phrase"].tolist()
    assert "quick brown fox jumps" in phrases or "quick brown fox" in phrases
    assert not any(ch.isdigit() for p in phrases for ch in p)


def test_text_stats():
    s = M.text_stats("Hello world. This is a test 123.")
    assert s["words"] == 7 and s["sentences"] == 2 and s["numeric tokens %"] == pytest.approx(100 / 7)


def test_classification(clean_doc):
    sc = M.classify_document(clean_doc.text)
    assert M.best_type(sc) == "invoice"
    assert M.best_type(M.classify_document("lorem ipsum dolor")) == "unknown"
    assert M.best_type(M.classify_document("Dear Sir, thanks. Sincerely, Bob")) == "letter"


def test_regex_search_and_bad_pattern():
    h = M.regex_search("a1\nb22\nc333", r"\d{2,}")
    assert h["match"].tolist() == ["22", "333"] and h["line_no"].tolist() == [2, 3]
    with pytest.raises(ValueError):
        M.regex_search("x", "(")


def test_corpus_mining_and_search():
    docs = {"inv1": "invoice total amount due payment terms subtotal tax",
            "inv2": "invoice subtotal tax amount payment due total",
            "cv": "curriculum vitae experience education skills university python"}
    c = M.corpus_mining(docs, 2)
    sim = c["similarity"]
    assert sim.loc["inv1", "inv2"] > sim.loc["inv1", "cv"]
    cl = c["clusters"].set_index("doc")["cluster"]
    assert cl["inv1"] == cl["inv2"] != cl["cv"]
    assert M.search_docs(docs, "python education").iloc[0]["doc"] == "cv"
    assert M.search_docs(docs, "   ").empty
    with pytest.raises(ValueError):
        M.corpus_mining({"a": "x"})


def test_mine_document_shape(clean_doc):
    m = mine_document(clean_doc.text, clean_doc.words, clean_doc.mean_conf)
    assert m["doc_type"] == "invoice" and len(m["tables"]) >= 1
    assert "sum" not in m["money_summary"].index and m["money_summary"]["max"] == 354.75


# ------------------------------------------------------------------ export
def test_exports(clean_doc):
    m = mine_document(clean_doc.text, clean_doc.words, clean_doc.mean_conf)
    sheets = mining_sheets("inv.png", clean_doc.text, m)
    xl = pd.read_excel(io.BytesIO(to_excel_bytes(sheets)), sheet_name=None)
    assert {"summary", "entities", "key_values", "keywords", "text"} <= set(xl)
    assert any(k.startswith("table_") for k in xl)
    js = json.loads(to_json_bytes("inv.png", clean_doc.text, m))
    assert js["doc_type"] == "invoice" and js["tables"][0]["rows"]
    md = report_markdown("inv.png", clean_doc.text, m)
    assert "# Document report" in md and "INV-2024-0457" in md


def test_searchable_pdf_roundtrip(invoice):
    from pypdf import PdfReader
    pdf = searchable_pdf([preprocess(invoice).aligned, preprocess(invoice).aligned])
    r = PdfReader(io.BytesIO(pdf))
    assert len(r.pages) == 2 and "INV-2024-0457" in r.pages[0].extract_text()
