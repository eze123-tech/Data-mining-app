"""Document Scanner & Miner — Streamlit entry point (run: ``streamlit run app.py``).

Take a picture of a document (or upload images / PDFs) → clean up the photo → OCR → mine the text for
entities, fields, tables, keywords, statistics and cross-document patterns → export.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src import mining as M
from src.export import mining_sheets, report_markdown, to_excel_bytes, to_json_bytes
from src.ocr import PSM_MODES, available_languages, draw_boxes, searchable_pdf
from src.pipeline import DocResult, PipelineOptions, mine_document, process_document
from src.preprocess import IMAGE_TYPES, METHODS, PreprocessOptions
from src.sample import image_bytes, make_photo, make_sample_invoice

st.set_page_config(page_title="Document Scanner & Miner", page_icon="📄", layout="wide")
ss = st.session_state
for k, v in {"files": {}, "digests": {}, "gen": 0, "cam_n": 0, "pdfs": {}}.items():
    ss.setdefault(k, v)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def opts_from_dict(d: dict) -> PipelineOptions:
    """Rebuild :class:`PipelineOptions` (with nested preprocessing options) from ``asdict`` output."""
    d = dict(d)
    return PipelineOptions(**{**d, "pre": PreprocessOptions(**d["pre"])})


@st.cache_data(show_spinner=False, max_entries=30)
def process_cached(data: bytes, name: str, opts_json: str) -> DocResult:
    """OCR pipeline, cached on file content + settings so reruns are instant."""
    return process_document(data, name, opts_from_dict(json.loads(opts_json)))


def add_file(name: str, data: bytes) -> None:
    """Register a captured / uploaded file (duplicates by content are ignored, name clashes renamed)."""
    digest = hashlib.md5(data).hexdigest()
    if digest in ss["digests"].values():
        return
    stem, ext, n = os.path.splitext(name)[0], os.path.splitext(name)[1], 2
    while name in ss["files"]:
        name, n = f"{stem}_{n}{ext}", n + 1
    ss["files"][name], ss["digests"][name] = data, digest


def remove_file(name: str) -> None:
    """Callback: drop one document."""
    ss["files"].pop(name, None)
    ss["digests"].pop(name, None)


def clear_all() -> None:
    """Callback: drop all documents and reset the input widgets (new widget keys)."""
    ss["files"], ss["digests"], ss["pdfs"] = {}, {}, {}
    ss["gen"] += 1


def num(x, nd: int = 1) -> str:
    """Format a number, tolerating NaN."""
    return "–" if x is None or x != x else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Sidebar: settings
# --------------------------------------------------------------------------- #
st.sidebar.title("📄 Settings")
st.sidebar.subheader("Recognition")
langs = available_languages()
sel_langs = st.sidebar.multiselect("OCR language(s)", langs, default=["eng"] if "eng" in langs else langs[:1], key="langs")
psm = st.sidebar.selectbox("Page layout analysis", list(PSM_MODES), format_func=lambda k: f"{k} · {PSM_MODES[k]}", key="psm")
layout_lbl = st.sidebar.radio("Reading order", ["Row-wise (invoices, forms, tables)", "Tesseract blocks (multi-column prose)"], key="layout")
min_conf = st.sidebar.slider("Ignore words below confidence", 0, 90, 0, 5, key="minconf")

st.sidebar.subheader("Photo clean-up")
pre = PreprocessOptions(
    auto_crop=st.sidebar.checkbox("Detect page & fix perspective", True, key="crop"),
    auto_rotate=st.sidebar.checkbox("Auto-rotate (90°/180°)", False, key="rot"),
    deskew=st.sidebar.checkbox("Straighten tilted text", True, key="deskew"),
    denoise=st.sidebar.checkbox("Denoise", True, key="denoise"),
    method=st.sidebar.selectbox("Enhancement", list(METHODS), format_func=METHODS.get, key="method"))

st.sidebar.subheader("PDF")
dpi = st.sidebar.slider("Rasterise at DPI", 100, 300, 200, 25, key="dpi")
max_pages = st.sidebar.slider("Max pages per file", 1, 30, 10, key="maxpages")
use_text = st.sidebar.checkbox("Use embedded PDF text when present", True, key="pdftext")

st.sidebar.subheader("Mining")
dayfirst = st.sidebar.checkbox("Dates are day-first (DD/MM/YYYY)", True, key="dayfirst")
top_n = st.sidebar.slider("Top keywords", 5, 40, 15, key="topn")
st.sidebar.button("🗑 Clear all documents", on_click=clear_all, key="clear")

opts = PipelineOptions("+".join(sel_langs) or "eng", int(psm), int(min_conf),
                       "rows" if layout_lbl.startswith("Row") else "blocks", int(dpi), int(max_pages), use_text, pre)
opts_json = json.dumps(asdict(opts), sort_keys=True)
sig = hashlib.md5(opts_json.encode()).hexdigest()[:8]

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("📄 Document Scanner & Miner")
st.caption("Photograph or upload documents → clean up → OCR → extract entities, fields, tables & statistics → export")
tab_cap, tab_prev, tab_text, tab_mine, tab_corpus, tab_exp, tab_about = st.tabs(
    ["1 · Capture", "2 · Preview", "3 · Text", "4 · Mine", "5 · Compare documents", "6 · Export", "About"])

# ---- 1 · Capture -----------------------------------------------------------------------------
with tab_cap:
    source = st.radio("Source", ["📷 Camera", "⬆ Upload files", "🧪 Try a sample"], horizontal=True, key="source")
    if source.startswith("📷"):
        st.info("Lay the page flat on a darker surface, fill the frame, avoid glare and shadows. "
                "On a phone, allow camera access when the browser asks.")
        shot = st.camera_input("Take a picture of the document", key=f"cam_{ss['gen']}")
        if shot is not None:
            data = shot.getvalue()
            if hashlib.md5(data).hexdigest() not in ss["digests"].values():
                ss["cam_n"] += 1
                add_file(f"photo_{ss['cam_n']}.jpg", data)
    elif source.startswith("⬆"):
        ups = st.file_uploader("Images or PDFs", type=IMAGE_TYPES + ["pdf"], accept_multiple_files=True, key=f"up_{ss['gen']}")
        for f in ups or []:
            add_file(f.name, f.getvalue())
    else:
        c1, c2 = st.columns(2)
        if c1.button("Add clean sample invoice", key="s1"):
            add_file("sample_invoice.png", image_bytes(make_sample_invoice()))
        if c2.button("Add simulated phone photo of it", key="s2"):
            add_file("sample_invoice_photo.png", image_bytes(make_photo(make_sample_invoice())))

results = {}
if ss["files"]:
    bar = st.progress(0.0, text="Reading documents…")
    todo = list(ss["files"].items())
    for i, (n, d) in enumerate(todo, 1):
        try:
            results[n] = process_cached(d, n, opts_json)
        except Exception as exc:
            st.error(f"Could not read **{n}** (not a valid image/PDF?) and removed it: {exc}")
            remove_file(n)
        bar.progress(i / len(todo), text=f"Processed {i}/{len(todo)}: {n}")
    bar.empty()

with tab_cap:
    if results:
        st.subheader(f"{len(results)} document(s)")
        for n, r in results.items():
            c = st.columns([4, 1, 1, 1, 1])
            c[0].write(f"**{n}**")
            c[1].write(f"{len(r.pages)} page(s)")
            c[2].write(f"conf {num(r.mean_conf)}%")
            c[3].write(f"{r.elapsed:.1f}s")
            c[4].button("Remove", key=f"rm_{n}", on_click=remove_file, args=(n,))
    else:
        st.caption("No documents yet — take a picture, upload a file, or try a sample.")

cur = None
if results:
    cur = st.selectbox("Document to inspect", list(results), key="cur")
res = results.get(cur) if cur else None


def current_text(name: str) -> str:
    """OCR text of ``name`` including any manual edits made in the Text tab."""
    return ss.get(f"txt_{name}_{sig}", results[name].text)


# ---- 2 · Preview -----------------------------------------------------------------------------
with tab_prev:
    if res is None:
        st.info("Add a document first.")
    else:
        pg = st.slider("Page", 1, len(res.pages), 1, key="pg") if len(res.pages) > 1 else 1
        page = res.pages[pg - 1]
        st.caption(" → ".join(page.steps) + f"  ·  text source: {page.source}")
        show_boxes = st.checkbox("Show recognised words (green = confident, red = doubtful)", False, key="boxes",
                                 disabled=page.ocr.words.empty)
        a, b, c = st.columns(3)
        a.image(page.original, caption="Original")
        b.image(page.aligned, caption="Straightened / cropped")
        c.image(draw_boxes(page.processed, page.ocr.words) if show_boxes else page.processed, caption="Image sent to OCR")

# ---- 3 · Text --------------------------------------------------------------------------------
with tab_text:
    if res is None:
        st.info("Add a document first.")
    else:
        m = st.columns(4)
        m[0].metric("Pages", len(res.pages))
        m[1].metric("Mean confidence", f"{num(res.mean_conf)} %")
        m[2].metric("Words", len(res.words) if len(res.words) else len(res.text.split()))
        m[3].metric("Processing time", f"{res.elapsed:.1f}s")
        st.text_area("Recognised text — edit to fix OCR mistakes; the Mine and Export tabs use your edits",
                     value=res.text, height=380, key=f"txt_{cur}_{sig}")
        st.download_button("⬇ Download text (.txt)", current_text(cur).encode("utf-8"), f"{cur}.txt", "text/plain", key="dl_txt")
        w = res.words
        if len(w):
            low = w[w["conf"] < 60][["page", "text", "conf"]].sort_values("conf")
            with st.expander(f"Words the OCR was unsure about (confidence < 60): {len(low)}"):
                st.dataframe(low, hide_index=True)

# ---- 4 · Mine --------------------------------------------------------------------------------
mined_cur = None
with tab_mine:
    if res is None:
        st.info("Add a document first.")
    else:
        text = current_text(cur)
        mined_cur = mined = mine_document(text, res.words, res.mean_conf, dayfirst, top_n)
        ents = mined["entities"]
        h = st.columns(4)
        h[0].metric("Detected type", mined["doc_type"])
        h[1].metric("Entities found", len(ents))
        h[2].metric("Fields (key: value)", len(mined["key_values"]))
        h[3].metric("Tables", len(mined["tables"]))

        st.subheader("Entities")
        if ents.empty:
            st.caption("None found.")
        else:
            kinds = st.multiselect("Show types", sorted(ents["type"].unique()), default=sorted(ents["type"].unique()), key="ent_types")
            st.dataframe(ents[ents["type"].isin(kinds)], hide_index=True)
            st.bar_chart(ents["type"].value_counts())

        st.subheader("Fields")
        if mined["key_values"].empty:
            st.caption("No `Key: value` lines found.")
        else:
            st.dataframe(mined["key_values"], hide_index=True)

        st.subheader("Tables")
        if not mined["tables"]:
            st.caption("No tables detected (needs a clear grid of aligned columns).")
        for i, (pg, t) in enumerate(mined["tables"], 1):
            st.markdown(f"**Table {i}** (page {pg})")
            st.dataframe(t, hide_index=True)
            tn = M.tables_to_numeric(t)
            numeric = tn.select_dtypes("number")
            if not numeric.empty:
                d = numeric.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T
                st.caption("Summary of numeric columns (count, mean, std, min, quartiles, 90th percentile, max)")
                st.dataframe(d)

        st.subheader("Amounts mentioned")
        ms = mined["money_summary"]
        if len(ms):
            c1, c2 = st.columns([1, 2])
            c1.dataframe(ms.rename("value").rename_axis("statistic").to_frame())
            amounts = ents.loc[ents["type"] == "money", ["value", "normalized"]].set_index("value")
            c2.bar_chart(amounts)
        else:
            st.caption("No currency amounts found.")

        st.subheader("Key phrases & words")
        k1, k2 = st.columns(2)
        k1.dataframe(mined["keywords"], hide_index=True)
        if len(mined["keywords"]):
            k1.bar_chart(mined["keywords"].set_index("phrase")["score"])
        k2.dataframe(mined["word_freq"], hide_index=True)
        if len(mined["word_freq"]):
            k2.bar_chart(mined["word_freq"].set_index("word")["count"])

        st.subheader("Text statistics")
        st.dataframe(pd.Series(mined["stats"], name="value").rename_axis("metric").to_frame())

        st.subheader("Document type evidence")
        st.dataframe(mined["classification"], hide_index=True)
        st.caption("Keyword-rule heuristic (transparent, not a trained model).")

        st.subheader("Search this document")
        pat = st.text_input("Regular expression (e.g. `\\bINV-\\d+` or `total.*`)", key="rx")
        if pat:
            try:
                hits = M.regex_search(text, pat)
                st.write(f"{len(hits)} match(es)")
                st.dataframe(hits, hide_index=True)
            except ValueError as exc:
                st.error(str(exc))

# ---- 5 · Compare documents ---------------------------------------------------------------------
with tab_corpus:
    if len(results) < 2:
        st.info("Add at least two documents to compare, cluster and search across them.")
    else:
        texts = {n: current_text(n) for n in results}
        all_mined = {n: mine_document(texts[n], results[n].words, results[n].mean_conf, dayfirst, top_n) for n in results}
        st.subheader("Overview")
        ov = pd.DataFrame([{"document": n, "type": a["doc_type"], "pages": len(results[n].pages), "words": a["stats"]["words"],
                            "confidence": results[n].mean_conf, "entities": len(a["entities"]), "tables": len(a["tables"])}
                           for n, a in all_mined.items()])
        st.dataframe(ov, hide_index=True)
        st.subheader("Entities across documents")
        long = pd.concat([a["entities"].assign(document=n) for n, a in all_mined.items() if len(a["entities"])], ignore_index=True) \
            if any(len(a["entities"]) for a in all_mined.values()) else pd.DataFrame()
        if len(long):
            st.dataframe(pd.crosstab(long["document"], long["type"]))
            kinds = st.multiselect("Show entity types", sorted(long["type"].unique()), default=["email", "date", "money"] if {"email", "date", "money"} & set(long["type"]) else sorted(long["type"].unique()), key="c_types")
            st.dataframe(long[long["type"].isin(kinds)][["document", "type", "value", "normalized", "context"]], hide_index=True)
        st.subheader("Similarity & clusters (TF-IDF)")
        k = st.slider("Number of clusters", 1, min(6, len(results)), min(3, len(results)), key="k")
        try:
            cm = M.corpus_mining(texts, k)
            st.dataframe(cm["similarity"])
            c1, c2 = st.columns(2)
            c1.dataframe(cm["clusters"], hide_index=True)
            c2.dataframe(cm["terms"], hide_index=True)
        except ValueError as exc:
            st.warning(f"Not enough text to compare: {exc}")
        st.subheader("Search across documents")
        q = st.text_input("Search query (free text)", key="q")
        if q:
            st.dataframe(M.search_docs(texts, q), hide_index=True)

# ---- 6 · Export --------------------------------------------------------------------------------
with tab_exp:
    if res is None:
        st.info("Add a document first.")
    else:
        text = current_text(cur)
        mined = mined_cur or mine_document(text, res.words, res.mean_conf, dayfirst, top_n)
        st.subheader(f"Export — {cur}")
        c = st.columns(4)
        c[0].download_button("⬇ Excel workbook", to_excel_bytes(mining_sheets(cur, text, mined)), f"{cur}_mined.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="x_xlsx")
        c[1].download_button("⬇ JSON", to_json_bytes(cur, text, mined), f"{cur}_mined.json", "application/json", key="x_json")
        c[2].download_button("⬇ Markdown report", report_markdown(cur, text, mined).encode("utf-8"), f"{cur}_report.md", "text/markdown", key="x_md")
        c[3].download_button("⬇ Entities (CSV)", mined["entities"].to_csv(index=False).encode("utf-8"), f"{cur}_entities.csv", "text/csv", key="x_csv")
        st.markdown("**Searchable PDF** — the original page with an invisible, selectable text layer")
        pdf_key = f"{cur}_{sig}"
        if st.button("Create searchable PDF", key="mkpdf"):
            with st.spinner("Building PDF…"):
                ss["pdfs"][pdf_key] = searchable_pdf([p.aligned for p in res.pages], opts.lang, opts.psm)
        if pdf_key in ss["pdfs"]:
            st.download_button("⬇ Download searchable PDF", ss["pdfs"][pdf_key], f"{cur}_searchable.pdf", "application/pdf", key="x_pdf")
        if len(results) > 1:
            st.subheader("All documents")
            frames_e, frames_k, rows = [], [], []
            for n, r in results.items():
                a = mine_document(current_text(n), r.words, r.mean_conf, dayfirst, top_n)
                frames_e.append(a["entities"].assign(document=n))
                frames_k.append(a["key_values"].assign(document=n))
                rows.append({"document": n, "type": a["doc_type"], "pages": len(r.pages), "confidence": r.mean_conf,
                             **{k: v for k, v in a["stats"].items() if k in ("words", "sentences")}})
            sheets = {"documents": pd.DataFrame(rows), "entities": pd.concat(frames_e, ignore_index=True),
                      "key_values": pd.concat(frames_k, ignore_index=True)}
            st.download_button("⬇ Combined Excel (all documents)", to_excel_bytes(sheets), "all_documents_mined.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="x_all")

# ---- About -------------------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### How it works
1. **Capture** – camera, upload (PNG/JPG/TIFF/WebP/BMP/PDF) or a built-in sample.
2. **Clean-up** (OpenCV) – detect the page in a photo and flatten its perspective, optional 90° auto-rotation, straighten
   tilted text, remove uneven lighting/shadows and binarise.
3. **OCR** (Tesseract) – text with per-word confidence and boxes. PDFs with embedded text skip OCR.
4. **Mining** – regex entities (emails, URLs, phones, dates, currency amounts, percentages, reference IDs), `Key: value` fields,
   tables rebuilt from word positions, RAKE key phrases, word frequencies, readability and numeric summaries
   (mean, median, quartiles, percentiles), a transparent keyword-rule document classifier, and — across documents — TF-IDF
   similarity, k-means clusters and ranked search.
5. **Export** – Excel, JSON, Markdown, CSV, searchable PDF.

### Tips for good results
* Fill the frame, hold the camera parallel to the page, use even light and a darker background; avoid glare.
* Pick the right **reading order**: row-wise for invoices/forms/tables, blocks for newspapers or multi-column text.
* Check low-confidence words in the **Text** tab and fix them – everything downstream uses your edits.
* Install extra OCR languages on the server to read non-English documents.

### Limits & privacy
* OCR on photos is imperfect (small print, handwriting, curved pages, glare). Verify important numbers.
* Table detection needs clearly separated columns; ruled lines are ignored, spacing is what counts.
* Dates like 03/04/2024 are ambiguous – set day-first / month-first in the sidebar.
* Files are processed in memory on the server running this app; nothing is stored by the app. If you deploy it publicly,
  do not upload sensitive documents unless you control the host.
""")
