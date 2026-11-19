# Document Scanner & Miner

Take a picture of a document (or upload images / PDFs), clean it up, read it with OCR, and **mine** it for data.

**Pipeline:** camera / upload → page detection + perspective fix → de-skew → shadow removal → Tesseract OCR →
entities · fields · tables · keywords · statistics · classification → Excel / JSON / Markdown / searchable PDF.

## Features
- **Capture:** browser camera (works on phones), multi-file upload (PNG, JPG, TIFF, WebP, BMP, PDF), built-in sample documents.
- **Photo clean-up (OpenCV):** finds the page in a photo and flattens it, optional 90°/180° auto-rotate, straightens tilted text,
  removes uneven lighting, five enhancement modes.
- **OCR (Tesseract):** any installed language, word confidences, boxes overlay, low-confidence review list, editable text.
  PDFs with embedded text skip OCR.
- **Mining**
  - entities: emails, URLs, phones, dates (normalised to ISO, day-first or month-first), currency amounts, percentages, reference IDs
  - `Key: value` fields (`Total:   $354.75`, `Balance ..... 15.00`)
  - tables rebuilt from word positions, with numeric summary (count, mean, std, min, quartiles, 90th percentile, max)
  - amounts summary, RAKE key phrases, word frequencies, readability, keyword-rule document-type scores
  - regex search inside a document
  - across documents: TF-IDF similarity, k-means clusters, ranked search, entity cross-tab
- **Export:** Excel workbook, JSON, Markdown report, entities CSV, searchable PDF, combined workbook for all documents.

## Run locally
Needs Python ≥ 3.9 plus the **Tesseract** and **Poppler** programs:

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd poppler-utils
# macOS:   brew install tesseract poppler
# Windows: install Tesseract (UB Mannheim build) and Poppler, and add both to PATH

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Tests: `pip install -r requirements-dev.txt && pytest -q`

## Deploy on Streamlit Community Cloud
1. Push this folder's **contents** to a GitHub repository (`app.py` must be at the repo root).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Deploy. Cloud installs `requirements.txt` with pip **and the system packages in `packages.txt`** (Tesseract, Poppler).
   That file is what makes OCR work in the cloud — keep it.
4. For other languages add e.g. `tesseract-ocr-fra`, `tesseract-ocr-deu` lines to `packages.txt`; they appear in the sidebar.

Camera capture needs HTTPS (Cloud provides it) and the user's permission in the browser.
Privacy: files are processed in server memory and not stored by the app; do not upload sensitive documents to a public deployment.

## Layout
```
app.py              UI
src/preprocess.py   load images/PDF, page detection, deskew, enhancement
src/ocr.py          Tesseract wrapper, row/cell grouping, searchable PDF
src/mining.py       entities, key-values, tables, keywords, stats, classification, corpus mining
src/pipeline.py     file -> pages -> OCR -> mining
src/export.py       Excel / JSON / Markdown
src/sample.py       synthetic invoice + simulated phone photo (demo & tests)
tests/              37 tests (real OCR on synthetic documents)
packages.txt        apt packages for Streamlit Cloud
```

## Limits
OCR of photos is imperfect (small print, handwriting, curved pages, glare) — verify important numbers; fix errors in the Text tab and
the mining updates. Table detection relies on column spacing. Document classification is a keyword heuristic, not a trained model.
