"""Data mining on OCR output: entities, key–value fields, tables, keywords, statistics, classification,
and cross-document analysis (TF-IDF similarity, clustering, search)."""
from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .ocr import group_lines, line_cells

# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
_MONTHS = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# (type, compiled pattern, group holding the value). Earlier patterns win overlapping text.
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), 0),
    ("url", re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"')]+[^\s<>\"').,;]"), 0),
    ("date", re.compile(
        rf"(?i)\b(?:\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}"
        rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{2,4}}"
        rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}})\b"), 0),
    ("money", re.compile(rf"(?i)(?:[$€£¥₦₹]\s?|\b(?:USD|EUR|GBP|NGN|INR|CAD|AUD)\s?)(?:{_NUM})|(?:{_NUM})\s?(?:USD|EUR|GBP|NGN|INR)\b"), 0),
    ("percent", re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?%"), 0),
    ("id", re.compile(
        r"(?i)\b(?:invoice|inv|receipt|order|po|ref(?:erence)?|account|acct|customer|ticket|id)\.?\s*(?:no\.?|number|num|#)?\s*[:#-]?\s*"
        r"([A-Z0-9][A-Z0-9/-]{3,})"), 1),
    ("phone", re.compile(r"(?<![\w/.-])\+?\(?\d{1,4}\)?(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w/-])"), 0),
]
ENTITY_TYPES = [p[0] for p in PATTERNS]


def parse_amount(s: str) -> Optional[float]:
    """Parse ``"$1,234.50"``, ``"1.234,50 EUR"`` or ``"12%"`` to a float (``None`` if impossible)."""
    t = re.sub(r"[^\d.,-]", "", s)
    if not re.search(r"\d", t):
        return None
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".") if re.search(r",\d{1,2}$", t) else t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _parse_date(s: str, dayfirst: bool) -> Optional[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(re.sub(r"(?<=\d)(st|nd|rd|th)", "", s), dayfirst=dayfirst, errors="coerce")
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")


def extract_entities(text: str, dayfirst: bool = True) -> pd.DataFrame:
    """Find emails, URLs, dates, money, percentages, reference IDs and phone numbers.

    Returns a DataFrame ``[type, value, normalized, line_no, context]`` (``normalized`` = ISO date or number).
    """
    taken: List[tuple] = []
    rows = []
    for etype, pat, grp in PATTERNS:
        for m in pat.finditer(text):
            a, b = m.span(grp)
            if any(a < y and x < b for x, y in taken):
                continue
            val = m.group(grp).strip()
            if etype == "id" and not re.search(r"\d", val):
                continue
            if etype == "phone":
                digits = re.sub(r"\D", "", val)
                if not 7 <= len(digits) <= 15 or not re.search(r"[+()\s.-]", val):
                    continue
            taken.append((a, b))
            norm = (_parse_date(val, dayfirst) if etype == "date" else
                    parse_amount(val) if etype in ("money", "percent") else None)
            line_no = text.count("\n", 0, a) + 1
            ls = text.rfind("\n", 0, a) + 1
            le = text.find("\n", a)
            rows.append({"type": etype, "value": val, "normalized": norm, "line_no": line_no,
                         "context": text[ls: le if le != -1 else len(text)].strip()[:140], "_pos": a})
    df = pd.DataFrame(rows, columns=["type", "value", "normalized", "line_no", "context", "_pos"])
    return df.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Key–value fields
# --------------------------------------------------------------------------- #
_KV = [re.compile(r"^\s*([A-Za-z][\w /&()#.%-]{1,40}?)\s*[:=]\s*(\S.*?)\s*$"),
       re.compile(r"^\s*([A-Za-z][\w /&()#.%-]{1,40}?)\s*\.{3,}\s*(\S.*?)\s*$"),
       re.compile(r"^\s*([A-Za-z][\w /&()#.%-]{1,30}?)\s{3,}((?:[$€£¥₦₹]\s?)?\d[\d,.]*\s?%?)\s*$")]


_PLAIN_NUMBER = re.compile(r"[$€£¥₦₹]?\s?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?(?:%|USD|EUR|GBP|NGN|INR)?", re.I)


def extract_key_values(text: str) -> pd.DataFrame:
    """Form-style fields (``Key: value``, ``Key ..... value`` or ``Label   123.45``).

    Returns ``[key, value, number, line_no]``; ``number`` is the parsed numeric value when there is one.
    """
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat in _KV:
            m = pat.match(line)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
                if "   " in val:                       # "Total:   $5   Date: x" -> keep first cell only
                    val = re.split(r"\s{3,}", val)[0]
                rows.append({"key": key, "value": val, "number": parse_amount(val) if _PLAIN_NUMBER.fullmatch(val) else None,
                             "line_no": i})
                break
    return pd.DataFrame(rows, columns=["key", "value", "number", "line_no"])


# --------------------------------------------------------------------------- #
# Tables from word boxes
# --------------------------------------------------------------------------- #
def detect_tables(words: pd.DataFrame, min_rows: int = 3, gap_factor: float = 1.5) -> List[pd.DataFrame]:
    """Reconstruct tables from OCR word boxes.

    Rows are visual lines; cells are split at wide horizontal gaps; consecutive multi-cell rows form a table whose
    columns are the horizontal bands covered by the cells.  Works for ruled and unruled tables with clear spacing.
    """
    if words is None or words.empty:
        return []
    mh = float(words["height"].median())
    lines = group_lines(words)
    cells = [line_cells(l, gap_factor * mh) for l in lines]
    cy = [float((l["top"] + l["height"] / 2).mean()) for l in lines]
    tables, run = [], []

    def flush():
        if len(run) >= min_rows:
            t = _run_to_table([cells[i] for i in run])
            if t is not None:
                tables.append(t)

    for i, c in enumerate(cells):
        if len(c) >= 2 and (not run or cy[i] - cy[run[-1]] < 3.2 * mh):
            run.append(i)
        else:
            flush()
            run = [i] if len(c) >= 2 else []
    flush()
    return tables


def _run_to_table(rows: List[List[tuple]]) -> Optional[pd.DataFrame]:
    spans = sorted((x0, x1) for r in rows for x0, x1, _ in r)
    cols: List[List[float]] = []
    for x0, x1 in spans:
        if cols and x0 <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], x1)
        else:
            cols.append([x0, x1])
    if len(cols) < 2:
        return None
    grid = [[""] * len(cols) for _ in rows]
    for ri, r in enumerate(rows):
        for x0, x1, txt in r:
            mid = (x0 + x1) / 2
            ci = next((k for k, (a, b) in enumerate(cols) if a <= mid <= b), 0)
            grid[ri][ci] = (grid[ri][ci] + " " + txt).strip()
    df = pd.DataFrame(grid)
    df = df.loc[:, (df != "").any(axis=0)]
    if df.shape[1] < 2:
        return None
    numeric = lambda s: parse_amount(s) is not None and not re.search(r"[A-Za-z]{2,}", s)
    first_numeric = any(numeric(v) for v in df.iloc[0] if v)
    rest_numeric = any(numeric(v) for v in df.iloc[1:].to_numpy().ravel() if v)
    if not first_numeric and rest_numeric:
        names, seen = [], Counter()
        for k, v in enumerate(df.iloc[0], 1):
            n = v or f"col_{k}"
            seen[n] += 1
            names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
        df = df.iloc[1:].copy()
        df.columns = names
    else:
        df.columns = [f"col_{k}" for k in range(1, df.shape[1] + 1)]
    return df.reset_index(drop=True)


def tables_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert table columns that are mostly numbers/amounts to float (keeps others as text)."""
    out = df.copy()
    for c in out.columns:
        vals = out[c].map(lambda v: parse_amount(str(v)) if not re.search(r"[A-Za-z]{2,}", str(v)) else None)
        if vals.notna().mean() >= 0.6:
            out[c] = vals
    return out


# --------------------------------------------------------------------------- #
# Keywords, statistics, numbers
# --------------------------------------------------------------------------- #
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if in into is it
its just me more most my no nor not now of off on once only or other our out over own same she should so some such than that the their
them then there these they this those through to too under until up very was we were what when where which while who whom why will with
would you your yours per via etc""".split())


def keywords(text: str, top_n: int = 15, max_len: int = 4) -> pd.DataFrame:
    """RAKE-style key phrases: ``[phrase, score, count]``."""
    phrases: List[List[str]] = []
    for sent in re.split(r"[.!?\n;:,()\[\]|]+", text.lower()):
        cur: List[str] = []
        for w in re.findall(r"[a-z][a-z'-]*[a-z]|[a-z]|\d+(?:\.\d+)?", sent):
            if w in STOPWORDS or len(w) < 2 or w[0].isdigit():      # stop-words and numbers delimit phrases
                if cur:
                    phrases.append(cur)
                cur = []
            else:
                cur.append(w)
        if cur:
            phrases.append(cur)
    phrases = [p for p in phrases if len(p) <= max_len]
    freq, deg = Counter(), Counter()
    for p in phrases:
        for w in p:
            freq[w] += 1
            deg[w] += len(p)
    counts = Counter(" ".join(p) for p in phrases)
    rows = [{"phrase": ph, "score": sum(deg[w] / freq[w] for w in ph.split()), "count": n} for ph, n in counts.items()]
    df = pd.DataFrame(rows, columns=["phrase", "score", "count"])
    return df.sort_values(["score", "count"], ascending=False).head(top_n).reset_index(drop=True)


def word_frequencies(text: str, top_n: int = 20) -> pd.DataFrame:
    """Most frequent non-stop-words: ``[word, count]``."""
    c = Counter(w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower()) if w not in STOPWORDS)
    return pd.DataFrame(c.most_common(top_n), columns=["word", "count"])


def _syllables(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def text_stats(text: str, mean_conf: float = float("nan")) -> Dict[str, float]:
    """Counts and readability of a document."""
    words = re.findall(r"[A-Za-z0-9'’-]+", text)
    sents = [s for s in re.split(r"[.!?]+\s|\n{2,}", text) if s.strip()]
    letters = [w for w in words if w.isalpha()]
    nw, ns = max(len(words), 1), max(len(sents), 1)
    syl = sum(_syllables(w) for w in letters) / max(len(letters), 1)
    return {"characters": len(text), "words": len(words), "lines": len([l for l in text.splitlines() if l.strip()]),
            "sentences": len(sents), "unique words": len({w.lower() for w in words}),
            "avg word length": float(np.mean([len(w) for w in words])) if words else 0.0,
            "numeric tokens %": 100.0 * sum(any(c.isdigit() for c in w) for w in words) / nw,
            "flesch reading ease": 206.835 - 1.015 * (nw / ns) - 84.6 * syl,
            "ocr mean confidence": mean_conf}


def describe_numbers(values: Sequence[float]) -> pd.Series:
    """count / sum / mean / median / std / min / quartiles / p90 / max of numeric values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return pd.Series(dtype=float)
    q = np.percentile(v, [25, 50, 75, 90])
    return pd.Series({"count": v.size, "sum": v.sum(), "mean": v.mean(), "median": q[1], "std": v.std(ddof=1) if v.size > 1 else 0.0,
                      "min": v.min(), "q1": q[0], "q3": q[2], "p90": q[3], "max": v.max()})


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
DOC_TYPES: Dict[str, List[str]] = {
    "invoice": ["invoice", "bill to", "due date", "subtotal", "total", "tax", "amount due", "qty", "unit price", "payment terms"],
    "receipt": ["receipt", "cashier", "change", "thank you", "paid", "card", "visa", "mastercard", "store", "transaction"],
    "resume / CV": ["curriculum vitae", "resume", "experience", "education", "skills", "references", "objective", "employment"],
    "contract / agreement": ["agreement", "party", "parties", "terms and conditions", "hereby", "whereas", "clause", "governing law", "signature", "witness"],
    "letter": ["dear", "sincerely", "regards", "yours faithfully", "yours sincerely", "re:"],
    "bank statement": ["statement", "balance", "debit", "credit", "account number", "opening balance", "closing balance", "transaction"],
    "id / certificate": ["certificate", "date of birth", "nationality", "issued", "expiry", "passport", "licence", "license", "surname"],
    "report / article": ["abstract", "introduction", "conclusion", "figure", "table", "references", "results", "methodology"],
}


def classify_document(text: str) -> pd.DataFrame:
    """Rule-based document type scores: ``[type, matches, share %]`` sorted high→low.

    A transparent keyword heuristic, not a trained model — good for triage of invoices vs. letters etc.
    """
    low = text.lower()
    rows = [{"type": t, "matches": sum(1 for k in kws if k in low)} for t, kws in DOC_TYPES.items()]
    df = pd.DataFrame(rows)
    tot = df["matches"].sum()
    df["share %"] = 100.0 * df["matches"] / tot if tot else 0.0
    return df.sort_values("matches", ascending=False).reset_index(drop=True)


def best_type(scores: pd.DataFrame, min_matches: int = 2) -> str:
    """Top document type, or ``"unknown"`` if evidence is weak."""
    return str(scores.iloc[0]["type"]) if len(scores) and scores.iloc[0]["matches"] >= min_matches else "unknown"


# --------------------------------------------------------------------------- #
# Pattern search
# --------------------------------------------------------------------------- #
def regex_search(text: str, pattern: str, ignore_case: bool = True) -> pd.DataFrame:
    """All matches of a user regex: ``[match, line_no, context]``. Raises ``ValueError`` on a bad pattern."""
    try:
        rx = re.compile(pattern, re.I if ignore_case else 0)
    except re.error as e:
        raise ValueError(f"Invalid regular expression: {e}") from e
    rows = []
    for m in rx.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.start())
        rows.append({"match": m.group(0), "line_no": text.count("\n", 0, m.start()) + 1,
                     "context": text[ls: le if le != -1 else len(text)].strip()[:140]})
    return pd.DataFrame(rows, columns=["match", "line_no", "context"])


# --------------------------------------------------------------------------- #
# Cross-document mining
# --------------------------------------------------------------------------- #
def corpus_mining(docs: Dict[str, str], n_clusters: Optional[int] = None, top_terms: int = 8) -> Dict[str, object]:
    """TF-IDF analysis of several documents.

    Returns ``{"terms": DataFrame[doc, top terms], "similarity": DataFrame (cosine), "clusters": DataFrame[doc, cluster]}``.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    names = list(docs)
    if len(names) < 2:
        raise ValueError("Need at least two documents.")
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b", ngram_range=(1, 2),
                          sublinear_tf=True)
    X = vec.fit_transform([docs[n] for n in names])
    vocab = np.array(vec.get_feature_names_out())
    terms = []
    for i, n in enumerate(names):
        row = X[i].toarray().ravel()
        top = row.argsort()[::-1][:top_terms]
        terms.append({"doc": n, "top terms": ", ".join(vocab[j] for j in top if row[j] > 0)})
    sim = pd.DataFrame(cosine_similarity(X), index=names, columns=names).round(3)
    k = n_clusters or min(3, len(names))
    k = max(1, min(k, len(names)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X) if k > 1 and X.shape[1] > 0 else np.zeros(len(names), int)
    return {"terms": pd.DataFrame(terms), "similarity": sim, "clusters": pd.DataFrame({"doc": names, "cluster": labels})}


def search_docs(docs: Dict[str, str], query: str, top_k: int = 10) -> pd.DataFrame:
    """Rank documents against a free-text query by TF-IDF cosine similarity: ``[doc, score, snippet]``."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    names = list(docs)
    if not query.strip() or not names:
        return pd.DataFrame(columns=["doc", "score", "snippet"])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        X = vec.fit_transform([docs[n] for n in names] + [query])
    except ValueError:                                   # empty vocabulary
        return pd.DataFrame(columns=["doc", "score", "snippet"])
    scores = cosine_similarity(X[-1], X[:-1]).ravel()
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
    rows = []
    for i in np.argsort(scores)[::-1][:top_k]:
        snippet = next((l.strip()[:140] for l in docs[names[i]].splitlines() if any(t in l.lower() for t in terms)), "")
        rows.append({"doc": names[i], "score": round(float(scores[i]), 4), "snippet": snippet})
    return pd.DataFrame(rows)
