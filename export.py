"""Export helpers: Excel workbook, JSON, Markdown report."""
from __future__ import annotations

import io
import json
from typing import Dict

import numpy as np
import pandas as pd


def _safe_sheet(name: str, used: set) -> str:
    base = "".join(c for c in name if c not in "[]:*?/\\")[:28] or "sheet"
    out, i = base, 2
    while out in used:
        out = f"{base[:25]}_{i}"
        i += 1
    used.add(out)
    return out


def mining_sheets(name: str, text: str, mined: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    """Collect one document's mining results into named DataFrames (for Excel / CSV)."""
    stats = pd.DataFrame({"metric": list(mined["stats"]), "value": list(mined["stats"].values())})
    summary = pd.concat([pd.DataFrame({"metric": ["document", "type"], "value": [name, mined["doc_type"]]}), stats])
    sheets = {"summary": summary, "entities": mined["entities"], "key_values": mined["key_values"],
              "keywords": mined["keywords"], "word_freq": mined["word_freq"], "classification": mined["classification"]}
    ms = mined["money_summary"]
    if len(ms):
        sheets["money_summary"] = ms.rename("value").rename_axis("statistic").reset_index()
    for i, (pg, t) in enumerate(mined["tables"], 1):
        sheets[f"table_{i}_p{pg}"] = t
    sheets["text"] = pd.DataFrame({"line": text.splitlines()})
    return sheets


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write DataFrames to an .xlsx workbook (sheet names sanitised and de-duplicated)."""
    buf, used = io.BytesIO(), set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for n, df in sheets.items():
            (df if len(df.columns) else pd.DataFrame({"(empty)": []})).to_excel(xw, sheet_name=_safe_sheet(n, used), index=False)
    return buf.getvalue()


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records"))
    if isinstance(o, pd.Series):
        return json.loads(o.to_json())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(type(o))


def to_json_bytes(name: str, text: str, mined: Dict[str, object]) -> bytes:
    """Serialise the full mining result of a document to JSON."""
    payload = {"document": name, "doc_type": mined["doc_type"], "stats": mined["stats"],
               "entities": mined["entities"], "key_values": mined["key_values"],
               "tables": [{"page": p, "rows": t} for p, t in mined["tables"]], "keywords": mined["keywords"],
               "money_summary": mined["money_summary"], "text": text}
    return json.dumps(payload, default=_jsonable, indent=2, ensure_ascii=False).encode("utf-8")


def report_markdown(name: str, text: str, mined: Dict[str, object]) -> str:
    """Human-readable Markdown report for one document."""
    def md(df, n=25):
        if df is None or not len(df):
            return "_none found_"
        d = df.head(n)
        head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns)
        body = "\n".join("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |"
                         for r in d.itertuples(index=False))
        return head + "\n" + body
    s = mined["stats"]
    lines = [f"# Document report — {name}", "", f"**Detected type:** {mined['doc_type']}", "",
             "## Statistics", "", *[f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in s.items()], "",
             "## Entities", "", md(mined["entities"]), "", "## Key–value fields", "", md(mined["key_values"]), ""]
    for i, (pg, t) in enumerate(mined["tables"], 1):
        lines += [f"## Table {i} (page {pg})", "", md(t), ""]
    if len(mined["money_summary"]):
        lines += ["## Money amounts — summary", "", md(mined["money_summary"].rename("value").rename_axis("statistic").reset_index()), ""]
    lines += ["## Key phrases", "", md(mined["keywords"]), "", "## Extracted text", "", "```", text, "```"]
    return "\n".join(lines)
