"""Synthetic sample documents (for the demo button and tests) — no external fonts or files needed."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1: scalable built-in font
    except TypeError:                                  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def make_sample_invoice() -> Image.Image:
    """A clean 1240×1754 (A4 @150 dpi) invoice with a four-column line-item table."""
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    big, mid, small = _font(54), _font(34), _font(30)
    d.text((100, 90), "ACME Supplies Ltd", font=big, fill="black")
    d.text((100, 170), "12 Market Road, Lagos 100001", font=small, fill="black")
    d.text((860, 100), "INVOICE", font=big, fill="black")
    rows = ["Invoice No: INV-2024-0457", "Date: 15 March 2024", "Due Date: 14/04/2024",
            "Bill To: Jane Doe", "Email: billing@acme-supplies.com", "Phone: +1 (555) 123-4567"]
    for i, t in enumerate(rows):
        d.text((100, 300 + i * 52), t, font=mid, fill="black")
    xs = (100, 620, 800, 1000)
    table = [("Item", "Qty", "Unit Price", "Amount"), ("Paper A4 box", "10", "$12.50", "$125.00"),
             ("Ink cartridge", "4", "$45.00", "$180.00"), ("Stapler", "2", "$8.75", "$17.50")]
    y = 700
    d.line((100, y - 12, 1140, y - 12), fill="black", width=3)
    for r in table:
        for x, c in zip(xs, r):
            d.text((x, y), c, font=mid, fill="black")
        y += 62
    d.line((100, y, 1140, y), fill="black", width=3)
    for label, val in (("Subtotal:", "$322.50"), ("Tax (10%):", "$32.25"), ("Total:", "$354.75")):
        y += 62
        d.text((800, y), label, font=mid, fill="black")
        d.text((1000, y), val, font=mid, fill="black")
    d.text((100, 1500), "Thank you for your business.", font=mid, fill="black")
    d.text((100, 1560), "Payment terms: net 30 days. Visit www.acme-supplies.com", font=mid, fill="black")
    return img


def make_photo(page: Image.Image, angle: float = 3.0, seed: int = 0) -> Image.Image:
    """Simulate a phone photo: page on a dark desk, perspective, tilt, uneven light, sensor noise."""
    rng = np.random.default_rng(seed)
    page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    h, w = page_np.shape[:2]
    canvas_w, canvas_h = int(w * 1.25), int(h * 1.2)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    ox, oy = (canvas_w - w) / 2, (canvas_h - h) / 2
    dst = np.float32([[ox + 40, oy + 20], [ox + w - 30, oy - 10], [ox + w + 10, oy + h + 15], [ox - 15, oy + h - 30]])
    m = cv2.getPerspectiveTransform(src, dst)
    bg = np.full((canvas_h, canvas_w, 3), (45, 60, 75), np.uint8)
    warped = cv2.warpPerspective(page_np, m, (canvas_w, canvas_h), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    rot = cv2.getRotationMatrix2D((canvas_w / 2, canvas_h / 2), angle, 1.0)
    warped = cv2.warpAffine(warped, rot, (canvas_w, canvas_h), borderValue=(45, 60, 75))
    gx = np.linspace(0.65, 1.05, canvas_w, dtype=np.float32)[None, :, None]       # light falls off to the left
    out = np.clip(warped.astype(np.float32) * gx + rng.normal(0, 5, warped.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
