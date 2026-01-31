import streamlit as st

st.title("Dobble Bot")
st.write("Upload images → generate a Dobble-style deck (coming next).")

files = st.file_uploader(
    "Upload images (PNG/JPG). Tip: 10–30 is great to start.",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if files:
    st.success(f"Loaded {len(files)} images.")


import io
import math
import random
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageOps

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


# ----------------------------
# Dobble math (prime order n)
# ----------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for k in range(3, r + 1, 2):
        if n % k == 0:
            return False
    return True


def best_prime_order_for_symbols(num_symbols: int) -> Optional[int]:
    """
    Return largest prime n such that n^2 + n + 1 <= num_symbols.
    If none fits, return None.
    """
    best = None
    # n^2 + n + 1 <= S => n approx sqrt(S)
    max_guess = int(math.sqrt(max(0, num_symbols))) + 2
    for n in range(2, max_guess + 1):
        if is_prime(n) and (n * n + n + 1) <= num_symbols:
            best = n
    return best


def generate_projective_plane_deck(n: int) -> List[List[int]]:
    """
    Finite projective plane of order n (prime n):
    - v = n^2 + n + 1 symbols (0..v-1)
    - v cards (blocks)
    - each card has k = n+1 symbols
    - any two cards share exactly one symbol

    Returns list of cards, each card list of symbol indices.
    """
    if not is_prime(n):
        raise ValueError("n must be prime for this implementation.")

    v = n * n + n + 1
    cards: List[List[int]] = []

    # We construct using a standard representation:
    # Symbols correspond to:
    #  - one "infinity" symbol: INF
    #  - n "slope infinity" symbols: S_m (m=0..n-1)
    #  - n^2 point symbols: P_(x,y) with x,y in GF(n)

    INF = 0
    S = [1 + m for m in range(n)]  # 1..n
    # Points start at index 1+n
    def P(x: int, y: int) -> int:
        return 1 + n + x * n + y  # 1+n .. 1+n+n^2-1

    # Cards:
    # 1) The "line at infinity": {INF} U {S_m}
    cards.append([INF] + S)

    # 2) For each slope m and intercept b: line y = m x + b
    # card: {S_m} U {P(x, m x + b)}
    for m in range(n):
        for b in range(n):
            card = [S[m]]
            for x in range(n):
                y = (m * x + b) % n
                card.append(P(x, y))
            cards.append(card)

    # 3) Vertical lines x = a
    # card: {INF} U {P(a, y) for y}
    for a in range(n):
        card = [INF]
        for y in range(n):
            card.append(P(a, y))
        cards.append(card)

    if len(cards) != v:
        raise RuntimeError("Construction error: wrong number of cards.")
    # Sanity: each card length n+1
    if any(len(c) != n + 1 for c in cards):
        raise RuntimeError("Construction error: wrong card size.")
    return cards


def verify_dobble_property(cards: List[List[int]]) -> Tuple[bool, str]:
    """
    Check: any two distinct cards share exactly one symbol.
    """
    sets = [set(c) for c in cards]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i].intersection(sets[j])
            if len(inter) != 1:
                return False, f"Failed at pair ({i},{j}): intersection size {len(inter)}"
    return True, "OK"


# ----------------------------
# Rendering (circle cards)
# ----------------------------

@dataclass
class RenderProfile:
    name: str
    # how busy / big icons should be:
    min_scale: float
    max_scale: float
    max_rotation_deg: int
    padding_ratio: float      # space from edge of circle (0..0.3)
    overlap_tries: int


PROFILES = {
    "K1 (bigger, fewer overlaps)": RenderProfile(
        name="K1",
        min_scale=0.22,
        max_scale=0.34,
        max_rotation_deg=18,
        padding_ratio=0.10,
        overlap_tries=300,
    ),
    "K2 (balanced)": RenderProfile(
        name="K2",
        min_scale=0.18,
        max_scale=0.30,
        max_rotation_deg=30,
        padding_ratio=0.08,
        overlap_tries=450,
    ),
    "K3 (denser, more challenge)": RenderProfile(
        name="K3",
        min_scale=0.16,
        max_scale=0.28,
        max_rotation_deg=40,
        padding_ratio=0.06,
        overlap_tries=650,
    ),
}


def load_images(uploaded_files) -> List[Image.Image]:
    imgs = []
    for f in uploaded_files:
        im = Image.open(f).convert("RGBA")
        # Auto-trim + add a little margin if needed (helps with stickers)
        im = ImageOps.exif_transpose(im)
        imgs.append(im)
    return imgs


def circle_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = Image.new("L", (size, size), 0)
    # fast circle via alpha composite
    # (we can just draw using ImageDraw)
    from PIL import ImageDraw
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    return mask


def bbox_intersect(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def place_symbols_on_circle(
    symbol_imgs: List[Image.Image],
    card_symbols: List[int],
    size_px: int,
    profile: RenderProfile,
    rng: random.Random,
) -> Image.Image:
    """
    Create a transparent PNG sized size_px x size_px with a circular cutout and placed symbols.
    """
    base = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    mask = circle_mask(size_px)

    placed_boxes: List[Tuple[int,int,int,int]] = []

    # circle center/radius with padding
    cx, cy = size_px / 2, size_px / 2
    radius = (size_px / 2) * (1.0 - profile.padding_ratio)

    for s in card_symbols:
        src = symbol_imgs[s]

        # try multiple placements
        ok = False
        for _ in range(profile.overlap_tries):
            scale = rng.uniform(profile.min_scale, profile.max_scale)
            target_w = int(size_px * scale)
            # keep aspect
            w, h = src.size
            target_h = max(1, int(target_w * (h / w)))
            icon = src.resize((target_w, target_h), Image.LANCZOS)

            rot = rng.uniform(-profile.max_rotation_deg, profile.max_rotation_deg)
            icon = icon.rotate(rot, expand=True, resample=Image.BICUBIC)

            iw, ih = icon.size

            # sample a random point within the circle (rejection sampling)
            for _pt in range(25):
                angle = rng.random() * 2 * math.pi
                r = radius * math.sqrt(rng.random())
                x = cx + r * math.cos(angle) - iw / 2
                y = cy + r * math.sin(angle) - ih / 2

                x = int(round(x))
                y = int(round(y))

                box = (x, y, x + iw, y + ih)

                # keep within image bounds
                if box[0] < 0 or box[1] < 0 or box[2] > size_px or box[3] > size_px:
                    continue

                # quick overlap check
                if any(bbox_intersect(box, pb) for pb in placed_boxes):
                    continue

                # also ensure icon mostly sits in circle (approx: check corners)
                corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
                inside = 0
                for px, py in corners:
                    dx = (px - cx)
                    dy = (py - cy)
                    if dx*dx + dy*dy <= radius*radius:
                        inside += 1
                if inside < 3:
                    continue

                # paste
                base.alpha_composite(icon, (x, y))
                placed_boxes.append(box)
                ok = True
                break

            if ok:
                break

        # if we fail to place without overlap, just place it anyway (still playable)
        if not ok:
            icon = src.copy()
            scale = profile.min_scale
            target_w = int(size_px * scale)
            w, h = icon.size
            target_h = max(1, int(target_w * (h / w)))
            icon = icon.resize((target_w, target_h), Image.LANCZOS)
            base.alpha_composite(icon, (int(cx - icon.size[0]/2), int(cy - icon.size[1]/2)))

    # apply circular mask to keep edge clean
    circ = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    circ.paste(base, (0, 0), mask=mask)
    return circ


# ----------------------------
# PDF export (A4 sheets)
# ----------------------------

def cards_to_a4_pdf_bytes(
    card_images: List[Image.Image],
    cards_per_page: int,
    circle_diameter_mm: float,
    margin_mm: float,
) -> bytes:
    """
    Put circles on A4 pages in a simple grid and export as PDF.
    """
    page_w, page_h = A4  # points
    diam_pt = circle_diameter_mm * mm
    margin_pt = margin_mm * mm

    # grid dims: choose rows/cols from cards_per_page
    cols = int(math.ceil(math.sqrt(cards_per_page)))
    rows = int(math.ceil(cards_per_page / cols))

    # spacing
    usable_w = page_w - 2 * margin_pt
    usable_h = page_h - 2 * margin_pt

    # step sizes
    step_x = usable_w / cols
    step_y = usable_h / rows

    # ensure circle fits in cells
    max_diam = min(step_x, step_y) * 0.92
    diam = min(diam_pt, max_diam)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    idx = 0
    while idx < len(card_images):
        for r in range(rows):
            for col in range(cols):
                if idx >= len(card_images) or (r * cols + col) >= cards_per_page:
                    break

                # center of the cell
                x0 = margin_pt + col * step_x + (step_x - diam) / 2
                y0 = page_h - (margin_pt + (r + 1) * step_y) + (step_y - diam) / 2

                # Convert PIL image to PNG bytes for reportlab
                png_buf = io.BytesIO()
                card_images[idx].save(png_buf, format="PNG")
                png_buf.seek(0)

                c.drawImage(
                    ImageReader(png_buf),
                    x0, y0,
                    width=diam, height=diam,
                    mask='auto'
                )
                idx += 1

        c.showPage()

    c.save()
    return buf.getvalue()


# reportlab helper
from reportlab.lib.utils import ImageReader


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Dobble Bot (K1/K2/K3)", layout="wide")

st.title("Dobble Bot 🎴 (K1 / K2 / K3)")
st.caption("Upload images → preview → generate Dobble deck → download circle PNGs + print-ready PDF")

with st.expander("How it works (and why image count matters)", expanded=False):
    st.write(
        """
A true Dobble deck is built from a projective plane (prime order **n**):
- Symbols needed: **n² + n + 1**
- Cards produced: **n² + n + 1**
- Symbols per card: **n + 1**
- Any two cards share **exactly one** symbol.

This app automatically picks the largest valid **prime n** that fits your uploaded image count.
        """
    )

colA, colB = st.columns([1.1, 0.9])

with colA:
    uploaded = st.file_uploader(
        "1) Upload your symbol images (PNG/JPG). Transparent PNG works best.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

with colB:
    profile_name = st.selectbox("Class profile", list(PROFILES.keys()), index=1)
    profile = PROFILES[profile_name]

    size_px = st.slider("Card export size (px)", 800, 2400, 1400, step=100)
    rng_seed = st.number_input("Random seed (same seed = same layout)", value=1234, step=1)

    st.markdown("**PDF settings**")
    cards_per_page = st.selectbox("Cards per A4 page", [6, 8, 9, 12], index=2)
    circle_diam_mm = st.slider("Circle diameter on A4 (mm)", 55, 95, 80, step=1)
    margin_mm = st.slider("Page margin (mm)", 5, 20, 10, step=1)

st.divider()

if not uploaded:
    st.info("Upload images to begin. Tip: start with 13 images to generate a full n=3 deck.")
    st.stop()

# Load images
imgs_raw = load_images(uploaded)
num_uploaded = len(imgs_raw)

# Determine best n
best_n = best_prime_order_for_symbols(num_uploaded)
if best_n is None:
    st.error(
        f"You uploaded {num_uploaded} images. The smallest true deck is n=2, which needs 7 images. "
        "Please upload at least 7 images."
    )
    st.stop()

needed_symbols = best_n * best_n + best_n + 1

st.subheader("2) Preview")
st.write(f"Uploaded images: **{num_uploaded}**")
st.write(f"Best valid Dobble order (prime n): **{best_n}** → needs **{needed_symbols}** images")

# Let user override n downward (still valid) if they want fewer cards
valid_ns = [n for n in range(2, best_n + 1) if is_prime(n) and (n*n + n + 1) <= num_uploaded]
chosen_n = st.selectbox("Deck size (choose a smaller n for fewer cards)", valid_ns, index=len(valid_ns)-1)
v = chosen_n * chosen_n + chosen_n + 1
k = chosen_n + 1

st.write(f"Selected: **n={chosen_n}** → **{v} cards**, **{k} symbols per card**")

# preview images (first 18)
preview_count = min(num_uploaded, 18)
prev_cols = st.columns(6)
for i in range(preview_count):
    with prev_cols[i % 6]:
        st.image(imgs_raw[i], caption=f"#{i}", use_container_width=True)

if num_uploaded < v:
    st.error(f"Not enough images for n={chosen_n}. Need {v}, you have {num_uploaded}.")
    st.stop()

# Use first v images as symbols (or you could shuffle; we keep deterministic order + seed)
# Make them square-ish sticker cutouts by fitting to canvas (optional)
def normalize_symbol(im: Image.Image, pad: int = 16) -> Image.Image:
    # put symbol on transparent canvas with padding
    w, h = im.size
    canvas_size = max(w, h) + 2 * pad
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    out.alpha_composite(im, ((canvas_size - w) // 2, (canvas_size - h) // 2))
    return out

symbols = [normalize_symbol(im) for im in imgs_raw[:v]]

st.divider()
st.subheader("3) Generate")

generate = st.button("Generate Dobble deck", type="primary")

if not generate:
    st.stop()

rng = random.Random(int(rng_seed))

# Generate deck
cards = generate_projective_plane_deck(chosen_n)
ok, msg = verify_dobble_property(cards)
if not ok:
    st.error("Deck generation failed verification: " + msg)
    st.stop()

st.success(f"Deck generated and verified ✅  ({v} cards, {k} symbols each)")

# Render cards
with st.spinner("Rendering circle card PNGs..."):
    rendered_cards: List[Image.Image] = []
    # shuffle card order for variety (seeded)
    order = list(range(len(cards)))
    rng.shuffle(order)

    for idx in order:
        card_symbols = cards[idx]
        # Shuffle symbol positions on card for variety (seeded)
        card_symbols = card_symbols[:]
        rng.shuffle(card_symbols)

        img = place_symbols_on_circle(
            symbol_imgs=symbols,
            card_symbols=card_symbols,
            size_px=size_px,
            profile=profile,
            rng=rng
        )
        rendered_cards.append(img)

st.subheader("Preview generated cards")
preview_gen = min(len(rendered_cards), 12)
gen_cols = st.columns(6)
for i in range(preview_gen):
    with gen_cols[i % 6]:
        st.image(rendered_cards[i], caption=f"Card {i+1}", use_container_width=True)

# Prepare downloads
st.divider()
st.subheader("4) Download")

# ZIP of PNGs
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for i, im in enumerate(rendered_cards, start=1):
        b = io.BytesIO()
        im.save(b, format="PNG")
        z.writestr(f"dobble_card_{i:03d}.png", b.getvalue())
zip_buf.seek(0)

st.download_button(
    "Download circle cards (PNG ZIP)",
    data=zip_buf.getvalue(),
    file_name=f"dobble_cards_n{chosen_n}_png.zip",
    mime="application/zip"
)

# PDF
with st.spinner("Building A4 PDF..."):
    # For PDF we keep current order of rendered_cards
    # Build PDF bytes using reportlab
    # Note: reportlab uses points; we draw the PNG circles on a grid.
    pdf_bytes = None
    try:
        pdf_bytes = build_a4_pdf(rendered_cards, cards_per_page, circle_diam_mm, margin_mm)
    except NameError:
        # If helper not defined due to import ordering issues
        pass

# Because we used ImageReader in the PDF function, we implement a local builder cleanly here:
def build_a4_pdf(card_images: List[Image.Image], cards_per_page: int, circle_diameter_mm: float, margin_mm: float) -> bytes:
    page_w, page_h = A4
    diam_pt = circle_diameter_mm * mm
    margin_pt = margin_mm * mm

    cols = int(math.ceil(math.sqrt(cards_per_page)))
    rows = int(math.ceil(cards_per_page / cols))

    usable_w = page_w - 2 * margin_pt
    usable_h = page_h - 2 * margin_pt
    step_x = usable_w / cols
    step_y = usable_h / rows
    max_diam = min(step_x, step_y) * 0.92
    diam = min(diam_pt, max_diam)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    idx = 0
    while idx < len(card_images):
        for r in range(rows):
            for col in range(cols):
                if idx >= len(card_images) or (r * cols + col) >= cards_per_page:
                    break

                x0 = margin_pt + col * step_x + (step_x - diam) / 2
                y0 = page_h - (margin_pt + (r + 1) * step_y) + (step_y - diam) / 2

                png_buf = io.BytesIO()
                card_images[idx].save(png_buf, format="PNG")
                png_buf.seek(0)

                c.drawImage(ImageReader(png_buf), x0, y0, width=diam, height=diam, mask='auto')
                idx += 1

        c.showPage()

    c.save()
    return buf.getvalue()

pdf_bytes = build_a4_pdf(rendered_cards, int(cards_per_page), float(circle_diam_mm), float(margin_mm))

st.download_button(
    "Download print-ready A4 PDF",
    data=pdf_bytes,
    file_name=f"dobble_cards_n{chosen_n}_A4.pdf",
    mime="application/pdf"
)

st.caption("Tip: For perfect cutting, print at 100% scale (no 'fit to page').")
