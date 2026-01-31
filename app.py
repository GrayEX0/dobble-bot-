import io
import math
import random
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


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
    max_guess = int(math.sqrt(max(0, num_symbols))) + 2
    for n in range(2, max_guess + 1):
        if is_prime(n) and (n * n + n + 1) <= num_symbols:
            best = n
    return best


def generate_projective_plane_deck(n: int) -> List[List[int]]:
    """
    Finite projective plane of order n (prime n):
    v = n^2 + n + 1 symbols, v cards, k = n+1 symbols per card
    any two cards share exactly one symbol.
    """
    if not is_prime(n):
        raise ValueError("n must be prime for this implementation.")

    v = n * n + n + 1
    cards: List[List[int]] = []

    INF = 0
    S = [1 + m for m in range(n)]  # 1..n

    def P(x: int, y: int) -> int:
        return 1 + n + x * n + y  # 1+n .. 1+n+n^2-1

    cards.append([INF] + S)

    for m in range(n):
        for b in range(n):
            card = [S[m]]
            for x in range(n):
                y = (m * x + b) % n
                card.append(P(x, y))
            cards.append(card)

    for a in range(n):
        card = [INF]
        for y in range(n):
            card.append(P(a, y))
        cards.append(card)

    if len(cards) != v:
        raise RuntimeError("Construction error: wrong number of cards.")
    if any(len(c) != n + 1 for c in cards):
        raise RuntimeError("Construction error: wrong card size.")
    return cards


def verify_dobble_property(cards: List[List[int]]) -> Tuple[bool, str]:
    sets = [set(c) for c in cards]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i].intersection(sets[j])
            if len(inter) != 1:
                return False, f"Failed at pair ({i},{j}): intersection size {len(inter)}"
    return True, "OK"


# ----------------------------
# Rendering helpers
# ----------------------------

@dataclass
class RenderProfile:
    name: str
    min_scale: float
    max_scale: float
    max_rotation_deg: int
    padding_ratio: float
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
        im = ImageOps.exif_transpose(im)
        imgs.append(im)
    return imgs


def circle_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    return mask


def bbox_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def normalize_symbol(im: Image.Image, pad: int = 16) -> Image.Image:
    w, h = im.size
    canvas_size = max(w, h) + 2 * pad
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    out.alpha_composite(im, ((canvas_size - w) // 2, (canvas_size - h) // 2))
    return out


def make_text_symbol(
    text: str,
    size_px: int = 512,
    font_size: int = 90,
    pad: int = 24,
    rounded: int = 28,
) -> Image.Image:
    """
    Create a transparent RGBA 'symbol' containing the text.
    """
    text = text.strip()
    if not text:
        return Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))

    # Try a default font; PIL will fall back if unavailable
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # measure
    tmp = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    w = min(size_px, tw + 2 * pad)
    h = min(size_px, th + 2 * pad)

    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # soft white label so it reads on busy backgrounds
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=rounded, fill=(255, 255, 255, 220))
    # text
    tx = (w - tw) // 2 - bbox[0]
    ty = (h - th) // 2 - bbox[1]
    d.text((tx, ty), text, font=font, fill=(0, 0, 0, 255))

    # put label onto square canvas so scaling behaves like other images
    canvas_size = max(w, h)
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    out.alpha_composite(im, ((canvas_size - w) // 2, (canvas_size - h) // 2))
    return out


def draw_cut_outline(card: Image.Image, shape: str, outline_px: int) -> Image.Image:
    """
    Draw a visible cut outline on the card image (non-destructive, keeps transparency).
    """
    if outline_px <= 0:
        return card

    out = card.copy()
    d = ImageDraw.Draw(out)

    w, h = out.size
    inset = outline_px // 2 + 2

    if shape == "Circle":
        d.ellipse((inset, inset, w - inset - 1, h - inset - 1), outline=(0, 0, 0, 255), width=outline_px)
    else:
        d.rectangle((inset, inset, w - inset - 1, h - inset - 1), outline=(0, 0, 0, 255), width=outline_px)

    return out


def place_symbols_on_card(
    symbol_imgs: List[Image.Image],
    card_symbols: List[int],
    size_px: int,
    profile: RenderProfile,
    rng: random.Random,
    shape: str,
) -> Image.Image:
    """
    Create a transparent PNG sized size_px x size_px with placed symbols.
    If shape == Circle, applies circular mask. If Square, keeps square.
    """
    base = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))

    # geometry for placement
    cx, cy = size_px / 2, size_px / 2
    if shape == "Circle":
        radius = (size_px / 2) * (1.0 - profile.padding_ratio)
    else:
        # treat square as inscribed circle-ish for placement comfort
        radius = (size_px / 2) * (1.0 - profile.padding_ratio)

    placed_boxes: List[Tuple[int, int, int, int]] = []

    for s in card_symbols:
        src = symbol_imgs[s]
        ok = False

        for _ in range(profile.overlap_tries):
            scale = rng.uniform(profile.min_scale, profile.max_scale)
            target_w = max(1, int(size_px * scale))

            w, h = src.size
            target_h = max(1, int(target_w * (h / w)))
            icon = src.resize((target_w, target_h), Image.LANCZOS)

            rot = rng.uniform(-profile.max_rotation_deg, profile.max_rotation_deg)
            icon = icon.rotate(rot, expand=True, resample=Image.BICUBIC)

            iw, ih = icon.size

            for _pt in range(25):
                angle = rng.random() * 2 * math.pi
                r = radius * math.sqrt(rng.random())
                x = cx + r * math.cos(angle) - iw / 2
                y = cy + r * math.sin(angle) - ih / 2

                x = int(round(x))
                y = int(round(y))

                box = (x, y, x + iw, y + ih)

                if box[0] < 0 or box[1] < 0 or box[2] > size_px or box[3] > size_px:
                    continue

                if any(bbox_intersect(box, pb) for pb in placed_boxes):
                    continue

                # keep mostly inside circle if Circle shape
                if shape == "Circle":
                    corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
                    inside = 0
                    for px, py in corners:
                        dx = (px - cx)
                        dy = (py - cy)
                        if dx * dx + dy * dy <= radius * radius:
                            inside += 1
                    if inside < 3:
                        continue

                base.alpha_composite(icon, (x, y))
                placed_boxes.append(box)
                ok = True
                break

            if ok:
                break

        if not ok:
            icon = src.copy()
            scale = profile.min_scale
            target_w = max(1, int(size_px * scale))
            w, h = icon.size
            target_h = max(1, int(target_w * (h / w)))
            icon = icon.resize((target_w, target_h), Image.LANCZOS)
            base.alpha_composite(icon, (int(cx - icon.size[0] / 2), int(cy - icon.size[1] / 2)))

    # Apply mask if circle
    if shape == "Circle":
        mask = circle_mask(size_px)
        circ = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
        circ.paste(base, (0, 0), mask=mask)
        return circ

    return base


# ----------------------------
# PDF export (A4 sheets with cut lines)
# ----------------------------

def build_a4_pdf(
    card_images: List[Image.Image],
    cards_per_page: int,
    card_diameter_mm: float,
    margin_mm: float,
    shape: str,
    cutline_pt: float = 1.0,
) -> bytes:
    page_w, page_h = A4
    size_pt = card_diameter_mm * mm
    margin_pt = margin_mm * mm

    cols = int(math.ceil(math.sqrt(cards_per_page)))
    rows = int(math.ceil(cards_per_page / cols))

    usable_w = page_w - 2 * margin_pt
    usable_h = page_h - 2 * margin_pt
    step_x = usable_w / cols
    step_y = usable_h / rows

    max_size = min(step_x, step_y) * 0.92
    size_pt = min(size_pt, max_size)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    idx = 0
    while idx < len(card_images):
        # set cut line style
        c.setLineWidth(cutline_pt)

        for r in range(rows):
            for col in range(cols):
                if idx >= len(card_images) or (r * cols + col) >= cards_per_page:
                    break

                x0 = margin_pt + col * step_x + (step_x - size_pt) / 2
                y0 = page_h - (margin_pt + (r + 1) * step_y) + (step_y - size_pt) / 2

                # Draw cut outline (vector)
                if shape == "Circle":
                    c.circle(x0 + size_pt / 2, y0 + size_pt / 2, size_pt / 2)
                else:
                    c.rect(x0, y0, size_pt, size_pt)

                # Draw image
                png_buf = io.BytesIO()
                card_images[idx].save(png_buf, format="PNG")
                png_buf.seek(0)

                c.drawImage(ImageReader(png_buf), x0, y0, width=size_pt, height=size_pt, mask="auto")

                idx += 1

        c.showPage()

    c.save()
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Dobble Bot (K1/K2/K3)", layout="wide")
st.title("Dobble Bot 🎴 (K1 / K2 / K3)")
st.caption("Upload images + optional words → preview → generate → download PNG ZIP + A4 PDF (with cut lines)")

with st.expander("How it works (and why image/word count matters)", expanded=False):
    st.write(
        """
A true Dobble deck is built from a projective plane (prime order **n**):
- Symbols needed: **n² + n + 1**
- Cards produced: **n² + n + 1**
- Symbols per card: **n + 1**
- Any two cards share **exactly one** symbol

This app treats **uploaded images + typed words** as the symbol pool.
It automatically picks the largest valid prime **n** that fits the total symbol count.
        """
    )

colA, colB = st.columns([1.15, 0.85])

with colA:
    uploaded = st.file_uploader(
        "1) Upload symbol images (PNG/JPG). Transparent PNG works best.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    words_text = st.text_area(
        "Optional: Add words as extra symbols (one per line)",
        placeholder="cat\nsun\nrainbow\n...",
        height=140
    )

with colB:
    profile_name = st.selectbox("Class profile", list(PROFILES.keys()), index=1)
    profile = PROFILES[profile_name]

    shape = st.selectbox("Card shape", ["Circle", "Square"], index=0)

    size_px = st.slider("Card export size (px)", 800, 2400, 1400, step=100)

    outline_px = st.slider("Cut outline thickness on PNG (px)", 0, 20, 6, step=1)

    rng_seed = st.number_input("Random seed (same seed = same layout)", value=1234, step=1)

    st.markdown("**PDF settings**")
    cards_per_page = st.selectbox("Cards per A4 page", [6, 8, 9, 12], index=2)
    card_size_mm = st.slider("Card size on A4 (mm)", 55, 95, 80, step=1)
    margin_mm = st.slider("Page margin (mm)", 5, 20, 10, step=1)
    cutline_pt = st.slider("PDF cut line thickness (pt)", 0.2, 2.0, 0.8, step=0.1)

st.divider()

# Parse words
words = [w.strip() for w in (words_text or "").splitlines() if w.strip()]
word_symbols: List[Image.Image] = []
if words:
    # Make text symbols at a base size; they will be scaled/rotated like images
    # font size tuned for typical single words
    for w in words:
        word_symbols.append(make_text_symbol(w, size_px=520, font_size=90))

num_words = len(word_symbols)

if not uploaded and num_words == 0:
    st.info("Upload images and/or add words to begin. Smallest true deck needs 7 symbols total.")
    st.stop()

# Load images
imgs_raw = load_images(uploaded) if uploaded else []
num_images = len(imgs_raw)

# Build symbol pool
symbols_pool = [normalize_symbol(im) for im in imgs_raw] + [normalize_symbol(im, pad=10) for im in word_symbols]
total_symbols = len(symbols_pool)

best_n = best_prime_order_for_symbols(total_symbols)
if best_n is None:
    st.error(
        f"You have {total_symbols} symbols total (images + words). "
        "The smallest true deck is n=2, which needs 7 symbols. Add more images/words."
    )
    st.stop()

needed_symbols = best_n * best_n + best_n + 1

st.subheader("2) Preview")
st.write(f"Images uploaded: **{num_images}**")
st.write(f"Words added: **{num_words}**")
st.write(f"Total symbols available: **{total_symbols}**")
st.write(f"Best valid Dobble order (prime n): **{best_n}** → needs **{needed_symbols}** symbols")

# allow override downward
valid_ns = [n for n in range(2, best_n + 1) if is_prime(n) and (n * n + n + 1) <= total_symbols]
chosen_n = st.selectbox("Deck size (choose smaller n for fewer cards)", valid_ns, index=len(valid_ns) - 1)

v = chosen_n * chosen_n + chosen_n + 1
k = chosen_n + 1
st.write(f"Selected: **n={chosen_n}** → **{v} cards**, **{k} symbols per card**")

if total_symbols < v:
    st.error(f"Not enough symbols for n={chosen_n}. Need {v}, you have {total_symbols}.")
    st.stop()

# Preview images + words
st.caption("Preview of your symbol pool (first 24):")
preview_syms = min(total_symbols, 24)
prev_cols = st.columns(6)
for i in range(preview_syms):
    with prev_cols[i % 6]:
        st.image(symbols_pool[i], caption=f"#{i}", use_container_width=True)

st.divider()
st.subheader("3) Generate")
generate = st.button("Generate Dobble deck", type="primary")

if not generate:
    st.stop()

rng = random.Random(int(rng_seed))

# Use first v symbols (stable); you can also shuffle here if you prefer
symbols = symbols_pool[:v]

cards = generate_projective_plane_deck(chosen_n)
ok, msg = verify_dobble_property(cards)
if not ok:
    st.error("Deck generation failed verification: " + msg)
    st.stop()

st.success(f"Deck generated and verified ✅  ({v} cards, {k} symbols each)")

# Render cards
with st.spinner("Rendering card PNGs..."):
    rendered_cards: List[Image.Image] = []

    order = list(range(len(cards)))
    rng.shuffle(order)

    for idx in order:
        card_symbols = cards[idx][:]
        rng.shuffle(card_symbols)

        card_img = place_symbols_on_card(
            symbol_imgs=symbols,
            card_symbols=card_symbols,
            size_px=size_px,
            profile=profile,
            rng=rng,
            shape=shape,
        )

        # Add outline for cutting
        card_img = draw_cut_outline(card_img, shape=shape, outline_px=int(outline_px))
        rendered_cards.append(card_img)

st.subheader("Preview generated cards")
preview_gen = min(len(rendered_cards), 12)
gen_cols = st.columns(6)
for i in range(preview_gen):
    with gen_cols[i % 6]:
        st.image(rendered_cards[i], caption=f"Card {i + 1}", use_container_width=True)

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
    "Download cards (PNG ZIP)",
    data=zip_buf.getvalue(),
    file_name=f"dobble_cards_{shape.lower()}_n{chosen_n}_png.zip",
    mime="application/zip",
)

# PDF (with cut lines)
with st.spinner("Building A4 PDF with cut lines..."):
    pdf_bytes = build_a4_pdf(
        card_images=rendered_cards,
        cards_per_page=int(cards_per_page),
        card_diameter_mm=float(card_size_mm),
        margin_mm=float(margin_mm),
        shape=shape,
        cutline_pt=float(cutline_pt),
    )

st.download_button(
    "Download print-ready A4 PDF (with cut lines)",
    data=pdf_bytes,
    file_name=f"dobble_cards_{shape.lower()}_n{chosen_n}_A4.pdf",
    mime="application/pdf",
)

st.caption("Print tip: use 100% scale (disable 'fit to page') for accurate cutting.")
