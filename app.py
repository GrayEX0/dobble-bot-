import io
import math
import random
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    best = None
    max_guess = int(math.sqrt(max(0, num_symbols))) + 2
    for n in range(2, max_guess + 1):
        if is_prime(n) and (n * n + n + 1) <= num_symbols:
            best = n
    return best


def generate_projective_plane_deck(n: int) -> List[List[int]]:
    if not is_prime(n):
        raise ValueError("n must be prime for this implementation.")

    v = n * n + n + 1
    cards: List[List[int]] = []

    INF = 0
    S = [1 + m for m in range(n)]

    def P(x: int, y: int) -> int:
        return 1 + n + x * n + y

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
    "K1 (bigger, fewer overlaps)": RenderProfile("K1", 0.22, 0.34, 18, 0.10, 300),
    "K2 (balanced)": RenderProfile("K2", 0.18, 0.30, 30, 0.08, 450),
    "K3 (denser, more challenge)": RenderProfile("K3", 0.16, 0.28, 40, 0.06, 650),
}


ALLOWED_WORD_COLORS = {
    "black": (0, 0, 0, 255),
    "red": (220, 30, 30, 255),
    "green": (20, 140, 60, 255),
    "orange": (230, 130, 20, 255),
    "blue": (30, 80, 220, 255),
}

# A pragmatic list of fonts that *often* exist on Streamlit Cloud/Linux images.
# If none exist, we fall back to PIL default font.
FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf",
    "LiberationSans-Regular.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSerif-Regular.ttf",
    "FreeSans.ttf",
    "FreeSerif.ttf",
]


def load_images(uploaded_files) -> List[Image.Image]:
    imgs = []
    for f in uploaded_files:
        im = Image.open(f).convert("RGBA")
        im = ImageOps.exif_transpose(im)
        imgs.append(im)
    return imgs


def normalize_symbol(im: Image.Image, pad: int = 16) -> Image.Image:
    w, h = im.size
    canvas_size = max(w, h) + 2 * pad
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    out.alpha_composite(im, ((canvas_size - w) // 2, (canvas_size - h) // 2))
    return out


def circle_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    return mask


def bbox_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def try_load_font(font_name: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        return None


def pick_font(rng: random.Random, font_size: int, randomize: bool, preferred: Optional[str] = None) -> ImageFont.ImageFont:
    # if user chooses a preferred font name and it loads, use it
    if preferred:
        f = try_load_font(preferred, font_size)
        if f:
            return f

    if randomize:
        candidates = FONT_CANDIDATES[:]
        rng.shuffle(candidates)
        for name in candidates:
            f = try_load_font(name, font_size)
            if f:
                return f

    # fallback: try a default known font
    f = try_load_font("DejaVuSans.ttf", font_size)
    if f:
        return f

    # final fallback
    return ImageFont.load_default()


def make_text_symbol(
    text: str,
    rng: random.Random,
    randomize_font: bool,
    randomize_color: bool,
    chosen_color_name: str,
    preferred_font: Optional[str],
    base_canvas_px: int = 520,
    font_size: int = 90,
    pad: int = 24,
    rounded: int = 28,
) -> Image.Image:
    text = text.strip()
    if not text:
        return Image.new("RGBA", (base_canvas_px, base_canvas_px), (0, 0, 0, 0))

    # font
    font = pick_font(rng, font_size, randomize_font, preferred=preferred_font)

    # color
    if randomize_color:
        color_name = rng.choice(list(ALLOWED_WORD_COLORS.keys()))
    else:
        color_name = chosen_color_name if chosen_color_name in ALLOWED_WORD_COLORS else "black"
    text_color = ALLOWED_WORD_COLORS[color_name]

    # measure
    tmp = Image.new("RGBA", (base_canvas_px, base_canvas_px), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    w = min(base_canvas_px, tw + 2 * pad)
    h = min(base_canvas_px, th + 2 * pad)

    label = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dl = ImageDraw.Draw(label)

    # translucent white pill background for readability
    dl.rounded_rectangle((0, 0, w - 1, h - 1), radius=rounded, fill=(255, 255, 255, 220))
    tx = (w - tw) // 2 - bbox[0]
    ty = (h - th) // 2 - bbox[1]
    dl.text((tx, ty), text, font=font, fill=text_color)

    # center onto square canvas for consistent scaling/rotation
    canvas_size = max(w, h)
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    out.alpha_composite(label, ((canvas_size - w) // 2, (canvas_size - h) // 2))
    return out


def draw_cut_outline(card: Image.Image, shape: str, outline_px: int) -> Image.Image:
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
    base = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))

    cx, cy = size_px / 2, size_px / 2
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
                x, y = int(round(x)), int(round(y))
                box = (x, y, x + iw, y + ih)

                if box[0] < 0 or box[1] < 0 or box[2] > size_px or box[3] > size_px:
                    continue

                if any(bbox_intersect(box, pb) for pb in placed_boxes):
                    continue

                if shape == "Circle":
                    corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
                    inside = 0
                    for px, py in corners:
                        dx, dy = (px - cx), (py - cy)
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

    if shape == "Circle":
        mask = circle_mask(size_px)
        circ = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
        circ.paste(base, (0, 0), mask=mask)
        return circ

    return base


def make_back_card_image(back_img: Image.Image, size_px: int, shape: str, outline_px: int) -> Image.Image:
    """
    Resize/crop a provided card-back image into a square canvas (size_px),
    then optionally mask to circle, and add outline.
    """
    back = back_img.convert("RGBA")
    # fit to square
    back = ImageOps.fit(back, (size_px, size_px), method=Image.LANCZOS, centering=(0.5, 0.5))

    if shape == "Circle":
        mask = circle_mask(size_px)
        circ = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
        circ.paste(back, (0, 0), mask=mask)
        back = circ

    back = draw_cut_outline(back, shape=shape, outline_px=outline_px)
    return back


# ----------------------------
# PDF export (single / duplex)
# ----------------------------

def draw_cut_outline_vector(c: canvas.Canvas, shape: str, x0: float, y0: float, size_pt: float, line_pt: float):
    c.setLineWidth(line_pt)
    if shape == "Circle":
        c.circle(x0 + size_pt / 2, y0 + size_pt / 2, size_pt / 2)
    else:
        c.rect(x0, y0, size_pt, size_pt)


def build_a4_pdf_duplex(
    front_cards: List[Image.Image],
    back_card_img: Optional[Image.Image],
    cards_per_page: int,
    card_size_mm: float,
    margin_mm: float,
    shape: str,
    cutline_pt: float,
    mirror_backs_horizontally: bool = True,
) -> bytes:
    """
    If back_card_img is None → single-sided PDF (fronts only).
    Else → duplex PDF: for each front page, the next page is the backs
    laid out to align when printing duplex (mirrored horizontally by default).
    """
    page_w, page_h = A4
    size_pt = card_size_mm * mm
    margin_pt = margin_mm * mm

    cols = int(math.ceil(math.sqrt(cards_per_page)))
    rows = int(math.ceil(cards_per_page / cols))

    usable_w = page_w - 2 * margin_pt
    usable_h = page_h - 2 * margin_pt
    step_x = usable_w / cols
    step_y = usable_h / rows

    max_size = min(step_x, step_y) * 0.92
    size_pt = min(size_pt, max_size)

    def cell_xy(r: int, col: int) -> Tuple[float, float]:
        x = margin_pt + col * step_x + (step_x - size_pt) / 2
        y = page_h - (margin_pt + (r + 1) * step_y) + (step_y - size_pt) / 2
        return x, y

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    idx = 0
    while idx < len(front_cards):
        # ----- FRONT PAGE -----
        front_positions = []  # store positions of each card on this page in draw order
        for r in range(rows):
            for col in range(cols):
                if idx >= len(front_cards) or (r * cols + col) >= cards_per_page:
                    break
                x0, y0 = cell_xy(r, col)
                front_positions.append((r, col, x0, y0, idx))

                # cut outline vector
                draw_cut_outline_vector(c, shape, x0, y0, size_pt, cutline_pt)

                # image
                png_buf = io.BytesIO()
                front_cards[idx].save(png_buf, format="PNG")
                png_buf.seek(0)
                c.drawImage(ImageReader(png_buf), x0, y0, width=size_pt, height=size_pt, mask="auto")

                idx += 1

        c.showPage()

        # ----- BACK PAGE (optional) -----
        if back_card_img is not None:
            # For duplex alignment, mirror column placement horizontally.
            # (Most printers: "flip on long edge". If yours differs, toggle mirror in UI.)
            for (r, col, x0, y0, _front_idx) in front_positions:
                back_col = (cols - 1 - col) if mirror_backs_horizontally else col
                bx0, by0 = cell_xy(r, back_col)

                draw_cut_outline_vector(c, shape, bx0, by0, size_pt, cutline_pt)

                png_buf = io.BytesIO()
                back_card_img.save(png_buf, format="PNG")
                png_buf.seek(0)
                c.drawImage(ImageReader(png_buf), bx0, by0, width=size_pt, height=size_pt, mask="auto")

            c.showPage()

    c.save()
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Dobble Bot (K1/K2/K3)", layout="wide")
st.title("Dobble Bot 🎴 (K1 / K2 / K3)")
st.caption("Upload images + optional words → generate → download PNG ZIP + A4 PDF with cut lines (optional duplex backs)")

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

    st.markdown("**Optional: Card Back (for double-sided PDF)**")
    back_upload = st.file_uploader(
        "Upload a card back image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )

with colB:
    profile_name = st.selectbox("Class profile", list(PROFILES.keys()), index=1)
    profile = PROFILES[profile_name]

    shape = st.selectbox("Card shape", ["Circle", "Square"], index=0)
    size_px = st.slider("Card export size (px)", 800, 2400, 1400, step=100)

    outline_px = st.slider("Cut outline thickness on PNG (px)", 0, 20, 6, step=1)
    rng_seed = st.number_input("Random seed (same seed = same layout)", value=1234, step=1)

    st.markdown("**Word styling**")
    randomize_word_fonts = st.checkbox("Randomize word fonts", value=True)
    randomize_word_colors = st.checkbox("Randomize word colors", value=True)

    preferred_font = st.selectbox(
        "Preferred font (used if randomize fonts OFF, or if available)",
        ["(auto)"] + FONT_CANDIDATES,
        index=0
    )
    preferred_font = None if preferred_font == "(auto)" else preferred_font

    chosen_word_color = st.selectbox(
        "Word color (used if randomize colors OFF)",
        list(ALLOWED_WORD_COLORS.keys()),
        index=0
    )

    st.markdown("**PDF settings**")
    cards_per_page = st.selectbox("Cards per A4 page", [6, 8, 9, 12], index=2)
    card_size_mm = st.slider("Card size on A4 (mm)", 55, 95, 80, step=1)
    margin_mm = st.slider("Page margin (mm)", 5, 20, 10, step=1)
    cutline_pt = st.slider("PDF cut line thickness (pt)", 0.2, 2.0, 0.8, step=0.1)

    duplex = st.checkbox("Make PDF double-sided with card backs", value=True)
    mirror_backs = st.checkbox("Mirror backs horizontally (recommended)", value=True)

st.divider()

# Parse words
words = [w.strip() for w in (words_text or "").splitlines() if w.strip()]

# Load symbols
imgs_raw = load_images(uploaded) if uploaded else []
num_images = len(imgs_raw)

rng = random.Random(int(rng_seed))

word_symbols = []
for w in words:
    sym = make_text_symbol(
        text=w,
        rng=rng,
        randomize_font=randomize_word_fonts,
        randomize_color=randomize_word_colors,
        chosen_color_name=chosen_word_color,
        preferred_font=preferred_font,
        base_canvas_px=520,
        font_size=90,
    )
    word_symbols.append(sym)

symbols_pool = [normalize_symbol(im) for im in imgs_raw] + [normalize_symbol(im, pad=10) for im in word_symbols]
total_symbols = len(symbols_pool)

if total_symbols == 0:
    st.info("Upload images and/or add words to begin. Smallest true deck needs 7 symbols total.")
    st.stop()

best_n = best_prime_order_for_symbols(total_symbols)
if best_n is None:
    st.error(f"You have {total_symbols} symbols total. The smallest true deck is n=2 (7 symbols). Add more.")
    st.stop()

needed_symbols = best_n * best_n + best_n + 1

st.subheader("2) Preview")
st.write(f"Images: **{num_images}** | Words: **{len(words)}** | Total symbols: **{total_symbols}**")
st.write(f"Best valid Dobble order (prime n): **{best_n}** → needs **{needed_symbols}** symbols")

valid_ns = [n for n in range(2, best_n + 1) if is_prime(n) and (n * n + n + 1) <= total_symbols]
chosen_n = st.selectbox("Deck size (choose smaller n for fewer cards)", valid_ns, index=len(valid_ns) - 1)

v = chosen_n * chosen_n + chosen_n + 1
k = chosen_n + 1

st.write(f"Selected: **n={chosen_n}** → **{v} cards**, **{k} symbols per card**")

if total_symbols < v:
    st.error(f"Not enough symbols for n={chosen_n}. Need {v}, you have {total_symbols}.")
    st.stop()

# Preview first 24 symbols
st.caption("Symbol pool preview (first 24):")
preview_syms = min(total_symbols, 24)
prev_cols = st.columns(6)
for i in range(preview_syms):
    with prev_cols[i % 6]:
        st.image(symbols_pool[i], caption=f"#{i}", use_container_width=True)

# Prepare back image (optional)
back_card_prepared = None
if duplex and back_upload is not None:
    back_img = Image.open(back_upload).convert("RGBA")
    back_card_prepared = make_back_card_image(back_img, size_px=size_px, shape=shape, outline_px=int(outline_px))

st.divider()
st.subheader("3) Generate")
generate = st.button("Generate Dobble deck", type="primary")

if not generate:
    st.stop()

# Use first v symbols (stable)
symbols = symbols_pool[:v]

cards = generate_projective_plane_deck(chosen_n)
ok, msg = verify_dobble_property(cards)
if not ok:
    st.error("Deck generation failed verification: " + msg)
    st.stop()

st.success(f"Deck generated and verified ✅  ({v} cards, {k} symbols each)")

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
        card_img = draw_cut_outline(card_img, shape=shape, outline_px=int(outline_px))
        rendered_cards.append(card_img)

st.subheader("Preview generated cards")
preview_gen = min(len(rendered_cards), 12)
gen_cols = st.columns(6)
for i in range(preview_gen):
    with gen_cols[i % 6]:
        st.image(rendered_cards[i], caption=f"Card {i + 1}", use_container_width=True)

if back_card_prepared is not None:
    st.caption("Card back preview:")
    st.image(back_card_prepared, use_container_width=False, width=220)

st.divider()
st.subheader("4) Download")

# ZIP of PNGs (fronts)
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for i, im in enumerate(rendered_cards, start=1):
        b = io.BytesIO()
        im.save(b, format="PNG")
        z.writestr(f"dobble_front_{i:03d}.png", b.getvalue())
    if back_card_prepared is not None:
        b = io.BytesIO()
        back_card_prepared.save(b, format="PNG")
        z.writestr("dobble_card_back.png", b.getvalue())
zip_buf.seek(0)

st.download_button(
    "Download cards (PNG ZIP)",
    data=zip_buf.getvalue(),
    file_name=f"dobble_{shape.lower()}_n{chosen_n}_png.zip",
    mime="application/zip",
)

# PDF
with st.spinner("Building A4 PDF..."):
    pdf_bytes = build_a4_pdf_duplex(
        front_cards=rendered_cards,
        back_card_img=back_card_prepared if duplex else None,
        cards_per_page=int(cards_per_page),
        card_size_mm=float(card_size_mm),
        margin_mm=float(margin_mm),
        shape=shape,
        cutline_pt=float(cutline_pt),
        mirror_backs_horizontally=bool(mirror_backs),
    )

pdf_name = f"dobble_{shape.lower()}_n{chosen_n}_A4.pdf" if back_card_prepared is None else f"dobble_{shape.lower()}_n{chosen_n}_A4_duplex.pdf"

st.download_button(
    "Download A4 PDF (cut lines + optional duplex backs)",
    data=pdf_bytes,
    file_name=pdf_name,
    mime="application/pdf",
)

st.caption(
    "Printing tip: Print at 100% scale (no 'fit to page'). For duplex: try 'flip on long edge'. "
    "If backs don’t line up, toggle 'Mirror backs horizontally'."
)
