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
