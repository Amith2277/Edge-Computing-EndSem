"""Synthetic drift for testing the drift monitor (notebook 12). Each perturbation maps a PIL RGB image to a new one.
These are stand-ins for real changes (dusk, fog, a dirty or defocused lens, sensor noise, bandwidth-starved
compression, a bumped camera); they are not claimed to reproduce them."""
import io
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def dark(im: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(factor)


def haze(im: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(im, Image.new("RGB", im.size, (200, 200, 200)), alpha)


def blur(im: Image.Image, radius: float) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius))


def noise(im: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    a = np.asarray(im, np.float32)
    a = a + np.random.RandomState(seed).normal(0, sigma, a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=int(quality))
    b.seek(0)
    return Image.open(b).convert("RGB")


def shift(im: Image.Image, frac: float) -> Image.Image:
    """Camera bumped: content moves right and down by frac of the width (slot polygons stay where they were)."""
    d = int(round(frac * im.width))
    return im.transform(im.size, Image.AFFINE, (1, 0, -d, 0, 1, -d), fillcolor=(0, 0, 0))


KINDS: Dict[str, Tuple[Callable, Dict[str, float]]] = {
    "dark": (dark, {"mild": 0.6, "strong": 0.3}),
    "haze": (haze, {"mild": 0.3, "strong": 0.6}),
    "blur": (blur, {"mild": 2.0, "strong": 5.0}),
    "noise": (noise, {"mild": 10.0, "strong": 25.0}),
    "jpeg": (jpeg, {"mild": 20, "strong": 5}),
    "shift": (shift, {"mild": 0.02, "strong": 0.05}),
}


def variants():
    """('none', None, None) then every (kind, level) pair."""
    out = [("none", None, None)]
    for k, (_, lv) in KINDS.items():
        for name in ("mild", "strong"):
            out.append(("{}_{}".format(k, name), k, lv[name]))
    return out


def apply(im: Image.Image, kind, value, seed: int = 0) -> Image.Image:
    im = im.convert("RGB")
    if kind is None:
        return im
    f = KINDS[kind][0]
    return f(im, value, seed) if kind == "noise" else f(im, value)
