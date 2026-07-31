"""Background compositing for logo-editing pipelines."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

type Color = str | Sequence[int]


def fill_background(
    image: Image.Image,
    *,
    color: Color = "black",
) -> Image.Image:
    """Return ``image`` composited over an opaque solid-color background."""
    # Resolve through RGB so background always stays opaque, even when source
    # contains transparent or partially transparent pixels.
    rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
    background = Image.new("RGBA", image.size, (*rgb, 255))
    return Image.alpha_composite(background, image.convert("RGBA"))
