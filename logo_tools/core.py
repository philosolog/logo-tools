"""Core image transforms for personal logo-editing macros."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageChops

type Color = str | Sequence[int]


def _validate_canvas_size(canvas_size: int) -> None:
    if isinstance(canvas_size, bool) or not isinstance(canvas_size, int):
        raise TypeError("canvas_size must be an integer")
    if canvas_size <= 0:
        raise ValueError("canvas_size must be greater than zero")


def _weight_mask(image: Image.Image) -> Image.Image:
    """Return the channel that represents visible mass."""
    if "A" in image.getbands() or "transparency" in image.info:
        return image.convert("RGBA").getchannel("A")
    return image.convert("L")


def _visible_crop(image: Image.Image) -> Image.Image:
    bounds = _weight_mask(image).getbbox()
    if bounds is None:
        raise ValueError("image contains no visible artwork")
    return image.crop(bounds)


def luminance_to_alpha(
    image: Image.Image,
    *,
    color: Color = "white",
    invert: bool = False,
    black_point: int = 0,
) -> Image.Image:
    """Map source luminance to alpha and paint the result a solid color.

    ``invert=False`` treats white as opaque. With ``invert=True``, black is
    opaque instead. Values at or below ``black_point`` in the resulting opacity
    signal become transparent; the remainder is linearly remapped to 0–255.
    Existing source alpha is multiplied into the computed opacity.
    """
    if isinstance(black_point, bool) or not isinstance(black_point, int):
        raise TypeError("black_point must be an integer")
    if not 0 <= black_point < 255:
        raise ValueError("black_point must be between 0 and 254")

    source = image.convert("RGBA")
    signal = image.convert("L")
    if invert:
        signal = ImageChops.invert(signal)

    if black_point:
        denominator = 255 - black_point
        levels = [
            0
            if value <= black_point
            else round((value - black_point) * 255 / denominator)
            for value in range(256)
        ]
        signal = signal.point(levels)

    alpha = ImageChops.multiply(signal, source.getchannel("A"))

    # Let Pillow validate named, hexadecimal, and RGB tuple color values.
    rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
    result = Image.new("RGBA", image.size, (*rgb, 255))
    result.putalpha(alpha)
    return result


def center_of_mass(image: Image.Image) -> tuple[float, float]:
    """Return the visible center of mass in pixel-edge coordinates.

    Alpha supplies the weights when available; otherwise luminance does. Pixel
    centers are represented as ``x + 0.5`` and ``y + 0.5``.
    """
    weights = _weight_mask(image)
    width, _ = weights.size
    total = 0
    weighted_x = 0.0
    weighted_y = 0.0

    for index, weight in enumerate(weights.get_flattened_data()):
        if not weight:
            continue
        y, x = divmod(index, width)
        total += weight
        weighted_x += (x + 0.5) * weight
        weighted_y += (y + 0.5) * weight

    if total == 0:
        raise ValueError("image contains no visible artwork")

    return weighted_x / total, weighted_y / total


def fit(
    image: Image.Image,
    *,
    canvas_size: int = 512,
    occupancy: float = 0.75,
) -> Image.Image:
    """Trim artwork and resize its longest edge to the requested occupancy."""
    _validate_canvas_size(canvas_size)
    if isinstance(occupancy, bool) or not isinstance(occupancy, (int, float)):
        raise TypeError("occupancy must be a number")
    if not math.isfinite(occupancy) or not 0 < occupancy <= 1:
        raise ValueError("occupancy must be greater than 0 and at most 1")

    artwork = _visible_crop(image)
    target_edge = max(1, round(canvas_size * occupancy))
    resize = target_edge / max(artwork.size)
    resized_size = (
        max(1, round(artwork.width * resize)),
        max(1, round(artwork.height * resize)),
    )

    if resized_size == artwork.size:
        return artwork.copy()
    return artwork.resize(resized_size, Image.Resampling.LANCZOS)


def center_by_mass(
    image: Image.Image,
    *,
    canvas_size: int = 512,
) -> Image.Image:
    """Place artwork on a square transparent canvas by visible center of mass.

    The desired position is clamped only when exact mass-centering would clip
    artwork at a canvas edge.
    """
    _validate_canvas_size(canvas_size)
    artwork = _visible_crop(image)

    if artwork.width > canvas_size or artwork.height > canvas_size:
        raise ValueError("artwork is larger than the canvas")

    center_x, center_y = center_of_mass(artwork)
    desired_x = round(canvas_size / 2 - center_x)
    desired_y = round(canvas_size / 2 - center_y)
    position = (
        min(max(desired_x, 0), canvas_size - artwork.width),
        min(max(desired_y, 0), canvas_size - artwork.height),
    )

    frame = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    frame.alpha_composite(artwork.convert("RGBA"), position)
    return frame
