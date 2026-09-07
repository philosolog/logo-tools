"""Core image transforms for personal logo-editing macros."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageChops, ImageFilter

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
    color: Color | None = "white",
    invert: bool = False,
    black_point: int = 0,
    white_point: int = 255,
    sharpen: float = 0.0,
) -> Image.Image:
    """Map source luminance to alpha, optionally repainting a solid color.

    ``invert=False`` treats white as opaque. With ``invert=True``, black is
    opaque instead. In the resulting opacity signal, values at or below
    ``black_point`` become transparent and values at or above ``white_point``
    become fully opaque; the band between them is linearly stretched across
    0–255, so relative differences are preserved while dull greys are pushed
    toward the nearer extreme. Existing source alpha is multiplied into the
    computed opacity.

    ``sharpen`` is an unsharp-mask amount in percent (``0`` disables it),
    applied to the luminance before keying so soft anti-aliased strokes gain
    crisper edges.

    ``color`` repaints every pixel one solid color. Pass ``color=None`` to keep
    each pixel's own RGB and replace only the alpha channel.
    """
    if isinstance(black_point, bool) or not isinstance(black_point, int):
        raise TypeError("black_point must be an integer")
    if isinstance(white_point, bool) or not isinstance(white_point, int):
        raise TypeError("white_point must be an integer")
    if not 0 <= black_point < white_point <= 255:
        raise ValueError(
            "points must satisfy 0 <= black_point < white_point <= 255"
        )
    if isinstance(sharpen, bool) or not isinstance(sharpen, (int, float)):
        raise TypeError("sharpen must be a number")
    if not math.isfinite(sharpen) or sharpen < 0:
        raise ValueError("sharpen must be a non-negative number")

    source = image.convert("RGBA")
    signal = image.convert("L")
    if sharpen:
        signal = signal.filter(
            ImageFilter.UnsharpMask(radius=2, percent=round(sharpen), threshold=0)
        )
    if invert:
        signal = ImageChops.invert(signal)

    if black_point or white_point != 255:
        span = white_point - black_point
        levels = [
            0
            if value <= black_point
            else 255
            if value >= white_point
            else round((value - black_point) * 255 / span)
            for value in range(256)
        ]
        signal = signal.point(levels)

    alpha = ImageChops.multiply(signal, source.getchannel("A"))

    if color is None:
        result = source.copy()
    else:
        # Let Pillow validate named, hexadecimal, and RGB tuple color values.
        rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
        result = Image.new("RGBA", image.size, (*rgb, 255))
    result.putalpha(alpha)
    return result


def square_crop(image: Image.Image, *, margin: float = 0.0) -> Image.Image:
    """Trim to visible artwork and center it on a square transparent canvas.

    The square's edge is the longer trimmed dimension grown by ``margin`` as a
    fraction of that edge (``margin=0.1`` adds a 10% border). The result is
    always ``RGBA`` so a later background fill can paint the padding.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise TypeError("margin must be a number")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a non-negative number")

    artwork = _visible_crop(image).convert("RGBA")
    edge = round(max(artwork.size) * (1 + margin))
    frame = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
    frame.alpha_composite(
        artwork,
        ((edge - artwork.width) // 2, (edge - artwork.height) // 2),
    )
    return frame


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
    vertical_offset: float = 0.0,
) -> Image.Image:
    """Place artwork on a square transparent canvas by visible center of mass.

    ``vertical_offset`` shifts the resting position down by that many pixels
    (negative moves it up), applied before edge clamping.

    The desired position is clamped only when exact mass-centering would clip
    artwork at a canvas edge.
    """
    _validate_canvas_size(canvas_size)
    if isinstance(vertical_offset, bool) or not isinstance(
        vertical_offset, (int, float)
    ):
        raise TypeError("vertical_offset must be a number")
    if not math.isfinite(vertical_offset):
        raise ValueError("vertical_offset must be finite")

    artwork = _visible_crop(image)

    if artwork.width > canvas_size or artwork.height > canvas_size:
        raise ValueError("artwork is larger than the canvas")

    center_x, center_y = center_of_mass(artwork)
    desired_x = round(canvas_size / 2 - center_x)
    desired_y = round(canvas_size / 2 - center_y + vertical_offset)
    position = (
        min(max(desired_x, 0), canvas_size - artwork.width),
        min(max(desired_y, 0), canvas_size - artwork.height),
    )

    frame = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    frame.alpha_composite(artwork.convert("RGBA"), position)
    return frame
