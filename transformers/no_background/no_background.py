#!/usr/bin/env python3
"""Monochrome logo sketch -> centered transparent logo (no background fill).

The Rogusan pipeline without the final ``fill_background`` step. For every
image in ``input/`` it runs:

1. ``luminance_to_alpha`` -- read the sketch's luminance as an alpha map, invert
   so the drawn strokes stay opaque, stretch the level band between
   ``BLACK_POINT`` and ``WHITE_POINT``, and unsharp-mask the edges
2. ``fit``               -- trim to the artwork and scale its long edge to
   ``OCCUPANCY`` of the canvas
3. ``center_by_mass``     -- place it on a square ``CANVAS_SIZE`` canvas by its
   visible center of mass

Each sketch produces one PNG in ``output/``: ``<name>_transparent.png`` --
``STROKE_COLOR`` strokes on a transparent square canvas, ready to composite
over any background later.
"""

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from logo_tools import center_by_mass, fit, luminance_to_alpha

# Keep progress lines printable when source filenames contain non-ASCII
# characters, whatever the console's native encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent

# Edit these settings for each logo job. Seeded from the Rogusan build.
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
CANVAS_SIZE = 512  # side of the square output, pixels
OCCUPANCY = 0.82  # fraction of the canvas the artwork's long edge fills
VERTICAL_OFFSET = 24  # shift artwork down this many pixels (negative = up)
INVERT = True  # True: dark strokes in the sketch become the opaque mark
BLACK_POINT = 20  # opacity at/below this goes transparent (drops paper haze)
WHITE_POINT = 150  # opacity at/above this snaps to a solid stroke
SHARPEN = 120  # unsharp-mask percent for crisper edges; 0 disables
STROKE_COLOR = "white"  # solid color painted onto the kept strokes
OUTPUT_SUFFIX = "_transparent"  # appended to the source stem

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def render(sketch: Image.Image) -> Image.Image:
    """Run the pipeline up to (but not including) the background fill."""
    strokes = luminance_to_alpha(
        sketch,
        color=STROKE_COLOR,
        invert=INVERT,
        black_point=BLACK_POINT,
        white_point=WHITE_POINT,
        sharpen=SHARPEN,
    )
    fitted = fit(strokes, canvas_size=CANVAS_SIZE, occupancy=OCCUPANCY)
    return center_by_mass(
        fitted, canvas_size=CANVAS_SIZE, vertical_offset=VERTICAL_OFFSET
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sketches = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sketches:
        raise SystemExit(f"no images found in {INPUT_DIR}")

    for path in sketches:
        try:
            with Image.open(path) as handle:
                sketch = handle.convert("RGBA")
        except (UnidentifiedImageError, OSError) as error:
            print(f"skip {path.name}: {error}")
            continue
        try:
            logo = render(sketch)
        except ValueError as error:
            print(f"skip {path.name}: {error}")
            continue
        destination = OUTPUT_DIR / f"{path.stem}{OUTPUT_SUFFIX}.png"
        logo.save(destination, format="PNG")
        print(f"{path.name} -> {destination.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
