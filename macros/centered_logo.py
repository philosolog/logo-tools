#!/usr/bin/env python3
"""Monochrome logo sketch -> finished black-ground and white-ground logos.

For every image in ``input/`` the pipeline runs:

1. ``luminance_to_alpha`` -- read the sketch's luminance as an alpha map, invert
   so the drawn strokes stay opaque, stretch the level band between
   ``BLACK_POINT`` and ``WHITE_POINT``, and unsharp-mask the edges
2. ``fit``               -- trim to the artwork and scale its long edge to
   ``OCCUPANCY`` of the canvas
3. ``center_by_mass``     -- place it on a square ``CANVAS_SIZE`` canvas by its
   visible center of mass
4. ``fill_background``     -- flatten onto a solid ground

Each sketch produces two PNGs in ``output/``:

* ``<name>_on_black.png`` -- white strokes on black
* ``<name>_on_white.png`` -- black strokes on white
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from logo_tools import center_by_mass, fill_background, fit, luminance_to_alpha

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these settings for each logo job.
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
CANVAS_SIZE = 512  # side of the square output, pixels
OCCUPANCY = 0.82  # fraction of the canvas the artwork's long edge fills
INVERT = True  # True: dark strokes in the sketch become the opaque mark
BLACK_POINT = 20  # opacity at/below this goes transparent (drops paper haze)
WHITE_POINT = 150  # opacity at/above this snaps to a solid stroke
SHARPEN = 120  # unsharp-mask percent for crisper edges; 0 disables

# (stroke color, background color, output-name suffix)
VARIANTS = (
    ("white", "black", "_on_black"),
    ("black", "white", "_on_white"),
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def render(sketch: Image.Image, *, stroke: str, background: str) -> Image.Image:
    """Run the full pipeline for one stroke/background pairing."""
    strokes = luminance_to_alpha(
        sketch,
        color=stroke,
        invert=INVERT,
        black_point=BLACK_POINT,
        white_point=WHITE_POINT,
        sharpen=SHARPEN,
    )
    fitted = fit(strokes, canvas_size=CANVAS_SIZE, occupancy=OCCUPANCY)
    placed = center_by_mass(fitted, canvas_size=CANVAS_SIZE)
    return fill_background(placed, color=background)


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
        for stroke, background, suffix in VARIANTS:
            try:
                logo = render(sketch, stroke=stroke, background=background)
            except ValueError as error:
                print(f"skip {path.name}{suffix}: {error}")
                continue
            destination = OUTPUT_DIR / f"{path.stem}{suffix}.png"
            logo.save(destination, format="PNG")
            print(f"{path.name} -> {destination.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
