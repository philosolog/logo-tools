#!/usr/bin/env python3
"""Editable white-on-black to centered-logo macro (original logo).

The pipeline that produced the first logo. Runs every image in ``input/``
through luminance keying, fitting, mass-centering, and a solid background,
writing one PNG per image to ``output/`` with the source file's stem.
"""

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from logo_tools import center_by_mass, fill_background, fit, luminance_to_alpha

# Keep progress lines printable when source filenames contain non-ASCII
# characters, whatever the console's native encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent

# Edit these settings for each logo job.
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
CANVAS_SIZE = 512
OCCUPANCY = 0.5
COLOR = "white"
BACKGROUND_COLOR = "black"
INVERT = False
BLACK_POINT = 0

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def build_logo(source: Image.Image) -> Image.Image:
    """Apply the full luminance-to-alpha logo pipeline to one source image."""
    artwork = luminance_to_alpha(
        source,
        color=COLOR,
        invert=INVERT,
        black_point=BLACK_POINT,
    )
    artwork = fit(artwork, canvas_size=CANVAS_SIZE, occupancy=OCCUPANCY)
    logo = center_by_mass(artwork, canvas_size=CANVAS_SIZE)
    return fill_background(logo, color=BACKGROUND_COLOR)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"no images found in {INPUT_DIR}")

    for path in sources:
        destination = OUTPUT_DIR / f"{path.stem}.png"
        try:
            with Image.open(path) as source:
                logo = build_logo(source)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(f"skip {path.name}: {error}")
            continue
        logo.save(destination, format="PNG")
        print(f"{path.name} -> {destination.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
