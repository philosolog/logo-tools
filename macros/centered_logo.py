#!/usr/bin/env python3
"""Editable white-on-black to centered-logo macro.

Runs every image in ``input/`` through the pipeline and writes one PNG per
image to ``output/``, keeping the source file's stem.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from logo_tools import center_by_mass, fill_background, fit, luminance_to_alpha

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these settings for each logo job.
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
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
        print(f"{path.name} -> {destination.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
