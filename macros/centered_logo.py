#!/usr/bin/env python3
"""Editable white-on-black to centered-logo macro."""

from pathlib import Path

from PIL import Image

from logo_tools import center_by_mass, fill_background, fit, luminance_to_alpha

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these settings for each logo job.
INPUT_PATH = PROJECT_ROOT / "input.png"
OUTPUT_PATH = PROJECT_ROOT / "output.png"
CANVAS_SIZE = 512
OCCUPANCY = 0.5
COLOR = "white"
BACKGROUND_COLOR = "black"
INVERT = False
BLACK_POINT = 0


def main() -> None:
    with Image.open(INPUT_PATH) as source:
        artwork = luminance_to_alpha(
            source,
            color=COLOR,
            invert=INVERT,
            black_point=BLACK_POINT,
        )

    artwork = fit(
        artwork,
        canvas_size=CANVAS_SIZE,
        occupancy=OCCUPANCY,
    )
    logo = center_by_mass(artwork, canvas_size=CANVAS_SIZE)
    output = fill_background(logo, color=BACKGROUND_COLOR)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT_PATH, format="PNG")


if __name__ == "__main__":
    main()
