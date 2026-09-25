"""Typst symbol -> pixelated square logo, white on black.

For every ``input/*.typ`` the pipeline runs:

1. ``typst compile`` renders the source to ``output/<name>.svg``
2. ``rsvg-convert`` rasterizes that SVG (transparent ground, white glyph)
3. ``fit`` / ``center_by_mass`` place it on a square canvas
4. ``fill_background`` flattens onto black
5. shrink to ``GRID`` x ``GRID`` cells, then blow back up with nearest-neighbour

Each source writes ``output/<name>.svg`` and ``output/<name>_pixel.png``.
"""

import io
import subprocess
from pathlib import Path

from PIL import Image

from logo_tools import center_by_mass, fill_background, fit

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"

CANVAS_SIZE = 1024  # side of the final square, pixels
GRID = 128  # pixel cells per side
OCCUPANCY = 0.4  # fraction of the canvas the glyph's long edge fills
THRESHOLD = 110  # 0-255 cutoff for pure black/white cells; None keeps gray


def make_svg(source: Path, destination: Path) -> None:
    subprocess.run(["typst", "compile", str(source), str(destination)], check=True)


def rasterize(svg: Path, *, size: int) -> Image.Image:
    png = subprocess.run(
        ["rsvg-convert", "--width", str(size * 2), str(svg)],
        check=True,
        capture_output=True,
    ).stdout
    return Image.open(io.BytesIO(png)).convert("RGBA")


def pixelate(image: Image.Image) -> Image.Image:
    small = image.convert("L").resize((GRID, GRID), Image.Resampling.BOX)
    if THRESHOLD is not None:
        small = small.point(lambda v: 255 if v >= THRESHOLD else 0)
    return small.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.NEAREST)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(INPUT_DIR.glob("*.typ"))
    if not sources:
        raise SystemExit(f"no .typ files found in {INPUT_DIR}")

    for source in sources:
        svg = OUTPUT_DIR / f"{source.stem}.svg"
        make_svg(source, svg)
        glyph = rasterize(svg, size=CANVAS_SIZE)
        fitted = fit(glyph, canvas_size=CANVAS_SIZE, occupancy=OCCUPANCY)
        placed = center_by_mass(fitted, canvas_size=CANVAS_SIZE)
        logo = pixelate(fill_background(placed, color="black"))
        destination = OUTPUT_DIR / f"{source.stem}_pixel.png"
        logo.save(destination, format="PNG")
        print(f"{source.name} -> {svg.name}, {destination.name}")


if __name__ == "__main__":
    main()
