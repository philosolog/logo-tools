import io
import re
import subprocess
from pathlib import Path

from PIL import Image

from logo_tools import center_by_mass, fill_background, fit

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
FONT_DIR = SCRIPT_DIR / "fonts"

CANVAS_SIZE = 1024  # side of the final square, pixels
GRID = 2048  # pixel cells per side
OCCUPANCY = 0.4  # fraction of the canvas the glyph's long edge fills
THRESHOLD = 110  # 0-255 cutoff for pure black/white cells; None keeps gray


# Prepended to every source: a page hugging the glyph, transparent ground,
# white ink. Sources can then be a bare formula like ``$star_M$``. Text edges
# default to baseline/cap-height, which would clip subscripts and descenders,
# so measure glyph bounds and leave a margin; ``fit`` crops it away anyway.
PREAMBLE = """#set page(width: auto, height: auto, margin: 10pt, fill: none)
#set text(fill: white, top-edge: "bounds", bottom-edge: "bounds")
"""


def typst(source: Path, destination: Path, *extra: str) -> bytes:
    """Compile ``source`` (with ``PREAMBLE``) to ``destination``; ``-`` for stdout."""
    return subprocess.run(
        ["typst", "compile", "--font-path", str(FONT_DIR), *extra, "-", str(destination)],
        input=(PREAMBLE + source.read_text(encoding="utf-8")).encode(),
        cwd=source.parent,
        check=True,
        capture_output=True,
    ).stdout


def page_size(svg: Path) -> tuple[float, float]:
    """Width and height of a Typst SVG, in points."""
    view_box = re.search(r'viewBox="([^"]+)"', svg.read_text()).group(1)
    _, _, width, height = map(float, view_box.split())
    return width, height


def rasterize(source: Path, svg: Path, *, size: int) -> Image.Image:
    """Render ``source`` as PNG with its long edge at ``2 * size`` pixels."""
    ppi = 2 * size * 72 / max(page_size(svg))
    png = typst(source, Path("-"), "--format", "png", "--ppi", f"{ppi:.2f}")
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
        typst(source, svg)
        glyph = rasterize(source, svg, size=CANVAS_SIZE)
        fitted = fit(glyph, canvas_size=CANVAS_SIZE, occupancy=OCCUPANCY)
        placed = center_by_mass(fitted, canvas_size=CANVAS_SIZE)
        logo = pixelate(fill_background(placed, color="black"))
        destination = OUTPUT_DIR / f"{source.stem}_pixel.png"
        logo.save(destination, format="PNG")
        print(f"{source.name} -> {svg.name}, {destination.name}")


if __name__ == "__main__":
    main()
