from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from logo_tools import (
    center_by_mass,
    center_of_mass,
    fill_background,
    fit,
    luminance_to_alpha,
)


class LuminanceToAlphaTests(unittest.TestCase):
    def test_maps_luminance_to_solid_color_alpha(self) -> None:
        source = Image.new("L", (3, 1))
        source.putdata([0, 128, 255])

        result = luminance_to_alpha(source, color="#123456")

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(
            list(result.get_flattened_data()),
            [(18, 52, 86, 0), (18, 52, 86, 128), (18, 52, 86, 255)],
        )

    def test_inverts_opacity_signal(self) -> None:
        source = Image.new("L", (3, 1))
        source.putdata([0, 128, 255])

        result = luminance_to_alpha(source, invert=True)

        self.assertEqual(
            list(result.getchannel("A").get_flattened_data()),
            [255, 127, 0],
        )

    def test_remaps_values_above_black_point(self) -> None:
        source = Image.new("L", (4, 1))
        source.putdata([0, 10, 128, 255])

        result = luminance_to_alpha(source, black_point=10)

        self.assertEqual(
            list(result.getchannel("A").get_flattened_data()),
            [0, 0, round(118 * 255 / 245), 255],
        )

    def test_multiplies_existing_alpha(self) -> None:
        source = Image.new("RGBA", (1, 1), (128, 128, 128, 128))

        result = luminance_to_alpha(source)

        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255, 64))

    def test_does_not_mutate_source(self) -> None:
        source = Image.new("RGBA", (1, 1), (20, 40, 60, 80))
        original = source.tobytes()

        luminance_to_alpha(source, invert=True, black_point=4)

        self.assertEqual(source.mode, "RGBA")
        self.assertEqual(source.tobytes(), original)

    def test_rejects_invalid_black_point(self) -> None:
        source = Image.new("L", (1, 1), 255)

        for value in (-1, 255):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    luminance_to_alpha(source, black_point=value)
        with self.assertRaises(TypeError):
            luminance_to_alpha(source, black_point=1.5)  # type: ignore[arg-type]


class CenterOfMassTests(unittest.TestCase):
    def test_uses_alpha_weights_and_pixel_center_coordinates(self) -> None:
        source = Image.new("RGBA", (3, 1), (255, 255, 255, 0))
        source.putalpha(Image.frombytes("L", (3, 1), bytes([0, 1, 3])))

        self.assertEqual(center_of_mass(source), (2.25, 0.5))

    def test_uses_luminance_without_alpha(self) -> None:
        source = Image.new("L", (2, 1))
        source.putdata([0, 255])

        self.assertEqual(center_of_mass(source), (1.5, 0.5))

    def test_rejects_empty_artwork(self) -> None:
        with self.assertRaises(ValueError):
            center_of_mass(Image.new("L", (2, 2), 0))


class FitTests(unittest.TestCase):
    def test_trims_padding_and_upscales_longest_edge(self) -> None:
        source = Image.new("RGBA", (10, 8), (0, 0, 0, 0))
        source.paste((255, 0, 0, 255), (2, 1, 6, 4))

        result = fit(source, canvas_size=100, occupancy=0.5)

        self.assertEqual(result.size, (50, 38))
        self.assertEqual(source.size, (10, 8))

    def test_downscales_longest_edge(self) -> None:
        source = Image.new("RGBA", (100, 20), (255, 255, 255, 255))

        result = fit(source, canvas_size=50, occupancy=0.5)

        self.assertEqual(result.size, (25, 5))

    def test_rejects_empty_artwork(self) -> None:
        with self.assertRaises(ValueError):
            fit(Image.new("RGBA", (2, 2), (0, 0, 0, 0)))

    def test_validates_settings(self) -> None:
        source = Image.new("L", (1, 1), 255)

        for canvas_size in (0, -1):
            with self.subTest(canvas_size=canvas_size):
                with self.assertRaises(ValueError):
                    fit(source, canvas_size=canvas_size)
        with self.assertRaises(TypeError):
            fit(source, canvas_size=2.5)  # type: ignore[arg-type]

        for occupancy in (0, -0.1, 1.1, math.inf, math.nan):
            with self.subTest(occupancy=occupancy):
                with self.assertRaises(ValueError):
                    fit(source, occupancy=occupancy)
        with self.assertRaises(TypeError):
            fit(source, occupancy="0.5")  # type: ignore[arg-type]


class CenterByMassTests(unittest.TestCase):
    def test_centers_mass_when_artwork_fits(self) -> None:
        source = Image.new("RGBA", (2, 2), (255, 255, 255, 255))

        result = center_by_mass(source, canvas_size=10)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (10, 10))
        self.assertEqual(result.getchannel("A").getbbox(), (4, 4, 6, 6))
        self.assertEqual(center_of_mass(result), (5.0, 5.0))

    def test_shifts_asymmetric_artwork_to_prevent_clipping(self) -> None:
        source = Image.new("RGBA", (8, 2), (255, 255, 255, 0))
        alpha = Image.new("L", source.size)
        alpha.putdata(([255] + [1] * 7) * 2)
        source.putalpha(alpha)
        original_alpha_total = sum(
            source.getchannel("A").get_flattened_data()
        )

        result = center_by_mass(source, canvas_size=10)

        self.assertEqual(result.getchannel("A").getbbox(), (2, 4, 10, 6))
        self.assertEqual(
            sum(result.getchannel("A").get_flattened_data()),
            original_alpha_total,
        )
        self.assertLess(center_of_mass(result)[0], 5)

    def test_rejects_artwork_larger_than_canvas(self) -> None:
        source = Image.new("RGBA", (11, 1), (255, 255, 255, 255))

        with self.assertRaises(ValueError):
            center_by_mass(source, canvas_size=10)


class FillBackgroundTests(unittest.TestCase):
    def test_composites_transparency_over_solid_color(self) -> None:
        source = Image.new("RGBA", (3, 1))
        source.putdata(
            [
                (255, 255, 255, 0),
                (255, 255, 255, 128),
                (255, 255, 255, 255),
            ]
        )

        result = fill_background(source, color="#000000")

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(
            list(result.get_flattened_data()),
            [
                (0, 0, 0, 255),
                (128, 128, 128, 255),
                (255, 255, 255, 255),
            ],
        )

    def test_does_not_mutate_source(self) -> None:
        source = Image.new("RGBA", (1, 1), (10, 20, 30, 40))
        original = source.tobytes()

        fill_background(source, color=(100, 110, 120))

        self.assertEqual(source.tobytes(), original)


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_writes_square_rgba_png(self) -> None:
        source = Image.new("L", (8, 4), 0)
        source.paste(255, (1, 1, 7, 3))

        artwork = luminance_to_alpha(source)
        artwork = fit(artwork, canvas_size=512, occupancy=0.75)
        logo = center_by_mass(artwork, canvas_size=512)
        output = fill_background(logo, color="black")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "logo.png"
            output.save(destination, format="PNG")
            with Image.open(destination) as result:
                result.load()
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (512, 512))
                self.assertEqual(result.getpixel((0, 0)), (0, 0, 0, 255))
                self.assertEqual(
                    result.getchannel("A").getextrema(),
                    (255, 255),
                )


if __name__ == "__main__":
    unittest.main()
