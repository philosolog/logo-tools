"""Composable Pillow primitives for logo-editing macros."""

from .background import fill_background
from .core import (
    center_by_mass,
    center_of_mass,
    fit,
    luminance_to_alpha,
    square_crop,
)

__all__ = [
    "center_by_mass",
    "center_of_mass",
    "fill_background",
    "fit",
    "luminance_to_alpha",
    "square_crop",
]
