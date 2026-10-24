from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp")


@dataclass
class ImageLimits:
    """Allowed image dimensions (pixels) and file size (megabytes)."""

    min_width: int = 100
    min_height: int = 100
    max_width: int = 4096
    max_height: int = 4096
    max_size_mb: float = 5.0


def is_image_file(file_name):
    return file_name.lower().rsplit(".", 1)[-1] in IMAGE_EXTENSIONS


def check_image(file_bytes, limits):
    """Return (width, height, problems) for an image.

    problems is an empty list when the image passes every limit.
    """
    problems = []

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > limits.max_size_mb:
        problems.append(
            f"file size {size_mb:.2f} MB is over the {limits.max_size_mb:g} MB limit"
        )

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        problems.append("the file is not a readable image")
        return None, None, problems

    if width < limits.min_width:
        problems.append(f"width {width}px is below the {limits.min_width}px minimum")
    if width > limits.max_width:
        problems.append(f"width {width}px is above the {limits.max_width}px maximum")
    if height < limits.min_height:
        problems.append(
            f"height {height}px is below the {limits.min_height}px minimum"
        )
    if height > limits.max_height:
        problems.append(
            f"height {height}px is above the {limits.max_height}px maximum"
        )

    return width, height, problems
