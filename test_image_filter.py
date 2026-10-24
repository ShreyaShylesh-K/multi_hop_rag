from io import BytesIO

from PIL import Image

from image_filter import ImageLimits, check_image, is_image_file


def make_png(width, height):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_within_limits_passes():
    width, height, problems = check_image(make_png(800, 600), ImageLimits())
    assert (width, height) == (800, 600)
    assert problems == []


def test_too_small_image_is_rejected():
    _, _, problems = check_image(make_png(50, 600), ImageLimits())
    assert problems == ["width 50px is below the 100px minimum"]


def test_too_large_dimensions_are_rejected():
    limits = ImageLimits(max_width=500, max_height=500)
    _, _, problems = check_image(make_png(800, 900), limits)
    assert len(problems) == 2


def test_file_size_limit():
    data = make_png(800, 600)
    limits = ImageLimits(max_size_mb=len(data) / (1024 * 1024) / 2)
    _, _, problems = check_image(data, limits)
    assert len(problems) == 1 and "file size" in problems[0]


def test_unreadable_file_is_rejected():
    _, _, problems = check_image(b"not an image", ImageLimits())
    assert problems == ["the file is not a readable image"]


def test_is_image_file():
    assert is_image_file("photo.JPG")
    assert not is_image_file("notes.pdf")
