import numpy as np

from core.image_io import load_image, save_image


def test_save_then_load_roundtrip(tmp_path):
    image = np.random.default_rng(0).random((8, 10, 3))
    path = tmp_path / "img.png"
    save_image(image, path)
    loaded = load_image(path)
    assert loaded.shape == (8, 10, 3)
    assert np.allclose(loaded, image, atol=1 / 255)


def test_load_gray_returns_2d(tmp_path):
    path = tmp_path / "img.png"
    save_image(np.ones((4, 6, 3)), path)
    loaded = load_image(path, gray=True)
    assert loaded.shape == (4, 6)
    assert np.allclose(loaded, 1.0)


def test_load_values_in_unit_range(tmp_path):
    path = tmp_path / "img.png"
    save_image(np.random.default_rng(1).random((5, 5)), path)
    loaded = load_image(path)
    assert loaded.dtype == np.float64
    assert loaded.min() >= 0.0 and loaded.max() <= 1.0


def test_save_clips_out_of_range_values(tmp_path):
    path = tmp_path / "img.png"
    save_image(np.array([[-1.0, 2.0]]), path)
    assert np.array_equal(load_image(path, gray=True), [[0.0, 1.0]])


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "img.png"
    save_image(np.zeros((2, 2)), path)
    assert path.exists()


def test_to_display_centers_zero_on_mid_gray():
    from core.image_io import to_display

    displayed = to_display(np.array([[-1.0, 0.0, 1.0]]))
    assert np.array_equal(displayed, [[0.0, 0.5, 1.0]])


def test_to_display_of_a_flat_image_is_mid_gray():
    from core.image_io import to_display

    assert np.allclose(to_display(np.zeros((3, 3))), 0.5)
