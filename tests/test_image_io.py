import numpy as np
import pytest

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

def test_to_heatmap_puts_gray_at_zero():
    from core.image_io import to_heatmap

    mid = to_heatmap(np.zeros((2, 2)))
    assert np.allclose(mid, np.array([0xF0, 0xEF, 0xEC]) / 255.0, atol=1e-6)
    assert mid.shape == (2, 2, 3)  # one channel in, three out


def test_to_heatmap_clips_outside_the_range():
    from core.image_io import to_heatmap

    assert np.allclose(to_heatmap(np.array([[-5.0]])), to_heatmap(np.array([[-1.0]])))
    assert np.allclose(to_heatmap(np.array([[5.0]])), to_heatmap(np.array([[1.0]])))


def test_to_heatmap_poles_are_blue_below_and_red_above():
    from core.image_io import to_heatmap

    ramp = to_heatmap(np.array([[-1.0, 1.0]]))
    blue, red = ramp[0, 0], ramp[0, 1]
    assert blue[2] > blue[0]  # more blue than red
    assert red[0] > red[2]  # more red than blue


def test_to_heatmap_is_symmetric_around_zero():
    from core.image_io import to_heatmap

    # a value as far below 0 as another is above gets the same intensity
    below, above = to_heatmap(np.array([[-0.6]]))[0, 0], to_heatmap(np.array([[0.6]]))[0, 0]
    assert np.isclose(np.abs(below - below.mean()).sum(), np.abs(above - above.mean()).sum(), rtol=0.3)


def test_to_heatmap_respects_the_image_contract():
    from core.image_io import to_heatmap

    out = to_heatmap(np.random.default_rng(0).random((6, 7)) * 2 - 1)
    assert out.shape == (6, 7, 3)
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_draw_circles_draws_a_ring_at_the_radius():
    from core.image_io import draw_circles

    out = draw_circles(np.zeros((21, 21)), np.array([[10.0, 10.0, 5.0]]))
    red = np.all(out == [1.0, 0.0, 0.0], axis=-1)
    yy, xx = np.nonzero(red)
    assert red.any()
    assert np.allclose(np.hypot(yy - 10, xx - 10), 5.0, atol=0.5)


def test_draw_circles_leaves_the_inside_and_the_outside_untouched():
    from core.image_io import draw_circles

    out = draw_circles(np.full((21, 21), 0.3), np.array([[10.0, 10.0, 5.0]]))
    assert np.all(out[10, 10] == 0.3)  # center
    assert np.all(out[0, 0] == 0.3)  # far corner


def test_draw_circles_uses_x_as_the_column_and_y_as_the_row():
    from core.image_io import draw_circles

    out = draw_circles(np.zeros((30, 30)), np.array([[20.0, 5.0, 2.0]]))
    red = np.all(out == [1.0, 0.0, 0.0], axis=-1)
    rows, cols = np.nonzero(red)
    assert rows.mean() == pytest.approx(5.0, abs=0.5)
    assert cols.mean() == pytest.approx(20.0, abs=0.5)


def test_draw_circles_clips_a_circle_that_leaves_the_image():
    from core.image_io import draw_circles

    out = draw_circles(np.zeros((10, 10)), np.array([[0.0, 0.0, 4.0], [9.0, 9.0, 30.0]]))
    assert out.shape == (10, 10, 3)
    assert np.all(out[4, 0] == [1.0, 0.0, 0.0])


def test_draw_circles_without_circles_only_converts_to_rgb():
    from core.image_io import draw_circles

    image = np.random.default_rng(0).random((6, 7))
    out = draw_circles(image, np.empty((0, 3)))
    assert out.shape == (6, 7, 3)
    assert np.array_equal(out[..., 0], image) and np.array_equal(out[..., 2], image)


def test_draw_circles_does_not_modify_its_input():
    from core.image_io import draw_circles

    image = np.random.default_rng(0).random((12, 12, 3))
    before = image.copy()
    draw_circles(image, np.array([[6.0, 6.0, 3.0]]))
    assert np.array_equal(image, before)


def test_draw_circles_respects_the_image_contract():
    from core.image_io import draw_circles

    out = draw_circles(np.random.default_rng(0).random((16, 16, 3)), np.array([[8.0, 8.0, 4.0]]), color=(0.0, 1.0, 0.0))
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(out[8, 12] == [0.0, 1.0, 0.0])
