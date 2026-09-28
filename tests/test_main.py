import numpy as np
import pytest

from main import ALGORITHMS


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_returns_something_save_image_can_write(name):
    # save_image clips to [0, 1], so a result outside it would be saved wrong
    # without any error: a signed map, for instance, would lose its negatives
    image = np.random.default_rng(0).random((24, 24, 3))
    result = ALGORITHMS[name](image)
    assert result.shape in {(24, 24), (24, 24, 3)}
    assert result.dtype == np.float64
    assert result.min() >= 0.0 and result.max() <= 1.0
