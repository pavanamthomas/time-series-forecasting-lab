"""Causal forecast pipeline: training data at origin t never includes times > t.

These tests are designed to fail if a caller (or a future refactor) fits on
the full sample, uses a global mean/last value, or lets sklearn-style splits
put a future index into the training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecast.validation import (
    LinearTrendForecaster,
    MeanForecaster,
    NaiveForecaster,
    PointForecast,
    evaluate_forecaster,
    expanding_origins,
    forecast_from_origin,
    iter_causal_windows,
    rolling_origins,
    sklearn_expanding_train_indices,
    slice_train_test,
    train_slice_for_origin,
)


SENTINEL = 1.0e9


def test_slice_train_test_excludes_future_positions() -> None:
    y = pd.Series(np.arange(20, dtype=float))
    train, test = slice_train_test(y, origin=9, horizon=3)
    assert list(train.index) == list(range(0, 10))
    assert list(test.index) == list(range(10, 13))
    assert train.index.max() < test.index.min()
    assert 19 not in train.index


def test_expanding_origins_last_train_leaves_room_for_horizon() -> None:
    origins = expanding_origins(n_obs=30, min_train=10, horizon=4, step=1)
    assert origins[0] == 9
    assert origins[-1] == 25  # test uses 26,27,28,29
    for origin in origins:
        assert origin + 4 < 30 or origin + 4 == 29 or origin + 4 <= 29
        assert origin + 1 + 4 <= 30


def test_rolling_origins_fixed_width_never_starts_negative() -> None:
    origins = rolling_origins(n_obs=25, train_size=8, horizon=2, step=1)
    y = pd.Series(np.arange(25, dtype=float))
    for origin in origins:
        train = train_slice_for_origin(y, int(origin), scheme="rolling", train_size=8)
        assert len(train) == 8
        assert train.index.max() == origin
        assert train.index.min() == origin - 7


def test_causal_windows_have_strict_time_order() -> None:
    for train_end, test_start, test_end in iter_causal_windows(50, 12, 4, 3):
        assert train_end == test_start
        assert test_end > test_start
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        assert train_idx.max() < test_idx.min()


def test_sklearn_time_series_split_never_peeks() -> None:
    for train_idx, test_idx in sklearn_expanding_train_indices(40, n_splits=6, test_size=2):
        assert train_idx.max() < test_idx.min()
        assert set(train_idx).isdisjoint(set(test_idx))


def test_mean_forecaster_ignores_future_sentinel() -> None:
    """If fit() saw the post-origin sentinel, the forecast mean would be huge.

    A leaky implementation of the form ``model.fit(y)`` instead of
    ``model.fit(y.iloc[: origin + 1])`` fails this test.
    """

    y = pd.Series(1.0, index=range(40))
    y.iloc[25:] = SENTINEL
    fc, train, test = forecast_from_origin(y, MeanForecaster(), origin=20, horizon=3)
    np.testing.assert_allclose(fc.point, np.repeat(1.0, 3), atol=1e-12)
    assert SENTINEL not in train.to_numpy()
    assert train.index.max() == 20
    assert test.index.min() == 21


def test_naive_forecaster_uses_origin_value_not_sample_tail() -> None:
    y = pd.Series(np.arange(40, dtype=float))
    y.iloc[30:] = SENTINEL
    fc, train, _ = forecast_from_origin(y, NaiveForecaster(), origin=20, horizon=2)
    assert fc.point[0] == pytest.approx(y.iloc[20])
    assert SENTINEL not in train.to_numpy()
    assert fc.point[0] != SENTINEL


def test_linear_trend_fit_does_not_see_future_level_shift() -> None:
    t = np.arange(50, dtype=float)
    y = pd.Series(2.0 + 0.1 * t)
    y.iloc[40:] = SENTINEL
    fc, train, _ = forecast_from_origin(
        y, LinearTrendForecaster(), origin=30, horizon=5
    )
    expected = 2.0 + 0.1 * np.arange(31, 36, dtype=float)
    np.testing.assert_allclose(fc.point, expected, atol=1e-8)
    assert SENTINEL not in train.to_numpy()


def test_evaluate_forecaster_passes_only_causal_slices() -> None:
    y = pd.Series(np.arange(40, dtype=float))
    lengths: list[int] = []
    last_values: list[float] = []
    max_index: list[int] = []

    class RecordingForecaster:
        name = "recording"

        def fit(self, train: pd.Series) -> RecordingForecaster:
            lengths.append(len(train))
            last_values.append(float(train.iloc[-1]))
            max_index.append(int(train.index.max()))
            self._last = float(train.iloc[-1])
            return self

        def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
            return PointForecast(point=np.repeat(self._last, horizon), name=self.name)

    table = evaluate_forecaster(
        y, RecordingForecaster, min_train=10, horizon=2, step=1, scheme="expanding"
    )
    assert lengths[0] == 10
    assert last_values[0] == pytest.approx(y.iloc[9])
    assert max_index[0] == 9
    assert max_index[-1] == 37
    assert last_values[-1] == pytest.approx(y.iloc[37])
    assert all(m < 39 for m in max_index)
    assert all(length < len(y) for length in lengths)
    # A leaky evaluator that passes the full series would record length 40
    # and last_value 39 at every origin.
    assert not all(length == len(y) for length in lengths)
    assert not all(val == y.iloc[-1] for val in last_values)
    assert table["train_length"].iloc[0] == 10


def test_rolling_scheme_excludes_observations_before_the_window_and_after_origin() -> None:
    y = pd.Series(np.arange(30, dtype=float))
    y.iloc[24:] = SENTINEL
    windows: list[tuple[int, int]] = []

    class Guard:
        name = "guard"

        def fit(self, train: pd.Series) -> Guard:
            start = int(train.index.min())
            origin = int(train.index.max())
            windows.append((start, origin))
            assert len(train) == 8
            assert start == origin - 7
            if origin < 24:
                assert SENTINEL not in train.to_numpy()
            assert int(y.index.max()) not in train.index or origin == int(y.index.max())
            self._mu = float(np.mean(train.to_numpy()))
            return self

        def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
            return PointForecast(point=np.repeat(self._mu, horizon), name=self.name)

    evaluate_forecaster(
        y,
        Guard,
        min_train=8,
        horizon=2,
        step=1,
        scheme="rolling",
        train_size=8,
    )
    assert windows
    assert all(end < 29 for _, end in windows)
