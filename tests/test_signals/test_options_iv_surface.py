from __future__ import annotations

import pandas as pd

from src.signals import options_signals


def _chain(iv_shift: float = 0.0, put_extra: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [21900, 22000, 22100, 21900, 22000, 22100],
            "option_type": ["PE", "PE", "PE", "CE", "CE", "CE"],
            "iv": [
                18.0 + iv_shift + put_extra,
                17.0 + iv_shift,
                16.0 + iv_shift,
                15.0 + iv_shift,
                14.0 + iv_shift,
                13.0 + iv_shift,
            ],
        }
    )


def test_iv_surface_parallel_shift_returns_average_iv_move():
    prev = _chain(iv_shift=0.0)
    curr = _chain(iv_shift=2.0)
    shift = options_signals.iv_surface_parallel_shift(prev, curr)
    assert shift == 2.0


def test_iv_surface_tilt_change_captures_put_wing_richening():
    prev = _chain(iv_shift=0.0, put_extra=0.0)
    curr = _chain(iv_shift=0.0, put_extra=1.5)
    tilt_change = options_signals.iv_surface_tilt_change(
        prev,
        curr,
        previous_underlying=22000.0,
        current_underlying=22000.0,
    )
    assert tilt_change > 0


def test_iv_surface_change_has_expected_columns():
    prev = _chain(iv_shift=0.0)
    curr = _chain(iv_shift=1.0)
    delta = options_signals.iv_surface_change(prev, curr)
    assert set(delta.columns) == {"strike", "option_type", "iv_prev", "iv_curr", "iv_change"}
    assert len(delta) == 6
