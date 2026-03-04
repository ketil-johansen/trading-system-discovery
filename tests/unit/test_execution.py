"""Unit tests for execution timing helpers."""

from __future__ import annotations

import pandas as pd

from tsd.strategy.execution import (
    ENTRY_TIMING,
    EXIT_TIMING_INDICATOR,
    EXIT_TIMING_LIMIT,
    EXIT_TIMING_TIME,
    check_limit_exit,
    shift_to_next_open,
)

# ---------------------------------------------------------------------------
# shift_to_next_open
# ---------------------------------------------------------------------------


class TestShiftToNextOpen:
    """Tests for signal shifting."""

    def test_shifts_forward_by_one(self) -> None:
        signals = pd.Series([False, True, False, True])
        shifted = shift_to_next_open(signals)
        assert list(shifted) == [False, False, True, False]

    def test_first_bar_always_false(self) -> None:
        signals = pd.Series([True, True, True])
        shifted = shift_to_next_open(signals)
        assert shifted.iloc[0] is False or not shifted.iloc[0]

    def test_last_signal_dropped(self) -> None:
        signals = pd.Series([False, False, True])
        shifted = shift_to_next_open(signals)
        # Last True at index 2 gets shifted out (would be index 3)
        assert not shifted.iloc[-1]

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        signals = pd.Series([True, False, True, False], index=idx)
        shifted = shift_to_next_open(signals)
        assert list(shifted.index) == list(idx)

    def test_returns_bool_dtype(self) -> None:
        signals = pd.Series([True, False, True])
        shifted = shift_to_next_open(signals)
        assert shifted.dtype == bool

    def test_empty_series(self) -> None:
        signals = pd.Series([], dtype=bool)
        shifted = shift_to_next_open(signals)
        assert len(shifted) == 0


# ---------------------------------------------------------------------------
# check_limit_exit
# ---------------------------------------------------------------------------


class TestCheckLimitExit:
    """Tests for intraday limit exit checking."""

    def test_no_hit(self) -> None:
        result = check_limit_exit(high=105.0, low=95.0, open_price=100.0, stop_level=90.0, target_level=110.0)
        assert result == (None, None)

    def test_stop_only_hit(self) -> None:
        exit_type, exit_price = check_limit_exit(
            high=102.0,
            low=94.0,
            open_price=100.0,
            stop_level=95.0,
            target_level=110.0,
        )
        assert exit_type == "stop_loss"
        assert exit_price == 95.0

    def test_target_only_hit(self) -> None:
        exit_type, exit_price = check_limit_exit(
            high=112.0,
            low=99.0,
            open_price=100.0,
            stop_level=90.0,
            target_level=110.0,
        )
        assert exit_type == "take_profit"
        assert exit_price == 110.0

    def test_both_hit_stop_wins(self) -> None:
        # Wide range bar: low hits stop, high hits target
        exit_type, exit_price = check_limit_exit(
            high=115.0,
            low=85.0,
            open_price=100.0,
            stop_level=90.0,
            target_level=110.0,
        )
        assert exit_type == "stop_loss"
        assert exit_price == 90.0

    def test_both_hit_open_exceeds_target(self) -> None:
        # Open gaps above target — take profit wins
        exit_type, exit_price = check_limit_exit(
            high=115.0,
            low=85.0,
            open_price=112.0,
            stop_level=90.0,
            target_level=110.0,
        )
        assert exit_type == "take_profit"
        assert exit_price == 112.0

    def test_both_hit_open_equals_target(self) -> None:
        exit_type, exit_price = check_limit_exit(
            high=115.0,
            low=85.0,
            open_price=110.0,
            stop_level=90.0,
            target_level=110.0,
        )
        assert exit_type == "take_profit"
        assert exit_price == 110.0

    def test_none_stop_level(self) -> None:
        exit_type, exit_price = check_limit_exit(
            high=112.0,
            low=85.0,
            open_price=100.0,
            stop_level=None,
            target_level=110.0,
        )
        assert exit_type == "take_profit"
        assert exit_price == 110.0

    def test_none_target_level(self) -> None:
        exit_type, exit_price = check_limit_exit(
            high=102.0,
            low=94.0,
            open_price=100.0,
            stop_level=95.0,
            target_level=None,
        )
        assert exit_type == "stop_loss"
        assert exit_price == 95.0

    def test_both_none(self) -> None:
        result = check_limit_exit(high=120.0, low=80.0, open_price=100.0, stop_level=None, target_level=None)
        assert result == (None, None)

    def test_stop_at_exact_low(self) -> None:
        exit_type, _ = check_limit_exit(
            high=105.0,
            low=95.0,
            open_price=100.0,
            stop_level=95.0,
            target_level=110.0,
        )
        assert exit_type == "stop_loss"

    def test_target_at_exact_high(self) -> None:
        exit_type, _ = check_limit_exit(
            high=110.0,
            low=99.0,
            open_price=100.0,
            stop_level=90.0,
            target_level=110.0,
        )
        assert exit_type == "take_profit"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for execution timing constants."""

    def test_entry_timing(self) -> None:
        assert ENTRY_TIMING == "next_open"

    def test_exit_timing_limit(self) -> None:
        assert EXIT_TIMING_LIMIT == "intraday"

    def test_exit_timing_indicator(self) -> None:
        assert EXIT_TIMING_INDICATOR == "next_open"

    def test_exit_timing_time(self) -> None:
        assert EXIT_TIMING_TIME == "at_open"
