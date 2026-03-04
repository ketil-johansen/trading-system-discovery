"""Tests for tsd.data.constituents module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsd.data.constituents import load_constituents, save_constituents


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample constituent CSV for testing."""
    csv_dir = tmp_path / "constituents"
    csv_dir.mkdir()
    csv_path = csv_dir / "test_market.csv"
    csv_path.write_text("ticker,name,yahoo_ticker\nAAPL,Apple Inc.,AAPL\nMSFT,Microsoft Corp.,MSFT\n")
    return tmp_path


@pytest.mark.unit
class TestLoadConstituents:
    """Tests for load_constituents."""

    def test_reads_csv_correctly(self, sample_csv: Path) -> None:
        df = load_constituents("test_market", sample_csv)
        assert len(df) == 2
        assert list(df.columns) == ["ticker", "name", "yahoo_ticker"]
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        (tmp_path / "constituents").mkdir()
        with pytest.raises(FileNotFoundError):
            load_constituents("nonexistent", tmp_path)

    def test_validates_required_columns(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "constituents"
        csv_dir.mkdir()
        (csv_dir / "bad.csv").write_text("ticker,wrong_col\nAAPL,foo\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_constituents("bad", tmp_path)


@pytest.mark.unit
class TestSaveConstituents:
    """Tests for save_constituents."""

    def test_round_trips(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["VOLV-B", "ERIC-B"],
                "name": ["AB Volvo B", "Ericsson B"],
                "yahoo_ticker": ["VOLV-B.ST", "ERIC-B.ST"],
            }
        )
        save_constituents(df, "test_round_trip", tmp_path)
        loaded = load_constituents("test_round_trip", tmp_path)
        assert len(loaded) == 2
        assert loaded.iloc[0]["ticker"] == "VOLV-B"
        assert loaded.iloc[1]["yahoo_ticker"] == "ERIC-B.ST"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "name": ["Test"],
                "yahoo_ticker": ["A"],
            }
        )
        path = save_constituents(df, "new_market", tmp_path)
        assert path.exists()
