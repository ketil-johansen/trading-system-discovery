"""Integration tests for the data quality pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsd.data.quality import save_report, validate_market


@pytest.mark.integration
class TestQualityPipeline:
    """End-to-end quality validation using real downloaded data."""

    def test_validate_omxs30_market(self) -> None:
        """Validate real OMXS30 data and check report structure."""
        data_dir = Path("data")
        market_dir = data_dir / "raw" / "omxs30"
        if not market_dir.exists() or not list(market_dir.glob("*.parquet")):
            pytest.skip("OMXS30 data not downloaded — run download_all_data.py first")

        report = validate_market("omxs30", data_dir)

        assert report.market_key == "omxs30"
        assert report.total_stocks > 0
        assert report.pass_count + report.warn_count + report.fail_count == report.total_stocks
        assert len(report.stock_results) == report.total_stocks

        # Every stock result should have valid fields
        for sr in report.stock_results:
            assert sr.status in ("PASS", "WARN", "FAIL")
            assert sr.row_count >= 0
            assert sr.ticker != ""

        # Save report and verify output
        path = save_report(report, data_dir)
        assert path.exists()
        assert path.name == "omxs30_quality.json"
