import unittest
from datetime import date, timedelta

import pandas as pd

from src.tools.stock import (
    build_price_frame,
    calculate_bollinger_bands,
    calculate_kd,
    calculate_rsi,
    latest_available,
    latest_indicator_row,
    resolve_stock_info,
    summarize_stock_data,
)


class FakeStock:
    def __init__(self):
        self.price = [100.0, 101.5, 103.0]
        self.date = [date(2026, 4, 21), date(2026, 4, 22), date(2026, 4, 23)]

    def moving_average(self, prices, days):
        self.last_args = (prices, days)
        return [None, None, 101.5]


class FakeOhlcStock:
    def __init__(self):
        self.date = [date(2026, 1, 1) + timedelta(days=index) for index in range(30)]
        self.price = [100.0 + index for index in range(30)]
        self.high = [101.0 + index for index in range(30)]
        self.low = [99.0 + index for index in range(30)]


def build_sample_price_frame():
    return pd.DataFrame(
        {
            "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(30)],
            "close": [100.0 + index for index in range(30)],
            "high": [101.0 + index for index in range(30)],
            "low": [99.0 + index for index in range(30)],
        }
    )


class StockToolTests(unittest.TestCase):
    def test_resolve_stock_info_supports_code(self):
        info = resolve_stock_info("2330")
        self.assertEqual(info["code"], "2330")
        self.assertEqual(info["name"], "台積電")

    def test_resolve_stock_info_supports_name(self):
        info = resolve_stock_info("台積電")
        self.assertEqual(info["code"], "2330")

    def test_latest_available_skips_none(self):
        self.assertEqual(latest_available([None, 1, None, 3]), 3)
        self.assertIsNone(latest_available([None, None]))

    def test_summarize_stock_data_formats_snapshot(self):
        stock = FakeStock()
        info = {
            "code": "2330",
            "name": "台積電",
            "market": "上市",
            "group": "半導體業",
        }
        summary = summarize_stock_data(stock, info, ma_days=3)
        self.assertIn("台積電 (2330)", summary)
        self.assertIn("最新收盤價：103.00 TWD", summary)
        self.assertIn("MA3：101.50 TWD", summary)

    def test_build_price_frame_keeps_clean_ohlc_values(self):
        frame = build_price_frame(FakeOhlcStock())
        self.assertEqual(len(frame), 30)
        self.assertEqual(frame.iloc[-1]["close"], 129.0)
        self.assertEqual(frame.iloc[-1]["high"], 130.0)
        self.assertEqual(frame.iloc[-1]["low"], 128.0)

    def test_calculate_kd_returns_latest_values(self):
        latest = latest_indicator_row(calculate_kd(build_sample_price_frame()), ["rsv", "k", "d"])
        self.assertAlmostEqual(latest["rsv"], 90.0)
        self.assertAlmostEqual(latest["k"], 90.0)
        self.assertAlmostEqual(latest["d"], 90.0)

    def test_calculate_rsi_returns_wilder_rsi(self):
        latest = latest_indicator_row(calculate_rsi(build_sample_price_frame()), ["rsi"])
        self.assertAlmostEqual(latest["rsi"], 100.0)

    def test_calculate_bollinger_bands_returns_latest_bands(self):
        latest = latest_indicator_row(
            calculate_bollinger_bands(build_sample_price_frame()),
            ["upper", "middle", "lower", "bandwidth"],
        )
        self.assertAlmostEqual(latest["middle"], 119.5)
        self.assertAlmostEqual(latest["upper"], 131.33215956619924)
        self.assertAlmostEqual(latest["lower"], 107.66784043380076)
        self.assertAlmostEqual(latest["bandwidth"], 19.80277751765563)


if __name__ == "__main__":
    unittest.main()
