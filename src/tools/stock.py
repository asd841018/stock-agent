from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from twstock import Stock, codes


DEFAULT_LOOKBACK_DAYS = 40
DEFAULT_TECHNICAL_LOOKBACK_DAYS = 120


def resolve_stock_info(stock_query: str) -> dict[str, str]:
    """Resolve a Taiwan stock code or company name to canonical stock info."""
    query = stock_query.strip()
    if not query:
        raise ValueError("stock_query 不能是空的。")

    exact = codes.get(query) or codes.get(query.upper())
    if exact:
        return {
            "code": exact.code,
            "name": exact.name,
            "market": exact.market,
            "group": exact.group,
        }

    normalized = query.lower()
    for info in codes.values():
        if normalized in {info.code.lower(), info.name.lower()}:
            return {
                "code": info.code,
                "name": info.name,
                "market": info.market,
                "group": info.group,
            }

    for info in codes.values():
        if normalized in info.name.lower():
            return {
                "code": info.code,
                "name": info.name,
                "market": info.market,
                "group": info.group,
            }

    raise ValueError(f"找不到股票：{stock_query}。目前工具僅支援 twstock 可解析的台股代碼或名稱。")


def fetch_recent_stock(stock_code: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Stock:
    stock = Stock(stock_code)
    start = date.today() - timedelta(days=max(lookback_days, 10))
    stock.fetch_from(start.year, start.month)
    return stock


def latest_available(values: list[Any]) -> Any:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def validate_period(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} 必須大於 0。")
    return value


def build_price_frame(stock: Stock) -> pd.DataFrame:
    """Build a clean OHLC frame from a twstock Stock object."""
    frame = pd.DataFrame(
        {
            "date": stock.date,
            "close": stock.price,
            "high": stock.high,
            "low": stock.low,
        }
    )
    frame = frame.dropna(subset=["date", "close", "high", "low"]).copy()
    if frame.empty:
        raise ValueError("目前抓不到可用的 OHLC 價格資料。")

    for column in ["close", "high", "low"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["close", "high", "low"]).reset_index(drop=True)


def calculate_kd(
    price_frame: pd.DataFrame,
    period: int = 9,
    k_smoothing: int = 3,
    d_smoothing: int = 3,
) -> pd.DataFrame:
    """Calculate stochastic KD values from an OHLC price frame."""
    period = validate_period(period, "period")
    k_smoothing = validate_period(k_smoothing, "k_smoothing")
    d_smoothing = validate_period(d_smoothing, "d_smoothing")
    if len(price_frame) < period:
        raise ValueError(f"可用價格資料只有 {len(price_frame)} 筆，不足以計算 KD{period}。")

    low_min = price_frame["low"].rolling(window=period, min_periods=period).min()
    high_max = price_frame["high"].rolling(window=period, min_periods=period).max()
    price_range = high_max - low_min
    rsv = ((price_frame["close"] - low_min) / price_range * 100).where(price_range != 0, 50)
    k_value = rsv.ewm(alpha=1 / k_smoothing, adjust=False).mean()
    d_value = k_value.ewm(alpha=1 / d_smoothing, adjust=False).mean()

    return pd.DataFrame(
        {
            "date": price_frame["date"],
            "rsv": rsv,
            "k": k_value,
            "d": d_value,
        }
    )


def calculate_rsi(price_frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Wilder-style RSI values from close prices."""
    period = validate_period(period, "period")
    if len(price_frame) <= period:
        raise ValueError(f"可用價格資料只有 {len(price_frame)} 筆，不足以計算 RSI{period}。")

    delta = price_frame["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss != 0, 100)

    return pd.DataFrame(
        {
            "date": price_frame["date"],
            "rsi": rsi,
        }
    )


def calculate_bollinger_bands(
    price_frame: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands from close prices."""
    period = validate_period(period, "period")
    if std_multiplier <= 0:
        raise ValueError("std_multiplier 必須大於 0。")
    if len(price_frame) < period:
        raise ValueError(f"可用價格資料只有 {len(price_frame)} 筆，不足以計算布林通道{period}。")

    middle = price_frame["close"].rolling(window=period, min_periods=period).mean()
    standard_deviation = price_frame["close"].rolling(window=period, min_periods=period).std()
    upper = middle + standard_deviation * std_multiplier
    lower = middle - standard_deviation * std_multiplier
    bandwidth = (upper - lower) / middle * 100

    return pd.DataFrame(
        {
            "date": price_frame["date"],
            "close": price_frame["close"],
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
        }
    )


def latest_indicator_row(indicator_frame: pd.DataFrame, required_columns: list[str]) -> pd.Series:
    clean_frame = indicator_frame.dropna(subset=required_columns)
    if clean_frame.empty:
        raise ValueError("目前可用價格資料不足，無法取得最新技術指標。")
    return clean_frame.iloc[-1]


def summarize_stock_data(stock: Stock, stock_info: dict[str, str], ma_days: int = 10) -> str:
    prices = [price for price in stock.price if price is not None]
    if not prices:
        raise ValueError(f"{stock_info['code']} 目前抓不到價格資料。")
    if len(prices) < ma_days:
        raise ValueError(
            f"{stock_info['code']} 可用價格資料只有 {len(prices)} 筆，不足以計算 MA{ma_days}。"
        )

    latest_price = latest_available(stock.price)
    latest_date = latest_available(stock.date)
    moving_averages = stock.moving_average(prices, ma_days)
    latest_ma = latest_available(moving_averages)
    previous_price = prices[-2] if len(prices) > 1 else prices[-1]
    price_change = float(latest_price) - float(previous_price)

    return (
        f"股票：{stock_info['name']} ({stock_info['code']})\n"
        f"市場：{stock_info['market']} / 類別：{stock_info['group']}\n"
        f"最新交易日：{latest_date}\n"
        f"最新收盤價：{format_number(latest_price)} TWD\n"
        f"單日變動：{price_change:+.2f} TWD\n"
        f"MA{ma_days}：{format_number(latest_ma)} TWD\n"
        f"最近 {len(prices)} 筆有效價格已載入，可用於進一步分析。"
    )


@tool
def resolve_taiwan_stock(stock_query: str) -> str:
    """當使用者提供模糊股票名稱或代碼時使用。輸入 stock_query，可為台股代碼或中文名稱，
    例如 2330、台積電、鴻海。輸出標準化後的股票名稱、代碼、市場與類別，適合後續工具串接。"""
    info = resolve_stock_info(stock_query)
    return (
        f"{stock_query} 對應到 {info['name']} ({info['code']})，"
        f"市場：{info['market']}，類別：{info['group']}。"
    )


@tool
def get_stock_price(stock_query: str) -> str:
    """當需要快速確認最新可得收盤價時使用。輸入 stock_query，可為台股代碼或中文名稱，
    例如 2330、台積電、鴻海。輸出最新可得交易日與該日收盤價的文字摘要。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(info["code"], lookback_days=20)
    latest_price = latest_available(stock.price)
    latest_date = latest_available(stock.date)
    return (
        f"{info['name']} ({info['code']}) 最新可得交易日 {latest_date} 的收盤價為 "
        f"{format_number(latest_price)} TWD。"
    )


@tool
def get_moving_average(stock_query: str, days: int = 10) -> str:
    """當需要技術面均線資料時使用。輸入 stock_query，可為台股代碼或中文名稱；days 為均線天數，
    常見值如 5、10、20。輸出指定天數的最新移動平均價格文字摘要。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(info["code"], lookback_days=max(40, days * 4))
    prices = [price for price in stock.price if price is not None]
    if len(prices) < days:
        raise ValueError(f"{info['code']} 可用價格資料不足，無法計算 MA{days}。")
    ma_values = stock.moving_average(prices, days)
    latest_ma = latest_available(ma_values)
    return f"{info['name']} ({info['code']}) 的 MA{days} 為 {format_number(latest_ma)} TWD。"


@tool
def get_stock_snapshot(stock_query: str, days: int = 10) -> str:
    """當需要一次取得可做多空判斷的完整行情摘要時使用。輸入 stock_query，可為台股代碼或中文名稱；
    days 為要計算的移動平均天數。輸出包含股票資訊、最新收盤價、單日變化與移動平均的完整文字快照。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(info["code"], lookback_days=max(DEFAULT_LOOKBACK_DAYS, days * 4))
    return summarize_stock_data(stock, info, ma_days=days)


@tool
def get_kd_indicator(
    stock_query: str,
    period: int = 9,
    k_smoothing: int = 3,
    d_smoothing: int = 3,
) -> str:
    """當需要 KD 隨機指標判斷超買超賣或黃金/死亡交叉時使用。輸入 stock_query，可為台股代碼或中文名稱；
    period 預設 9，k_smoothing 與 d_smoothing 預設 3。輸出最新 RSV、K、D 與簡短訊號說明。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(
        info["code"],
        lookback_days=max(DEFAULT_TECHNICAL_LOOKBACK_DAYS, period * 6),
    )
    price_frame = build_price_frame(stock)
    kd_values = calculate_kd(price_frame, period, k_smoothing, d_smoothing)
    latest = latest_indicator_row(kd_values, ["rsv", "k", "d"])
    signal = "偏中立"
    if latest["k"] > 80 and latest["d"] > 80:
        signal = "可能超買"
    elif latest["k"] < 20 and latest["d"] < 20:
        signal = "可能超賣"
    elif latest["k"] > latest["d"]:
        signal = "K 值高於 D 值，短線動能偏多"
    elif latest["k"] < latest["d"]:
        signal = "K 值低於 D 值，短線動能偏弱"

    return (
        f"{info['name']} ({info['code']}) 最新交易日 {latest['date']} 的 KD{period}："
        f"RSV {format_number(latest['rsv'])}、K {format_number(latest['k'])}、"
        f"D {format_number(latest['d'])}。訊號：{signal}。"
    )


@tool
def get_rsi_indicator(stock_query: str, period: int = 14) -> str:
    """當需要 RSI 相對強弱指標判斷超買超賣或股價動能時使用。輸入 stock_query，可為台股代碼或中文名稱；
    period 預設 14。輸出最新 RSI 與簡短訊號說明。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(
        info["code"],
        lookback_days=max(DEFAULT_TECHNICAL_LOOKBACK_DAYS, period * 6),
    )
    price_frame = build_price_frame(stock)
    rsi_values = calculate_rsi(price_frame, period)
    latest = latest_indicator_row(rsi_values, ["rsi"])
    signal = "偏中立"
    if latest["rsi"] >= 70:
        signal = "可能超買"
    elif latest["rsi"] <= 30:
        signal = "可能超賣"
    elif latest["rsi"] >= 50:
        signal = "動能略偏多"
    else:
        signal = "動能略偏弱"

    return (
        f"{info['name']} ({info['code']}) 最新交易日 {latest['date']} 的 RSI{period} 為 "
        f"{format_number(latest['rsi'])}。訊號：{signal}。"
    )


@tool
def get_bollinger_bands(
    stock_query: str,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> str:
    """當需要布林通道判斷價格相對區間、突破或跌破時使用。輸入 stock_query，可為台股代碼或中文名稱；
    period 預設 20，std_multiplier 預設 2。輸出最新上軌、中軌、下軌、帶寬與收盤價位置。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(
        info["code"],
        lookback_days=max(DEFAULT_TECHNICAL_LOOKBACK_DAYS, period * 6),
    )
    price_frame = build_price_frame(stock)
    bollinger_values = calculate_bollinger_bands(price_frame, period, std_multiplier)
    latest = latest_indicator_row(bollinger_values, ["upper", "middle", "lower", "bandwidth"])

    signal = "位於通道內"
    if latest["close"] >= latest["upper"]:
        signal = "收盤價觸及或突破上軌，留意過熱或趨勢延伸"
    elif latest["close"] <= latest["lower"]:
        signal = "收盤價觸及或跌破下軌，留意轉弱或反彈機會"
    elif latest["close"] >= latest["middle"]:
        signal = "收盤價在中軌之上，區間位置偏強"
    else:
        signal = "收盤價在中軌之下，區間位置偏弱"

    return (
        f"{info['name']} ({info['code']}) 最新交易日 {latest['date']} 的布林通道{period}："
        f"上軌 {format_number(latest['upper'])}、中軌 {format_number(latest['middle'])}、"
        f"下軌 {format_number(latest['lower'])}、帶寬 {format_number(latest['bandwidth'])}%。"
        f"最新收盤價 {format_number(latest['close'])}，訊號：{signal}。"
    )


@tool
def get_technical_indicators(stock_query: str) -> str:
    """當需要一次取得常用技術指標摘要時使用。輸入 stock_query，可為台股代碼或中文名稱。
    輸出最新 KD(9,3,3)、RSI14、布林通道20，以及簡短技術面解讀。"""
    info = resolve_stock_info(stock_query)
    stock = fetch_recent_stock(info["code"], lookback_days=DEFAULT_TECHNICAL_LOOKBACK_DAYS)
    price_frame = build_price_frame(stock)

    kd_latest = latest_indicator_row(calculate_kd(price_frame), ["rsv", "k", "d"])
    rsi_latest = latest_indicator_row(calculate_rsi(price_frame), ["rsi"])
    bollinger_latest = latest_indicator_row(
        calculate_bollinger_bands(price_frame),
        ["upper", "middle", "lower", "bandwidth"],
    )

    return (
        f"股票：{info['name']} ({info['code']})\n"
        f"最新交易日：{bollinger_latest['date']}\n"
        f"KD9：K {format_number(kd_latest['k'])}、D {format_number(kd_latest['d'])}、"
        f"RSV {format_number(kd_latest['rsv'])}\n"
        f"RSI14：{format_number(rsi_latest['rsi'])}\n"
        f"布林通道20：上軌 {format_number(bollinger_latest['upper'])}、"
        f"中軌 {format_number(bollinger_latest['middle'])}、"
        f"下軌 {format_number(bollinger_latest['lower'])}、"
        f"帶寬 {format_number(bollinger_latest['bandwidth'])}%\n"
        f"最新收盤價：{format_number(bollinger_latest['close'])} TWD。"
    )


def get_stock_tools():
    return [
        resolve_taiwan_stock,
        get_stock_price,
        get_moving_average,
        get_kd_indicator,
        get_rsi_indicator,
        get_bollinger_bands,
        get_technical_indicators,
        get_stock_snapshot,
    ]
