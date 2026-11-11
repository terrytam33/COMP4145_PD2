"""Golden cross backtest, buy-and-hold benchmark, and Yahoo Finance data helpers."""

from pathlib import Path

import pandas as pd
import yfinance as yf

FALLBACK_CSV = Path(__file__).parent / "stock_data.csv"
MAX_PORTFOLIO_STOCKS = 20


def load_price_history(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame with Date index and Close column, source label).

    Tries Yahoo Finance first, then falls back to the bundled stock_data.csv.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame(), ""

    try:
        data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    except Exception:
        data = pd.DataFrame()

    if not data.empty:
        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data[["Close"]].dropna(), "Yahoo Finance"

    fallback = load_fallback_csv(ticker, start, end)
    if not fallback.empty:
        return fallback, "Offline sample data (stock_data.csv)"

    return pd.DataFrame(), ""


def load_fallback_csv(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not FALLBACK_CSV.exists():
        return pd.DataFrame()

    raw = pd.read_csv(FALLBACK_CSV)
    raw = raw[raw["Symbol"] == ticker]
    if raw.empty:
        return pd.DataFrame()

    raw["Date"] = pd.to_datetime(raw["Date"].str[:10])
    raw = raw.set_index("Date").sort_index()
    raw = raw.loc[start:end]
    return raw[["Adj Close"]].rename(columns={"Adj Close": "Close"}).dropna()


def run_golden_cross(prices: pd.DataFrame, short_window: int, long_window: int,
                     initial_capital: float) -> dict:
    """Backtest the golden cross strategy.

    Buy when the short MA crosses above the long MA; sell when it crosses below.
    Trades are executed at the closing price of the signal day.
    """
    df = prices.copy()
    df["Short_MA"] = df["Close"].rolling(short_window).mean()
    df["Long_MA"] = df["Close"].rolling(long_window).mean()

    has_both_ma = df["Long_MA"].notna()
    df["Position"] = ((df["Short_MA"] > df["Long_MA"]) & has_both_ma).astype(int)
    change = df["Position"].diff().fillna(0)
    df["Signal"] = change.map({1: "BUY", -1: "SELL"}).fillna("")

    trades = build_trades(df)

    daily_return = df["Close"].pct_change().fillna(0)
    df["Strategy_Equity"] = initial_capital * (1 + daily_return * df["Position"].shift(1).fillna(0)).cumprod()
    df["BuyHold_Equity"] = initial_capital * df["Close"] / df["Close"].iloc[0]

    stats = build_stats(df, trades, initial_capital)
    return {"data": df, "trades": trades, "stats": stats}


def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        if row["Signal"] == "BUY":
            entry_date, entry_price = date, row["Close"]
            continue
        if row["Signal"] == "SELL" and entry_date is not None:
            rows.append(make_trade(entry_date, entry_price, date, row["Close"], "Closed"))
            entry_date = None

    if entry_date is not None:
        last_date = df.index[-1]
        rows.append(make_trade(entry_date, entry_price, last_date, df["Close"].iloc[-1], "Open (valued at last price)"))

    return pd.DataFrame(rows)


def make_trade(entry_date, entry_price, exit_date, exit_price, status: str) -> dict:
    return {
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Exit Date": exit_date.date(),
        "Exit Price": round(exit_price, 2),
        "Holding Days": (exit_date - entry_date).days,
        "Return (%)": round((exit_price / entry_price - 1) * 100, 2),
        "Status": status,
    }


def build_stats(df: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict:
    final_strategy = df["Strategy_Equity"].iloc[-1]
    final_buy_hold = df["BuyHold_Equity"].iloc[-1]

    total_trades = len(trades)
    winning = int((trades["Return (%)"] > 0).sum()) if total_trades else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": total_trades - winning,
        "win_rate": (winning / total_trades * 100) if total_trades else 0.0,
        "avg_trade_return": trades["Return (%)"].mean() if total_trades else 0.0,
        "best_trade": trades["Return (%)"].max() if total_trades else 0.0,
        "worst_trade": trades["Return (%)"].min() if total_trades else 0.0,
        "initial_capital": initial_capital,
        "final_strategy_value": final_strategy,
        "strategy_return": (final_strategy / initial_capital - 1) * 100,
        "final_buy_hold_value": final_buy_hold,
        "buy_hold_return": (final_buy_hold / initial_capital - 1) * 100,
    }


def get_latest_price(ticker: str) -> float | None:
    """Return the latest close price, or None if the ticker is invalid."""
    try:
        data = yf.Ticker(ticker).history(period="5d")
    except Exception:
        data = pd.DataFrame()

    if not data.empty:
        return float(data["Close"].dropna().iloc[-1])

    fallback = load_fallback_csv(ticker, "1900-01-01", "2100-01-01")
    if not fallback.empty:
        return float(fallback["Close"].iloc[-1])

    return None


def get_stock_news(ticker: str, limit: int = 10) -> list[dict]:
    """Return a list of {title, publisher, published, link, summary} for a ticker."""
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    return [item for item in (parse_news_item(n) for n in raw_news[:limit]) if item["title"]]


def parse_news_item(item: dict) -> dict:
    content = item.get("content")
    if content:
        url = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url", "")
        return {
            "title": content.get("title", ""),
            "publisher": (content.get("provider") or {}).get("displayName", ""),
            "published": (content.get("pubDate") or "")[:10],
            "link": url,
            "summary": content.get("summary", ""),
        }

    published = item.get("providerPublishTime")
    return {
        "title": item.get("title", ""),
        "publisher": item.get("publisher", ""),
        "published": pd.to_datetime(published, unit="s").strftime("%Y-%m-%d") if published else "",
        "link": item.get("link", ""),
        "summary": "",
    }


if __name__ == "__main__":
    prices, source = load_price_history("AAPL", "2021-01-01", "2025-01-01")
    result = run_golden_cross(prices, 20, 50, 10000)
    print(f"Data source: {source}")
    print(result["trades"])
    for key, value in result["stats"].items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
