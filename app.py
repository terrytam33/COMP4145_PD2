"""Streamlit frontend for the stock trading app (COMP4145 PD2 - Idea 1)."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_strategy import (
    MAX_PORTFOLIO_STOCKS,
    get_latest_price,
    get_stock_news,
    load_price_history,
    run_golden_cross,
)

st.set_page_config(page_title="Stock Trading App", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_history(ticker: str, start: str, end: str):
    return load_price_history(ticker, start, end)


@st.cache_data(ttl=900, show_spinner=False)
def cached_price(ticker: str):
    return get_latest_price(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker: str):
    return get_stock_news(ticker)


def main():
    st.sidebar.title("📈 Stock Trading App")
    page = st.sidebar.radio("Page", PAGES)

    st.sidebar.divider()
    st.sidebar.subheader("Strategy settings")
    ticker = st.sidebar.text_input("Stock ticker", value="AAPL").strip().upper()
    start = st.sidebar.date_input("Start date", value=date.today() - timedelta(days=365 * 3))
    end = st.sidebar.date_input("End date", value=date.today())
    short_window = st.sidebar.number_input("Short-term MA (days)", min_value=2, max_value=200, value=20)
    long_window = st.sidebar.number_input("Long-term MA (days)", min_value=5, max_value=400, value=50)
    capital = st.sidebar.number_input("Initial capital (USD)", min_value=100.0, value=10000.0, step=1000.0)

    if page == "💼 Portfolio":
        render_portfolio()
        return
    if page == "📰 Stock News":
        render_news(ticker)
        return

    if not ticker:
        st.warning("Please enter a stock ticker in the sidebar.")
        return
    if start >= end:
        st.error("Start date must be earlier than end date.")
        return
    if short_window >= long_window:
        st.error("Short-term MA must be smaller than long-term MA.")
        return

    with st.spinner(f"Loading {ticker} data..."):
        prices, source = cached_history(ticker, str(start), str(end))

    if prices.empty:
        st.error(f"No price data found for '{ticker}'. Please check the ticker symbol.")
        return
    if len(prices) <= long_window:
        st.error(f"Only {len(prices)} trading days found. Choose a longer date range or a smaller long-term MA.")
        return

    result = run_golden_cross(prices, int(short_window), int(long_window), float(capital))
    st.caption(f"Data source: {source} · {prices.index[0].date()} to {prices.index[-1].date()}")

    if page == "📊 Price Chart":
        render_price_chart(ticker, result, short_window, long_window)
    elif page == "📋 Trade Statistics":
        render_statistics(ticker, result)
    elif page == "🧾 Detailed Trades":
        render_trades(ticker, result)
    elif page == "⚖️ Strategy Comparison":
        render_comparison(ticker, result)


def render_price_chart(ticker, result, short_window, long_window):
    st.title(f"{ticker} Price Chart")
    df = result["data"]
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close price", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Short_MA"], name=f"Short MA ({short_window})", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Long_MA"], name=f"Long MA ({long_window})", line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                             marker=dict(color="green", size=11, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                             marker=dict(color="red", size=11, symbol="triangle-down")))
    fig.update_layout(height=550, xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.info("Buy (green) when the short-term MA crosses above the long-term MA. "
            "Sell (red) when it crosses back below.")


def render_statistics(ticker, result):
    st.title(f"{ticker} Trade Statistics")
    stats = result["stats"]

    if stats["total_trades"] == 0:
        st.warning("No golden cross trades in this period. Try a longer date range or different MA settings.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total trades", stats["total_trades"])
    col2.metric("Winning trades", stats["winning_trades"])
    col3.metric("Losing trades", stats["losing_trades"])
    col4.metric("Win rate", f"{stats['win_rate']:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average trade return", f"{stats['avg_trade_return']:.2f}%")
    col2.metric("Best trade", f"{stats['best_trade']:.2f}%")
    col3.metric("Worst trade", f"{stats['worst_trade']:.2f}%")
    col4.metric("Total strategy return", f"{stats['strategy_return']:.2f}%")

    col1, col2 = st.columns(2)
    col1.metric("Initial capital", f"${stats['initial_capital']:,.2f}")
    col2.metric("Final portfolio value", f"${stats['final_strategy_value']:,.2f}",
                f"{stats['strategy_return']:.2f}%")


def render_trades(ticker, result):
    st.title(f"{ticker} Detailed Trades")
    trades = result["trades"]
    if trades.empty:
        st.warning("No trades were generated in this period.")
        return
    st.dataframe(trades, width="stretch", hide_index=True)
    st.download_button("Download trades (CSV)", trades.to_csv(index=False), f"{ticker}_trades.csv", "text/csv")


def render_comparison(ticker, result):
    st.title(f"{ticker}: Golden Cross vs Buy-and-Hold")
    df = result["data"]
    stats = result["stats"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Golden cross return", f"{stats['strategy_return']:.2f}%",
                f"${stats['final_strategy_value']:,.0f} final value", delta_color="off")
    col2.metric("Buy-and-hold return", f"{stats['buy_hold_return']:.2f}%",
                f"${stats['final_buy_hold_value']:,.0f} final value", delta_color="off")
    gap = stats["strategy_return"] - stats["buy_hold_return"]
    col3.metric("Difference", f"{gap:+.2f} pp", "Golden cross wins" if gap > 0 else "Buy-and-hold wins", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Strategy_Equity"], name="Golden cross", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BuyHold_Equity"], name="Buy-and-hold", line=dict(color="#1f77b4")))
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Portfolio value (USD)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.info("Buy-and-hold buys on the first day and holds to the end. "
            "Both start with the same initial capital, so the lines are directly comparable.")


def render_portfolio():
    st.title("💼 Multi-Stock Portfolio")
    st.caption(f"Track up to {MAX_PORTFOLIO_STOCKS} stocks (Yahoo Finance limit).")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}
    portfolio: dict = st.session_state.portfolio

    with st.form("add_stock"):
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.text_input("Ticker (e.g. MSFT)", key="new_ticker")
        col2.number_input("Shares", min_value=0.01, value=10.0, step=1.0, key="new_shares")
        col3.write("")
        col3.form_submit_button("Add", width="stretch", on_click=handle_add_stock)

    message = st.session_state.pop("portfolio_message", None)
    if message:
        level, text = message
        st.error(text) if level == "error" else st.success(text)

    if not portfolio:
        st.info("Your portfolio is empty. Add a ticker above to get started.")
        return

    rows = []
    for symbol, qty in portfolio.items():
        price = cached_price(symbol)
        rows.append({"Ticker": symbol, "Shares": qty, "Latest Price (USD)": price,
                     "Market Value (USD)": price * qty if price else None})
    table = pd.DataFrame(rows)
    total_value = table["Market Value (USD)"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Total portfolio value", f"${total_value:,.2f}")
    col2.metric("Number of stocks", f"{len(portfolio)} / {MAX_PORTFOLIO_STOCKS}")

    table["Weight (%)"] = (table["Market Value (USD)"] / total_value * 100).round(2)
    st.dataframe(table.style.format({"Shares": "{:,.2f}", "Latest Price (USD)": "{:,.2f}",
                                     "Market Value (USD)": "{:,.2f}", "Weight (%)": "{:.2f}"}),
                 width="stretch", hide_index=True)

    fig = go.Figure(go.Pie(labels=table["Ticker"], values=table["Market Value (USD)"], hole=0.4))
    fig.update_layout(height=400, title="Portfolio allocation")
    st.plotly_chart(fig, width="stretch")

    to_remove = st.multiselect("Remove stocks", list(portfolio.keys()))
    if st.button("Remove selected", disabled=not to_remove):
        for symbol in to_remove:
            portfolio.pop(symbol, None)
        st.rerun()


def handle_add_stock():
    portfolio = st.session_state.setdefault("portfolio", {})
    ticker = st.session_state.new_ticker.strip().upper()
    shares = st.session_state.new_shares
    st.session_state.new_ticker = ""
    st.session_state.portfolio_message = add_to_portfolio(portfolio, ticker, shares)


def add_to_portfolio(portfolio: dict, ticker: str, shares: float) -> tuple[str, str]:
    if not ticker:
        return "error", "Please enter a ticker."
    if ticker not in portfolio and len(portfolio) >= MAX_PORTFOLIO_STOCKS:
        return "error", f"Portfolio limit reached ({MAX_PORTFOLIO_STOCKS} stocks). Remove a stock before adding another."
    price = cached_price(ticker)
    if price is None:
        return "error", f"'{ticker}' is not a valid ticker or has no price data."
    portfolio[ticker] = portfolio.get(ticker, 0) + shares
    return "success", f"Added {shares:g} shares of {ticker} at ${price:,.2f}."


def render_news(ticker):
    st.title(f"📰 Latest News: {ticker or '-'}")
    st.caption("Change the ticker in the sidebar to see news for another stock.")

    if not ticker:
        st.warning("Please enter a stock ticker in the sidebar.")
        return

    with st.spinner(f"Fetching news for {ticker}..."):
        news = cached_news(ticker)

    if not news:
        st.info(f"No recent news found for {ticker}.")
        return

    for item in news:
        with st.container(border=True):
            title = f"[{item['title']}]({item['link']})" if item["link"] else item["title"]
            st.markdown(f"**{title}**")
            st.caption(" · ".join(part for part in [item["publisher"], item["published"]] if part))
            if item["summary"]:
                st.write(item["summary"])


PAGES = [
    "📊 Price Chart",
    "📋 Trade Statistics",
    "🧾 Detailed Trades",
    "💼 Portfolio",
    "📰 Stock News",
    "⚖️ Strategy Comparison",
]

if __name__ == "__main__":
    main()
