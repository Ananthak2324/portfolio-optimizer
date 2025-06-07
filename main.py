import streamlit as st
import pandas as pd
from data_handler import get_price_data
from optimizer import optimize_portfolio
from visuals import plot_weights, plot_return_vs_risk

st.title("📈 Portfolio Optimizer")

# User Inputs
tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, GOOGL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

if st.button("Optimize Portfolio"):
    with st.spinner("Fetching data and optimizing..."):
        prices = get_price_data(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if prices.empty:
            st.error("⚠️ Failed to fetch price data. Please check your tickers and dates.")
        else:
            results = optimize_portfolio(prices)
            st.success("Optimization complete!")

            st.subheader("📊 Optimized Portfolio Allocation")
            plot = plot_weights(results['weights'], tickers)
            st.pyplot(plot)

            #Print weights as a table
            weight_table = {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, results['weights'])}
            st.write("### Portfolio Weights")
            st.table(weight_table)

            st.subheader("📉 Expected Performance")
            st.write(f"**Expected Annual Return:** {results['return']:.2%}")
            st.write(f"**Annual Volatility (Risk):** {results['risk']:.2%}")
            st.write(f"**Sharpe Ratio:** {results['sharpe']:.2f}")

            st.subheader("🔍 Asset-Level Risk & Return")
            bar = plot_return_vs_risk(prices)
            st.pyplot(bar)

