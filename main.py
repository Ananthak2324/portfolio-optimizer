import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_handler import get_price_data
from optimizer import optimize_portfolio
from visuals import plot_weights, plot_return_vs_risk
from portfolio_analyzer import PortfolioAnalyzer
from portfolio_optimizer import PortfolioOptimizer

def main():
    st.title("📈 Portfolio Optimizer")

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Portfolio Analysis", "Portfolio Optimization", "Stock Recommendations"])

    # Initialize analyzers
    analyzer = PortfolioAnalyzer()
    optimizer = PortfolioOptimizer()

    with tab1:
        st.header("Analyze Your Portfolio")
        
        # Input for portfolio analysis
        st.subheader("Enter Your Portfolio Details")
        
        # Initialize portfolio data
        portfolio_data = {}
        
        # Create a form for portfolio input
        with st.form("portfolio_form"):
            num_stocks = st.number_input("Number of stocks in your portfolio", min_value=1, max_value=20, value=3)
            
            for i in range(num_stocks):
                col1, col2, col3 = st.columns(3)
                with col1:
                    ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
                with col2:
                    price = st.number_input(f"Current Price {i+1}", min_value=0.0, key=f"price_{i}")
                with col3:
                    shares = st.number_input(f"Number of Shares {i+1}", min_value=1, key=f"shares_{i}")
                
                if ticker and price and shares:
                    portfolio_data[ticker.upper()] = (price, shares)
            
            submitted = st.form_submit_button("Analyze Portfolio")
            
            if submitted and portfolio_data:
                try:
                    analyzer = PortfolioAnalyzer()
                    results = analyzer.analyze_proposed_portfolio(portfolio_data)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Display portfolio metrics
                        st.subheader("📊 Portfolio Analysis Results")
                        
                        # Display current portfolio value
                        st.write("### Current Portfolio Value")
                        current_value = results["portfolio_metrics"]["current_value"]
                        total_value = sum(current_value.values())
                        st.write(f"Total Portfolio Value: ${total_value:,.2f}")
                        
                        # Display individual stock values
                        for ticker, value in current_value.items():
                            st.write(f"{ticker}: ${value:,.2f}")
                        
                        # Display historical metrics
                        st.write("### Historical Performance (Past Year)")
                        hist_metrics = results["portfolio_metrics"]["historical_metrics"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Historical Annual Return", f"{hist_metrics['annual_return']:.2%}")
                            st.metric("Historical Sharpe Ratio", f"{hist_metrics['sharpe_ratio']:.2f}")
                        with col2:
                            st.metric("Historical Volatility", f"{hist_metrics['annual_volatility']:.2%}")
                            st.metric("Historical Max Drawdown", f"{hist_metrics['max_drawdown']:.2%}")
                        
                        # Display projected metrics
                        st.write("### Projected Performance (Next Year)")
                        proj_metrics = results["portfolio_metrics"]["projected_metrics"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Projected Annual Return", f"{proj_metrics['annual_return']:.2%}")
                            st.metric("Projected Sharpe Ratio", f"{proj_metrics['sharpe_ratio']:.2f}")
                        with col2:
                            st.metric("Projected Volatility", f"{proj_metrics['annual_volatility']:.2%}")
                            st.metric("Beta", f"{results['portfolio_metrics']['beta']:.2f}")
                        
                        # Display future price projections
                        st.write("### Future Price Projections (1 Year)")
                        future_prices = results["portfolio_metrics"]["future_prices"]
                        
                        # Create interactive plot using plotly
                        fig = go.Figure()
                        for ticker in future_prices.columns:
                            if ticker != 'Date':  # Skip the Date column
                                fig.add_trace(go.Scatter(
                                    x=future_prices['Date'],
                                    y=future_prices[ticker],
                                    name=ticker,
                                    mode='lines'
                                ))
                        
                        fig.update_layout(
                            title="Projected Stock Prices",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig)
                        
                        # Display opinion
                        st.write("### Portfolio Opinion")
                        for point in results["opinion"]:
                            st.write(f"- {point}")
                        
                        # Display market conditions
                        st.write("### Current Market Conditions")
                        market = results["market_conditions"]
                        st.write(f"Market Trend: {market['market_condition']}")
                        st.write(f"Market Volatility: {market['market_volatility']:.2%}")
                        st.write(f"Market Trend Strength: {market['market_trend']:.2%}")
                        
                        # Display portfolio weights
                        st.write("### Portfolio Allocation")
                        weights_df = pd.DataFrame.from_dict(results["portfolio_metrics"]["weights"], 
                                                          orient='index', columns=['Weight'])
                        st.bar_chart(weights_df)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check that all ticker symbols are valid and try again.")

    with tab2:
        st.header("Portfolio Optimization")
        
        # Create a form for optimization input
        with st.form("optimization_form"):
            st.subheader("Enter Optimization Parameters")
            
            # Create three columns for input
            col1, col2, col3 = st.columns(3)
            
            with col1:
                opt_tickers = st.text_input("Stock Tickers (comma-separated)", "AAPL,GOOGL,SCHD")
            
            with col2:
                target_return = st.number_input("Target Annual Return (%)", min_value=0.0, max_value=100.0, value=10.0)
            
            with col3:
                investment_amount = st.number_input("Investment Amount ($)", min_value=1000.0, value=10000.0)
            
            opt_submitted = st.form_submit_button("Optimize Portfolio")
            
            if opt_submitted:
                try:
                    # Parse input
                    opt_ticker_list = [t.strip() for t in opt_tickers.split(",")]
                    
                    # Get historical data
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.DateOffset(years=1)
                    
                    price_data = get_price_data(opt_ticker_list, 
                                              start=start_date.strftime("%Y-%m-%d"),
                                              end=end_date.strftime("%Y-%m-%d"))
                    
                    if price_data.empty:
                        st.error("Failed to fetch price data. Please check ticker symbols.")
                        st.stop()
                    
                    # Optimize portfolio
                    try:
                        optimal_portfolio = optimize_portfolio(price_data)
                        optimal_weights = optimal_portfolio['weights']
                        
                        # Calculate optimal allocation
                        allocation = {ticker: weight * investment_amount 
                                    for ticker, weight in zip(opt_ticker_list, optimal_weights)}
                        
                        # Display results
                        st.subheader("Optimal Portfolio Allocation")
                        
                        # Create a DataFrame for better display
                        allocation_df = pd.DataFrame({
                            'Ticker': opt_ticker_list,
                            'Weight': [f"{w*100:.2f}%" for w in optimal_weights],
                            'Amount': [f"${a:,.2f}" for a in allocation.values()]
                        })
                        
                        st.dataframe(allocation_df)
                        
                        # Display portfolio metrics
                        st.subheader("Portfolio Metrics")
                        st.metric("Expected Annual Return", f"{optimal_portfolio['return']*100:.2f}%")
                        st.metric("Expected Annual Volatility", f"{optimal_portfolio['risk']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{optimal_portfolio['sharpe']:.2f}")
                        
                        # Create pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=opt_ticker_list,
                            values=optimal_weights,
                            hole=.3
                        )])
                        
                        fig.update_layout(
                            title="Optimal Portfolio Weights",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig)
                        
                    except ValueError as e:
                        st.error(f"Failed to optimize portfolio: {str(e)}")
                        st.stop()
                    
                except Exception as e:
                    st.error(f"Error optimizing portfolio: {str(e)}")

    with tab3:
        st.header("Stock Recommendations")
        
        # Create a form for recommendation input
        with st.form("recommendation_form"):
            st.subheader("Enter Current Portfolio")
            
            # Create three columns for input
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rec_tickers = st.text_input("Current Stock Tickers (comma-separated)", "AAPL,GOOGL,SCHD")
            
            with col2:
                rec_prices = st.text_input("Current Prices (comma-separated)", "150.0,2800.0,75.0")
            
            with col3:
                rec_shares = st.text_input("Number of Shares (comma-separated)", "10,5,20")
            
            rec_submitted = st.form_submit_button("Get Recommendations")
            
            if rec_submitted:
                try:
                    # Parse input
                    rec_ticker_list = [t.strip() for t in rec_tickers.split(",")]
                    rec_price_list = [float(p.strip()) for p in rec_prices.split(",")]
                    rec_share_list = [int(s.strip()) for s in rec_shares.split(",")]
                    
                    if len(rec_ticker_list) != len(rec_price_list) or len(rec_ticker_list) != len(rec_share_list):
                        st.error("Number of tickers, prices, and shares must match!")
                        st.stop()
                    
                    # Create portfolio data
                    rec_portfolio_data = {ticker: (price, share) for ticker, price, share in zip(rec_ticker_list, rec_price_list, rec_share_list)}
                    
                    # Get recommendations
                    recommendations = analyzer.recommend_stocks(rec_portfolio_data)
                    
                    if "error" in recommendations:
                        st.error(recommendations["error"])
                        st.stop()
                    
                    # Display current portfolio analysis
                    st.subheader("Current Portfolio Analysis")
                    st.write(f"Current Volatility: {recommendations['analysis']['current_volatility']*100:.2f}%")
                    st.write(f"Current Beta: {recommendations['analysis']['current_beta']:.2f}")
                    st.write("Low Correlation Sectors:", ", ".join(recommendations['analysis']['low_correlation_sectors']))
                    
                    # Display recommendations or fallback
                    st.subheader("Recommended Stocks")
                    if recommendations['recommendations']:
                        for i, rec in enumerate(recommendations['recommendations'], 1):
                            st.write(f"Recommendation {i}:")
                            st.write(f"• Ticker: {rec['ticker']}")
                            st.write(f"• Sector: {rec['sector']}")
                            st.write(f"• Current Price: ${rec['current_price']:.2f}")
                            st.write(f"• Volatility: {rec['volatility']*100:.2f}%")
                            st.write(f"• Beta: {rec['beta']:.2f}")
                            st.write(f"• Potential Volatility Reduction: {rec['potential_improvement']['volatility_reduction']*100:.2f}%")
                            st.write(f"• Potential Beta Reduction: {rec['potential_improvement']['beta_reduction']:.2f}")
                            st.write("---")
                    elif recommendations.get('fallback_message') and recommendations.get('fallback_suggestions'):
                        st.info(recommendations['fallback_message'])
                        for i, rec in enumerate(recommendations['fallback_suggestions'], 1):
                            st.write(f"Suggestion {i}:")
                            st.write(f"• Ticker: {rec['ticker']}")
                            st.write(f"• Sector: {rec['sector']}")
                            st.write("---")
                    else:
                        st.warning("No recommendations or suggestions available at this time.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()

