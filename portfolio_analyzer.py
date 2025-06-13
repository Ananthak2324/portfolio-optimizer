import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from data_handler import get_price_data

class PortfolioAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate assumption
        # Define sector ETFs for diversification
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
    def analyze_proposed_portfolio(self, portfolio_data):
        """
        Analyze a proposed portfolio with given stock prices and shares.
        
        Args:
            portfolio_data (dict): Dictionary with tickers as keys and (price, shares) as values
                e.g., {'AAPL': (150.0, 10), 'GOOGL': (2800.0, 5)}
        
        Returns:
            dict: Analysis results including metrics and market conditions
        """
        try:
            # Calculate total investment and weights
            total_investment = sum(price * shares for price, shares in portfolio_data.values())
            weights = {ticker: (price * shares) / total_investment 
                      for ticker, (price, shares) in portfolio_data.items()}
            
            # Get historical data for analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Use 1 year of historical data
            
            # Get historical price data
            tickers = list(portfolio_data.keys())
            historical_data = get_price_data(tickers, start=start_date.strftime("%Y-%m-%d"), 
                                          end=end_date.strftime("%Y-%m-%d"))
            
            if historical_data.empty:
                return {"error": "Failed to fetch historical price data. Please check ticker symbols."}
            
            # Calculate historical returns and volatility
            historical_returns = historical_data.pct_change().dropna()
            historical_volatility = historical_returns.std() * np.sqrt(252)
            
            # Calculate historical portfolio metrics
            historical_portfolio_returns = historical_returns.dot(pd.Series(weights))
            historical_annual_return = historical_portfolio_returns.mean() * 252
            historical_annual_volatility = historical_portfolio_returns.std() * np.sqrt(252)
            historical_sharpe_ratio = (historical_annual_return - self.risk_free_rate) / historical_annual_volatility
            
            # Calculate historical maximum drawdown
            historical_cumulative_returns = (1 + historical_portfolio_returns).cumprod()
            historical_rolling_max = historical_cumulative_returns.expanding().max()
            historical_drawdowns = historical_cumulative_returns / historical_rolling_max - 1
            historical_max_drawdown = historical_drawdowns.min()
            
            # Calculate beta using S&P 500 as market proxy
            market_data = get_price_data(['^GSPC'], start=start_date.strftime("%Y-%m-%d"), 
                                       end=end_date.strftime("%Y-%m-%d"))
            
            # Initialize beta
            beta = 1.0
            
            if not market_data.empty:
                market_returns = market_data.pct_change().dropna()
                common_index = historical_returns.index.intersection(market_returns.index)
                
                if len(common_index) > 0:
                    # Use .loc to pick those dates out of the Series/DataFrame
                    hist_portfolio_returns = historical_returns.dot(pd.Series(weights)).loc[common_index]
                    # Grab the first (and only) column as a Series, then .loc the dates
                    market_ret_series = market_returns.loc[common_index].iloc[:, 0]
                    beta = np.cov(hist_portfolio_returns, market_ret_series)[0,1] / np.var(market_ret_series)
            
            # Generate future projections (252 trading days)
            num_days = 252
            future_prices = {}
            dates = []
            
            # Generate dates
            current_date = end_date
            for _ in range(num_days):
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:  # Only weekdays
                    dates.append(current_date)
            
            # Generate price projections
            for ticker in tickers:
                current_price = portfolio_data[ticker][0]
                daily_vol = historical_volatility[ticker] / np.sqrt(252)
                # Generate random walk with drift based on historical return
                daily_returns = np.random.normal(historical_annual_return/252, daily_vol, len(dates))
                future_prices[ticker] = current_price * (1 + daily_returns).cumprod()
            
            # Create DataFrame for future prices
            future_prices_df = pd.DataFrame(future_prices, index=dates)
            
            # Calculate projected metrics
            future_returns = future_prices_df.pct_change().dropna()
            projected_portfolio_returns = future_returns.dot(pd.Series(weights))
            projected_annual_return = projected_portfolio_returns.mean() * 252
            projected_annual_volatility = projected_portfolio_returns.std() * np.sqrt(252)
            projected_sharpe_ratio = (projected_annual_return - self.risk_free_rate) / projected_annual_volatility
            
            # Calculate current portfolio value
            current_value = {ticker: price * shares 
                           for ticker, (price, shares) in portfolio_data.items()}
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(market_data)
            
            # Generate portfolio opinion
            opinion = self._generate_portfolio_opinion(
                historical_annual_return, historical_annual_volatility, 
                historical_sharpe_ratio, beta, historical_max_drawdown,
                market_conditions
            )
            
            # Prepare future prices for display
            future_prices_display = future_prices_df.reset_index()
            future_prices_display = future_prices_display.rename(columns={'index': 'Date'})
            
            return {
                "portfolio_metrics": {
                    "total_investment": total_investment,
                    "weights": weights,
                    "current_value": current_value,
                    "historical_metrics": {
                        "annual_return": historical_annual_return,
                        "annual_volatility": historical_annual_volatility,
                        "sharpe_ratio": historical_sharpe_ratio,
                        "max_drawdown": historical_max_drawdown
                    },
                    "projected_metrics": {
                        "annual_return": projected_annual_return,
                        "annual_volatility": projected_annual_volatility,
                        "sharpe_ratio": projected_sharpe_ratio
                    },
                    "beta": beta,
                    "future_prices": future_prices_display
                },
                "market_conditions": market_conditions,
                "opinion": opinion
            }
            
        except Exception as e:
            return {"error": f"Error analyzing portfolio: {str(e)}"}
    
    def _analyze_market_conditions(self, market_data):
        """Analyze current market conditions using S&P 500 data."""
        try:
            if market_data.empty:
                return {
                    "market_condition": "Unknown",
                    "market_volatility": 0.0,
                    "market_trend": 0.0
                }
            
            returns = market_data.pct_change().dropna()
            recent_returns = returns.tail(20)  # Last 20 trading days
            
            # Calculate market metrics
            volatility = returns.std() * np.sqrt(252)
            trend = (market_data.iloc[-1] / market_data.iloc[0] - 1) * 100
            
            # Determine market condition
            if trend > 5:
                condition = "Bullish"
            elif trend < -5:
                condition = "Bearish"
            else:
                condition = "Neutral"
            
            return {
                "market_condition": condition,
                "market_volatility": float(volatility.iloc[0]),
                "market_trend": float(trend.iloc[0])
            }
            
        except Exception as e:
            return {
                "market_condition": "Unknown",
                "market_volatility": 0.0,
                "market_trend": 0.0
            }
    
    def _generate_portfolio_opinion(self, annual_return, annual_volatility, 
                                  sharpe_ratio, beta, max_drawdown, market_conditions):
        """Generate an opinion about the portfolio based on various metrics."""
        opinion = []
        
        # Return analysis
        if annual_return > 0.15:
            opinion.append("Strong return potential with annual return above 15%")
        elif annual_return > 0.10:
            opinion.append("Good return potential with annual return above 10%")
        elif annual_return > 0.05:
            opinion.append("Moderate return potential with annual return above 5%")
        else:
            opinion.append("Low return potential with annual return below 5%")
        
        # Risk analysis
        if annual_volatility < 0.15:
            opinion.append("Low volatility portfolio, suitable for conservative investors")
        elif annual_volatility < 0.25:
            opinion.append("Moderate volatility, balanced risk-reward profile")
        else:
            opinion.append("High volatility portfolio, suitable for aggressive investors")
        
        # Sharpe ratio analysis
        if sharpe_ratio > 1.5:
            opinion.append("Excellent risk-adjusted returns with Sharpe ratio above 1.5")
        elif sharpe_ratio > 1.0:
            opinion.append("Good risk-adjusted returns with Sharpe ratio above 1.0")
        else:
            opinion.append("Below-average risk-adjusted returns")
        
        # Beta analysis
        if beta < 0.8:
            opinion.append("Low market sensitivity, good for diversification")
        elif beta > 1.2:
            opinion.append("High market sensitivity, consider adding defensive stocks")
        
        # Drawdown analysis
        if max_drawdown < -0.2:
            opinion.append("Significant drawdown risk, consider adding defensive positions")
        
        # Market condition analysis
        if market_conditions["market_condition"] == "Bearish":
            opinion.append("Current market conditions are bearish, consider defensive positioning")
        elif market_conditions["market_condition"] == "Bullish":
            opinion.append("Current market conditions are bullish, good time for growth stocks")
        
        return opinion 

    def recommend_stocks(self, portfolio_data, num_recommendations=2):
        """
        Recommend stocks to add to the current portfolio for optimization.
        
        Args:
            portfolio_data (dict): Current portfolio with tickers as keys and (price, shares) as values
            num_recommendations (int): Number of stocks to recommend
        
        Returns:
            dict: Recommended stocks with analysis
        """
        try:
            # Get current portfolio metrics
            current_analysis = self.analyze_proposed_portfolio(portfolio_data)
            if "error" in current_analysis:
                return {"error": current_analysis["error"]}
            
            current_metrics = current_analysis["portfolio_metrics"]
            current_beta = current_metrics["beta"]
            current_volatility = current_metrics["historical_metrics"]["annual_volatility"]
            
            # Get historical data for sector ETFs
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Get historical data for all sector ETFs
            etf_data = get_price_data(list(self.sector_etfs.keys()), 
                                    start=start_date.strftime("%Y-%m-%d"),
                                    end=end_date.strftime("%Y-%m-%d"))
            
            if etf_data.empty:
                return {"error": "Failed to fetch sector ETF data"}
            
            # Calculate sector correlations with current portfolio
            returns = etf_data.pct_change().dropna()
            
            # Get current portfolio returns
            current_tickers = list(portfolio_data.keys())
            current_prices = get_price_data(current_tickers,
                                          start=start_date.strftime("%Y-%m-%d"),
                                          end=end_date.strftime("%Y-%m-%d"))
            
            if current_prices.empty:
                return {"error": "Failed to fetch current portfolio price data"}
            
            current_returns = current_prices.pct_change().dropna()
            
            # Calculate portfolio returns using weights
            weights = {ticker: (price * shares) / sum(price * shares for price, shares in portfolio_data.values())
                      for ticker, (price, shares) in portfolio_data.items()}
            
            portfolio_returns = current_returns.dot(pd.Series(weights))
            
            # Calculate correlations with each sector ETF
            correlations = {}
            for etf in self.sector_etfs.keys():
                if etf in returns.columns:
                    # Align dates between portfolio returns and ETF returns
                    common_dates = portfolio_returns.index.intersection(returns.index)
                    if len(common_dates) > 0:
                        corr = returns.loc[common_dates, etf].corr(portfolio_returns.loc[common_dates])
                        correlations[etf] = corr
            
            # Find sectors with low correlation to current portfolio
            low_corr_sectors = sorted(correlations.items(), key=lambda x: abs(x[1]))[:3]
            
            # Get top stocks from low correlation sectors
            recommendations = []
            for etf, _ in low_corr_sectors:
                sector = self.sector_etfs[etf]
                # Get top holdings of the ETF
                etf_info = yf.Ticker(etf)
                try:
                    holdings = etf_info.get_holdings()
                    if holdings is not None and not holdings.empty:
                        # Get top 5 holdings
                        top_holdings = holdings.head(5)
                        for _, holding in top_holdings.iterrows():
                            ticker = holding['ticker']
                            if ticker not in portfolio_data:  # Don't recommend stocks already in portfolio
                                # Get stock data
                                stock_data = get_price_data([ticker], 
                                                          start=start_date.strftime("%Y-%m-%d"),
                                                          end=end_date.strftime("%Y-%m-%d"))
                                if not stock_data.empty:
                                    stock_returns = stock_data.pct_change().dropna()
                                    
                                    # Align dates for correlation calculation
                                    common_dates = portfolio_returns.index.intersection(stock_returns.index)
                                    if len(common_dates) > 0:
                                        stock_volatility = stock_returns.loc[common_dates].std() * np.sqrt(252)
                                        stock_beta = stock_returns.loc[common_dates].corr(portfolio_returns.loc[common_dates])
                                        
                                        # Calculate potential improvement
                                        potential_volatility = (current_volatility + stock_volatility.iloc[0]) / 2
                                        potential_beta = (current_beta + stock_beta.iloc[0]) / 2
                                        
                                        recommendations.append({
                                            'ticker': ticker,
                                            'sector': sector,
                                            'current_price': stock_data.iloc[-1].iloc[0],
                                            'volatility': float(stock_volatility.iloc[0]),
                                            'beta': float(stock_beta.iloc[0]),
                                            'potential_improvement': {
                                                'volatility_reduction': float(current_volatility - potential_volatility),
                                                'beta_reduction': float(current_beta - potential_beta)
                                            }
                                        })
                except Exception as e:
                    continue
            
            # Sort recommendations by potential improvement
            recommendations.sort(key=lambda x: (
                x['potential_improvement']['volatility_reduction'] +
                x['potential_improvement']['beta_reduction']
            ), reverse=True)
            
            # If no recommendations, suggest top 2 stocks from best sector
            fallback_suggestions = []
            fallback_message = None
            if not recommendations and low_corr_sectors:
                best_etf, _ = low_corr_sectors[0]
                sector = self.sector_etfs[best_etf]
                etf_info = yf.Ticker(best_etf)
                try:
                    holdings = etf_info.get_holdings()
                    if holdings is not None and not holdings.empty:
                        top_holdings = holdings.head(5)
                        for _, holding in top_holdings.iterrows():
                            ticker = holding['ticker']
                            if ticker not in portfolio_data:
                                fallback_suggestions.append({
                                    'ticker': ticker,
                                    'sector': sector
                                })
                                if len(fallback_suggestions) == 2:
                                    break
                    if fallback_suggestions:
                        fallback_message = f"No suitable recommendations found, but here are two top stocks from the best low-correlation sector ({sector}):"
                except Exception as e:
                    pass
            
            return {
                "recommendations": recommendations[:num_recommendations],
                "analysis": {
                    "current_volatility": current_volatility,
                    "current_beta": current_beta,
                    "low_correlation_sectors": [self.sector_etfs[etf] for etf, _ in low_corr_sectors]
                },
                "fallback_message": fallback_message,
                "fallback_suggestions": fallback_suggestions
            }
            
        except Exception as e:
            return {"error": f"Error generating recommendations: {str(e)}"} 