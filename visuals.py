import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_weights(weights, tickers):
    """
    Pie chart of optimized portfolio weights.
    """
    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title("Optimized Portfolio Allocation")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_return_vs_risk(returns_df):
    """
    Bar chart of expected return and risk per asset.
    """
    mean_returns = returns_df.pct_change().mean() * 252
    risks = returns_df.pct_change().std() * np.sqrt(252)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=returns_df.columns, y=mean_returns.values, color='green', label='Return')
    sns.barplot(x=returns_df.columns, y=risks.values, color='red', alpha=0.5, label='Risk')
    plt.title("Annualized Return vs. Risk")
    plt.ylabel("Percentage")
    plt.legend()
    plt.tight_layout()
    plt.show()