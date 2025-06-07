import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_weights(weights, tickers):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    ax.set_title("Optimized Portfolio Allocation")
    ax.axis('equal')
    return fig

def plot_return_vs_risk(returns_df):
    mean_returns = returns_df.pct_change().mean() * 252
    risks = returns_df.pct_change().std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=returns_df.columns, y=mean_returns.values, ax=ax, color='green', label='Return')
    sns.barplot(x=returns_df.columns, y=risks.values, ax=ax, color='red', alpha=0.5, label='Risk')
    ax.set_title("Annualized Return vs. Risk")
    ax.set_ylabel("Percentage")
    ax.legend()
    return fig
