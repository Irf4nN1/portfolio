
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:27:32 2023

@author: irfan
"""
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import pandas as pd
import seaborn as sns

stocks = ['VNQ', 'QQQ', 'VGT', 'VOO']


data = yf.download(stocks, start = "2000-01-01", end= "2023-01-01")["Adj Close"].dropna()

data.plot(figsize= (15,8), fontsize =12)

daily = data.pct_change().dropna()
summary = daily.describe().T.loc[:,["mean","std"]]
summary["mean"] = summary["mean"]*252
summary["std"] = summary["std"]*np.sqrt(252)
print(summary)

summary.plot.scatter(x="std", y="mean", figsize=(12,8), s = 50, fontsize=15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i,"std"]+0.002, summary.loc[i,"mean"]+0.002),size=15)
plt.xlabel("Annual Risk", fontsize=15)
plt.ylabel("Annual Return", fontsize=15)
plt.title("Risk/Return")

mean_returns = data.pct_change().mean()
cov_matrix = data.pct_change().cov()
cor_matrix = data.pct_change().corr()

plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(cor_matrix, cmap= "Reds", annot = True, annot_kws = {"size": 15}, vmax= 0.6)

num_portfolios = 25000
port_weights = []
portfolio_returns = []
portfolio_risk = []
port_ratio = []
rf = 0.0533


for i in range(num_portfolios):

    weights = np.random.random(len(stocks))

    weights = np.round((weights / np.sum(weights)),2)
    port_weights.append(weights)

    annualized_return = np.sum(mean_returns * weights) * 252
    portfolio_returns.append(annualized_return)
    
    portfolio_stdev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    portfolio_risk.append(portfolio_stdev)
    
    sharpe_ratio = (annualized_return - rf)/portfolio_stdev
    port_ratio.append(sharpe_ratio)
    
portfolio_returns = np.array(portfolio_returns)
portfolio_risk = np.array(portfolio_risk)
port_ratio = np.array(port_ratio)

results = [portfolio_returns, portfolio_risk, port_ratio, port_weights]

results_frame = pd.DataFrame(results).T
results_frame.columns = ['ret','stdev','sharpe','weights']


maxIndex = results_frame.sharpe.astype(float).argmax()
minIndex = results_frame.stdev.astype(float).argmin()

plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe)
plt.xlabel('Risk')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results_frame.stdev[maxIndex], results_frame.ret[maxIndex], c="red")
plt.scatter(results_frame.stdev[minIndex], results_frame.ret[minIndex], c="blue")

plt.show()


Opt = (f"Risk = {round(results_frame['stdev'][maxIndex],3)}", 
       f"Return = {round(results_frame['ret'][maxIndex],3)}", 
       f"Sharpe = {round(results_frame['sharpe'][maxIndex],3)}",
       f"Weights = {results_frame['weights'][maxIndex]}")



MinVar = (f"Risk = {round(results_frame['stdev'][minIndex],3)}", 
      f"Return = {round(results_frame['ret'][minIndex],3)}",
      f"Sharpe = {round(results_frame['sharpe'][minIndex],3)}",
      f"Weights = {results_frame['weights'][minIndex]}")

print(f"Optimized Portfolio = {Opt}")
print(f"Minimun Variance Portfolio = {MinVar}")




