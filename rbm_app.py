# rbm_app.py
'''A small application to backtest simple risk based portfolios.'''

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf


st.title('Risk Based Portfolios')
st.sidebar.write("This is a small application that allows you to backtest some simple risk based portfolios.")


# input list of tickers
ticker_input = st.sidebar.text_input('Enter list of tickers separated by a single space: ', "MSFT AMZN KO MA COST LUV XOM PFE JPM UNH ACN DIS GILD F TSLA")

tickers = ticker_input.split()

# offer option to see the list of tickers
expand_tickers = st.beta_expander("Tickers")
expand_tickers.write(f"{tickers}")


# fetch data from yfinance    
@st.cache
def get_data(tickers):
    '''
    A function to get historical data from Yahoo Finance.
    Input:
        tickers: list of tickers
            list
    Output:
        pd.DataFrame with historical data.
    '''
    ohlc = yf.download(tickers, start="2010-07-01")
    ohlc.dropna(inplace=True)
    return ohlc["Adj Close"]

prices = get_data(tickers)

#if st.sidebar.checkbox('See table of prices'):
#    st.write(prices)

# offer option to see the plot of the prices
expand_plot = st.beta_expander("Plot of prices")
expand_plot.line_chart(prices)

#if st.sidebar.checkbox('See plot of prices'):
#    st.line_chart(prices)



# read initial date
start_input = st.sidebar.text_input('Enter start date for backtesting (with format %Y-%m-%d):', '2012-01-01')
#st.write(f"start date:{start_input}")

# input initial investment
initial_text = st.sidebar.text_input('Enter the initial amount in dollars:', '10000')
initial = float(initial_text)
#st.write(f"initial investment:{initial}")


# select the frequency of rebalancing
rebalancing_period = st.sidebar.selectbox('Enter the frequency of rebalancing:', ['Monthly', 'Bi-Monthly', 'Quarterly', 'Semi-Annually', 'Yearly'])

if rebalancing_period == 'Monthly':
    period = 21
elif rebalancing_period == 'Bi-Monthly':
    period = 42
elif rebalancing_period == 'Quarterly':
    period = 63
elif rebalancing_period == 'Semi-Annually':
    period = 126
elif rebalancing_period == 'Yearly':
    period = 252
    
#st.write(f"period:{period}") 


# select risk based portfolio
portfolio = st.sidebar.selectbox('Choose the risk portfolio:', ['Equally Weighted', 'Equally Weighted CPPI', 'Global Minimum Variance', 'Global Minimum Variance CPPI'])

#st.write(f"portfolio:{portfolio}")

# input parameters for CPPI
if (portfolio == 'Equally Weighted CPPI') or (portfolio == 'Global Minimum Variance CPPI'):
    floor_text = st.sidebar.text_input('Enter the floor for the CPPI strategy:', '0.8')
    floor = float(floor_text)
    m_text = st.sidebar.text_input('Enter the cushion multiplier for the CPPI strategy', '3')
    m = int(m_text)
    
    #st.write(f"floor:{floor}, m:{m}")

# input risk free rate
rate_text = st.sidebar.text_input('Enter the risk free rate:', '0.01')
risk_free_rate = float(rate_text)
#st.write(f"risk free rate:{risk_free_rate}")
    
    
# import covariance matrix shrinkage and optimization
if (portfolio == 'Global Minimum Variance') or (portfolio=='Global Minimum Variance CPPI'):
    from sklearn.covariance import LedoitWolf
    from scipy.optimize import minimize

    
# equal weight portfolio 
@ st.cache
def equal_weight(prices=prices, initial=initial, rebalance = period, start = start_input, risk_free_rate=risk_free_rate):
    '''
    A function that generates the equal weighted portfolio.
    Input:
        prices:    closing prices of assets
                   pd.DataFrame
        initial:   initial investment
                   float
        rebalance: balancing period
                   int
        start:     start date
                   date
    Output:
        total:     total value of portfolio
                   pd.DataFrame
        shares:    share distribution of assets
                   pd.DataFrame
        left:      amount of cash left after rebalancing which get compounded using risk free rate
                   pd.DataFrame  
    '''
    prices = prices[prices.index>=start]
    N = len(prices.columns)
    M = len(prices.index)
    shares = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)
    shares.iloc[0] = np.floor(initial/(N*prices.iloc[0]))
    left = pd.Series(0.0, index=prices.index)
    total = pd.Series(0.0, index=prices.index)
    left.iloc[0] = initial - (shares.iloc[0]*prices.iloc[0]).sum()
    total.iloc[0] = initial
    for i in range(0, M-1):
        if i%rebalance == 0:
            total_balance = left.iloc[i] + (shares.iloc[i]*prices.iloc[i+1]).sum()
            shares.iloc[i+1] = np.floor(total_balance/(N*prices.iloc[i+1]))
            left.iloc[i+1] = total_balance - (shares.iloc[i+1]*prices.iloc[i+1]).sum()
            total.iloc[i+1] = total_balance
        else:
            shares.iloc[i+1] = shares.iloc[i]
            left.iloc[i+1]=((1+risk_free_rate)**(1/252))*left.iloc[i]
            total.iloc[i+1] = left.iloc[i] + (shares.iloc[i]*prices.iloc[i+1]).sum()

    return total, shares, left


# minimum variance portfolio
@st.cache
def gmv_weights(cov):
    '''
    A function that computes the weights for the minimum variance portfolio.
    Input:
        cov:     covariance matrix (typically shrinked)
                 np.array   
    Output:  
        weights: weights minumizing the portfolio variance 
                 np.array
    '''
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def port_var(weights, cov):
        return weights.T @ cov @ weights

    weights = minimize(port_var, init_guess,
                       args=(cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

@st.cache
def gmv(prices=prices, initial=initial, rebalance = period, start = start_input, risk_free_rate=risk_free_rate):
    '''
    A function that generates the minimum variance portfolio.
    Input:
        prices:    closing prices of assets
                   pd.DataFrame
        initial:   initial investment
                   float
        rebalance: balancing period
                   int
        start:     start date
                   date
    Output:
        total:     total value of portfolio
                   pd.DataFrame
        shares:    share distribution of assets
                   pd.DataFrame
        left:      amount of cash left after rebalancing which get compounded using risk free rate
                   pd.DataFrame  
    '''
    
    given_prices = prices
    prices = prices[prices.index>=start]
    N = len(prices.columns)
    M = len(prices.index)
    shares = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)
    shares.iloc[0] = np.floor(initial/(N*prices.iloc[0]))
    left = pd.Series(0.0, index=prices.index)
    total = pd.Series(0.0, index=prices.index)
    left.iloc[0] = initial - (shares.iloc[0]*prices.iloc[0]).sum()
    total.iloc[0] = initial
    for i in range(0, M-1):
        if i%rebalance == 0:
            past_prices = given_prices[given_prices.index<prices.index[i]]
            exp_cov = past_prices.ewm(252).cov()
            exp_cov_lw = LedoitWolf().fit(exp_cov.tail(exp_cov.shape[1]).values).covariance_
            weights_values = list(gmv_weights(exp_cov_lw))
            tickers = prices.columns
            weights = {tickers[i]:weights_values[i] for i in range(len(tickers))}
            total_balance = left.iloc[i] + (shares.iloc[i]*prices.iloc[i+1]).sum()
            for key in list(weights.keys()):
                shares[key].iloc[i+1] = np.floor((weights[key]*total_balance)/prices[key].iloc[i+1])
            left.iloc[i+1] = total_balance - (shares.iloc[i+1]*prices.iloc[i+1]).sum()
            total.iloc[i+1] = total_balance
        else:
            shares.iloc[i+1] = shares.iloc[i]
            left.iloc[i+1]= ((1+risk_free_rate)**(1/252))*left.iloc[i]
            total.iloc[i+1] = left.iloc[i] + (shares.iloc[i]*prices.iloc[i+1]).sum()
    return total, shares, left


# CPPI backtesting
@st.cache
def cppi(risky_r, safe_r=None, m=3, start=initial, floor=0.8, riskfree_rate=risk_free_rate, drawdown=None):
    """
    A function that runs a backtest of the CPPI strategy.
    Input:
        risky_r: returns
                 pd.DataFrame
        safe_r:  safe assets
                 default None
        m:       cushion multiplier
                 int
        start:   initial investment
                 float
        floor:   floor for CPPI
                 float
        risk_free_rate: given risk free rate
                 float
        drawdown:option to modify CPPI 
                 default None
    Output:
         A dictionary containing: Asset Value History, Risk Budget History, Risky Weight History.
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result

# returns given the total portfolio value
@st.cache
def portfolio_returns(prices):
    '''
    A function that computes the returns given the price series.
    Input: 
        prices: historical prices
                pd.DataFrame
    Output:
        rets:   returns
                pd.DataFrame
    '''
    rets = prices.pct_change()
    rets.dropna(inplace=True)
    return rets

# annualized returns given returns
@st.cache
def annualize_rets(r, periods_per_year=252):
    '''
    A function that computes the annualized returns.
    Input: 
        r:  returns time series
            pd.DataFrame
        periods_per_year: periods per year
            int
    Output:
            annualized returns
            float
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return float(compounded_growth**(periods_per_year/n_periods)-1)

# annualize volatlity given returns
@st.cache
def annualize_vol(r, periods_per_year=252):
    '''
    A function that computes the annualized volatility.
    Input: 
        r:  returns time series
            pd.DataFrame
        periods_per_year: periods per year
            int
    Output:
            annualized volatility
            float
    '''
    return float(r.std()*(periods_per_year**0.5))

# Sharpe ratio given returns and risk free rate
@st.cache
def sharpe_ratio(r, riskfree_rate=risk_free_rate, periods_per_year=252):
    '''
    A function that computes the Sharpe ratio.
    Input: 
        r:  returns time series
            pd.DataFrame
        riskfree_rate : risk free rate
            float
        periods_per_year: periods per year
            int
    Output:
            Sharpe ratio
            float
    '''
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return float(ann_ex_ret/ann_vol)

# portfolio analysis
if portfolio == 'Equally Weighted':
    portfolio_total, portfolio_shares, left = equal_weight()
elif portfolio == 'Equally Weighted CPPI':
    total, portfolio_shares, left = equal_weight()
    returns = portfolio_returns(total)
    portfolio_total = cppi(risky_r=returns, safe_r=None, m=m, start=initial, floor=floor, riskfree_rate=risk_free_rate, drawdown=None)['Wealth']
    portfolio_total = portfolio_total.iloc[:, 0]

elif portfolio == 'Global Minimum Variance':
    portfolio_total, portfolio_shares, left = gmv()
elif portfolio == 'Global Minimum Variance CPPI':
    total, portfolio_shares, left = gmv()
    returns = portfolio_returns(total)
    portfolio_total = cppi(risky_r=returns, safe_r=None, m=m, start=initial, floor=floor, riskfree_rate=risk_free_rate, drawdown=None)['Wealth']
    portfolio_total = portfolio_total.iloc[:, 0]
    
st.subheader('Portfolio Value')
st.line_chart(portfolio_total)
portfolio_rets = portfolio_returns(portfolio_total)
#st.write(portfolio_rets)
ann_rets = annualize_rets(portfolio_rets)
st.write("Annualized Returns: {:.2f}".format(ann_rets))
ann_vol = annualize_vol(portfolio_rets)
st.write("Annualized Volatility: {:.2f}".format(ann_vol))
sharpe = sharpe_ratio(portfolio_rets)
st.write("Sharpe Ratio: {:.2f}".format(sharpe))

# percentage drawdown
@st.cache
def drawdown(returns, initial=initial):
    '''
    A function that computes the percentage drawdowns.
    Input:
        returns: returns time series
            pd.DataFrame
        initial: initial investiment
            float
    Output:
        pd.DataFrame with columns for
       Wealth Index 
       Previous Peaks 
       Drawdowns
    '''
    wealth_index = initial*(1+returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth Index": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdowns": drawdowns})
 
# offer option to plot drawdown
expand_drawdown = st.beta_expander("Drawdowns")
drawdowns = drawdown(portfolio_rets)['Drawdowns']
expand_drawdown.line_chart(drawdowns)
    