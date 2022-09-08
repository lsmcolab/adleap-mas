import numpy as np
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import *
import statsmodels.api as sm
import yfinance as yf

def get_prices(tickers,start,end):
    prices = yf.download(tickers,start=start, end=end)['Adj Close'].ffill()
    return prices

def get_model(series, lag_order = 10, coint_rank = 1): 
    """
    Parameters
    ----------
    series : Expecting the dataframe consisting the prices series of trading pair (2 price series)..
    lag_order : lag_order for model estimation (use select_order to determine). The default is 10.
    coint_rank : Integration rank of series. The default is 1.

    Returns
    -------
    vecm_res : model.
    beta : Cointegration term.
    """
    model = VECM(series, deterministic="ci",
             k_ar_diff=lag_order, 
             coint_rank=coint_rank) 
    vecm_res = model.fit()
    beta = float(vecm_res.beta[coint_rank])
    
    return vecm_res , beta

def get_spread(series, beta, window = 60):
    """
    Parameters
    ----------
    series : Expecting the dataframe consisting the prices series of trading pair (2 price series).
    beta : cointegration term beta.
    window: size of window for spread function normalisation 

    Returns Normalised Z-score spread
    -------
    None.
    """
    stock1 = series[series.columns[0]]
    stock2 = series[series.columns[1]]
    
    spread = stock2 - (beta * stock1)
    spread_mavg5 = spread.rolling(window=5, center=False).mean()
    spread_mavg60 = spread.rolling(window=window, center=False).mean()
    std_60 = spread.rolling(window=window, center=False).std()
    zscore_60_5 = (spread_mavg5 - spread_mavg60)/std_60
    
    return zscore_60_5

def cointegrated_pairs(data):
    n =  (len(data.columns) -1)
    stocks = data.iloc[:,1:].columns.values #pulls all stock names 
    data = data.to_numpy()
    pvalue_matrix = np.zeros((n,n),dtype = float)
    pairs = np.empty((n*n,2),dtype=object)
    m = 0
    #loop through all possilbe combintions of stocks ie 3!=6
    for i in range(0,n):
        for j in range(0,n):
            S1 = data[:,i+1]
            S2 = data[:,j+1]
            result = ts.coint(S1,S2)
            pvalue_matrix[i,j] = result[1]
            #storing all paris in matrix form
            if (result[1] < 0.05) & (i!=j):
                pairs[m,0] = stocks[i]
                pairs[m,1] = stocks[j] 
                m = m +1
    return pvalue_matrix,  pairs

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

def get_train_test(tickers,start,end):
    # Downloading the data
    prices = yf.download(tickers,start=start, end=end)['Adj Close'].ffill()

    # Analysing the data
    scores, pvalues, pairs = find_cointegrated_pairs(prices)
    S1, S2 = prices[pairs[0][0]], prices[pairs[0][1]]
    #score, pvalue, _ = coint(S1, S2)
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = S1['AXP']
    b = results.params['AXP']
    spread = S2 - b * S1
    #ratios = S1 / S2

    # Defining the train and test data
    train, test = spread[:1761], spread[1761:]
    
    return train, test, S1, S2

def cross_check( series, level = 2):
    # Calculate where a time series crosses a specific value/ level
    change = []
    for i, el in enumerate(series):
        if i != 0 and el > level and series[i-1] < level:
        # If the current value (el) is greater the level and the previous value is less, then the boundary was crossed moving up
            change.append(1)
        elif i != 0 and el < level and series[i-1] > level:
        # If the current value (el) is less the level and the previous value is greater, then the boundary was crossed moving down    
            change.append(-1)
        else:
            # if those conditions are not meant than the boundary has not been crossed
            change.append(0)
    return change
