"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, normalize_data, get_portfolio_stats
import matplotlib.pyplot as plt

import pdb

def compute_portvals(df_orders, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    
    trading_symbols = set(df_orders['Symbol'].values)
    
    orders = df_orders.sort_index(ascending=True)    
    
    dates = [orders.index.min(), orders.index.max()] 
    symbols = orders.Symbol.unique().tolist()
    
    df = get_data(symbols, pd.date_range(dates[0], dates[1]))
    if "SPY" not in trading_symbols:
        del df['SPY'] # remove SPY
    
    df.loc[:,'Cash'] = 1.0

    # Fill NAN values if any
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(1.0, inplace=True)

    trades_df = compute_trades(orders, df, commission, impact)
    holdings_df = compute_holdings(trades_df, start_val)
    
    values_df = df*holdings_df
    
    portvals = values_df.sum(axis=1)

    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
    
    #pdb.set_trace()

    return rv

def compute_portvals_ga_train(df_orders, df, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    
    orders = df_orders.sort_index(ascending=True)    
    
    dates = [orders.index.min(), orders.index.max()] 
    symbols = orders.Symbol.unique().tolist()
    
    dfcopy = df.copy()
    #Process portfolio orders
    dfcopy.loc[:,'Cash'] = 1.0
    # Fill NAN values if any
    dfcopy.fillna(method="ffill", inplace=True)
    dfcopy.fillna(method="bfill", inplace=True)
    dfcopy.fillna(1.0, inplace=True)

    trades_df = compute_trades(orders, dfcopy, commission, impact)
    holdings_df = compute_holdings(trades_df, start_val)
    
    values_df = dfcopy*holdings_df
    
    portvals = values_df.sum(axis=1)

    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
    
    #pdb.set_trace()

    return rv


def compute_trades(orders, df, commission, impact):
    trades_df = df.copy()
    trades_df[:] = 0
    
    for index, row in orders.iterrows():
        date_index = index
        symbol = row['Symbol']
        num_shares = row['Shares']
        order = row['Order']
        modifier = 1
        if order=="BUY":
            modifier = 1
        elif order=="SELL":
            modifier = -1

        trades_df.ix[date_index][symbol] += modifier * num_shares
        trades_df.ix[date_index]['Cash'] -= (modifier * num_shares * df.ix[date_index][symbol])-(commission+impact*df.ix[date_index][symbol])
    
    return trades_df

def compute_holdings(trades_df, start_val):
    
    holdings_df = trades_df.copy()
    holdings_df[:]=0
    #holdings_df.ix[0]['Cash'] = start_val
    symbols = list(trades_df)
    if 'Cash' in symbols:
        symbols.remove('Cash')
        
    for i,row in enumerate(trades_df.itertuples()):
        for symbol in symbols:
            if (i == 0):
                holdings_df.ix[i][symbol] = trades_df.ix[i][symbol]
                holdings_df.ix[i]['Cash'] = start_val + trades_df.ix[i]['Cash']
            else:
                holdings_df.ix[i][symbol] = holdings_df.ix[i-1][symbol] + trades_df.ix[i][symbol]
                holdings_df.ix[i]['Cash'] = holdings_df.ix[i-1]['Cash'] + trades_df.ix[i]['Cash']
        
    return holdings_df
            
def market_simulator(df_orders, df_orders_benchmark, start_val=1000000, commission=9.95, impact=0.005, daily_rf=0.0, insample=True):
    #Process portfolio orders
    portvals = compute_portvals(df_orders=df_orders, start_val=start_val, commission=commission, impact=impact)
    #pdb.set_trace()
    #Get stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals, daily_rf=daily_rf)
    
    #Process benchmark orders
    portvals_bm = compute_portvals(df_orders=df_orders_benchmark, start_val=start_val, commission=commission, impact=impact)
    #Get stats
    cum_ret_bm, avg_daily_bm, std_daily_bm, sharpe_ratio_bm = get_portfolio_stats(portvals_bm, daily_rf=daily_rf)
    
    # Compare portfolio against Benchmark
    print ("Sharpe Ratio of Portfolio: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_bm))
    print (" ")
    print ("Cumulative Return of Portfolio: {}".format(cum_ret))
    print ("Cumulative Return of Benchmark : {}".format(cum_ret_bm))
    print (" ")
    print ("Standard Deviation of Portfolio: {}".format(std_daily_ret))
    print ("Standard Deviation of Benchmark : {}".format(std_daily_bm))
    print (" ")
    print ("Average Daily Return of Portfolio: {}".format(avg_daily_ret))
    print ("Average Daily Return of Benchmark : {}".format(avg_daily_bm))
    print (" ")
    print ("Final Portfolio Value: {}".format(portvals.iloc[-1, -1]))
    print ("Final Benchmark Value: {}".format(portvals_bm.iloc[-1, -1]))
    print (" ")
    
    # Rename columns and plot data
    portvals.rename(columns={'port_val':'Portfolio'}, inplace=True)
    portvals_bm.rename(columns={'port_val':'Benchmark'}, inplace=True)
    plot_data_vert(df_orders, portvals, portvals_bm, insample)
    
def market_simulator_train_ga(df_orders, df, start_val=1000000, commission=9.95, impact=0.005, daily_rf=0.0,  samples_per_year=252):
    
    portvals = compute_portvals_ga_train(df_orders=df_orders, df = df, start_val=start_val, commission=commission, impact=impact)
    
    # Get Stats    
    cr = portvals.iloc[-1, 0]/portvals.iloc[0, 0] - 1

    daily_returns = portvals.pct_change(1)[1:]
    adr = daily_returns.iloc[:, 0].mean()
    sddr = daily_returns.iloc[:, 0].std()
    sr = np.sqrt(samples_per_year)*((adr-daily_rf)/sddr)
    
    #pdb.set_trace()
    
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals, daily_rf=daily_rf)
    
    return sr
    
    
def plot_data_vert(df_orders, portvals, portvals_bm, insample=True):
    #Plot Data using vertical lines for the orders
    #Normalize Data
    portvals = normalize_data(portvals)
    portvals_bm = normalize_data(portvals_bm)
    
    
    df = portvals_bm.join(portvals, lsuffix='Benchmark', rsuffix='Portfolio')
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.rename(columns={'0Benchmark':'Benchmark', '0Portfolio':'Portfolio'}, inplace=True)
    
    plt.plot(df.loc[:, "Portfolio"], label='Portfolio', linewidth=1.2, color = 'black')
    plt.plot(df.loc[:, "Benchmark"], label='Benchmark', linewidth=1.2, color='b')
    
    # Plot Vertical Lines
    for date in df_orders.index:
        if df_orders.loc[date, "Order"] == "BUY":
            plt.axvline(x=date, color = 'g', linestyle = '--')
        else:
            plt.axvline(x=date, color = 'r', linestyle = '--')

    plt.title("Portfolio vs. Benchmark")
    plt.xlabel("Date")
    plt.xticks(rotation=50)
    plt.ylabel("Normalized prices")
    plt.legend(loc="upper left")

    if insample==True:
        filename = "portfolio_vs_benchmark(InSample).jpg"
    else:
        filename = "portfolio_vs_benchmark(OutSample).jpg"
    plt.savefig(filename)
    plt.clf()
    plt.cla()

    

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    dates = [portvals.axes[0][0].date(), portvals.axes[0][-1].date()] 
    
    daily_rets = portvals.pct_change(1)[1:]
    cum_ret = (portvals[-1]/portvals[0])-1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = (avg_daily_ret/std_daily_ret)*(252**1/2)
    
    
    #TODO: Calculate Real SPY (Market) portfolio values to compare
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(dates[0], dates[1])
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    

if __name__ == "__main__":
    test_code()
    #test_short_orders()
