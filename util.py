import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from alpha_vantage.timeseries import TimeSeries

import pdb

# get_google_data source => https://mktstk.com/2014/12/31/how-to-get-free-intraday-stock-data-with-python/
# TODO: Add function to get online daily data
# TODO; Modify online data to save to csv, then when loading data check if csv exists first

alphavantage_api = read_key('alphavantage.key')
ts = TimeSeries(key=alphavantage_api, output_format='pandas')

def read_key(filename):
  with open(filename) as f:
    fkey = f.read()
  return fkey

def symbol_to_path(symbol, base_dir="Data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    # Read Stock Data (adjusted close) for given symbols
    df = pd.DataFrame(index = dates)
    if 'SPY' not in symbols and addSPY:
        symbols.insert(0, 'SPY')
        
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",
                        parse_dates=True, usecols=['Date',colname],
                        na_values=['nan'])
        df_temp = df_temp.rename(columns={colname:symbol})
        df = df.join(df_temp)
        if symbol == 'SPY': # drop dates if SPY didn't trade
            df = df.dropna(subset=['SPY'])
            
        #df.dropna(inplace=True)
    
    return df


def get_all_data(symbol, dates):
    # Read Stock Data (adjusted close) for given symbols
    #df = pd.DataFrame(index = dates)
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True, usecols=columns ,na_values=['nan'])
    #df = pd.read_csv(symbol_to_path(symbol), na_values=['nan'])
    #df_temp = df_temp.rename(columns={'Adj Close':symbol})
    #pdb.set_trace()
    #df  = df.join(df_temp)
    #if symbol != 'SPY' and addSPY: # drop dates if SPY didn't trade
        #spy_dates = get_data(['SPY'], dates).index
        #drop_index = df.index.difference(spy_dates)
        #df.drop(drop_index, inplace=True)
    #else:
    df.dropna(inplace=True)
        
    #pdb.set_trace()
    
    return df[dates[0]:dates[1]]
    #return df

def get_online_intraday_data(symbol, interval="1min", output_size="full"):
    # Intervals Supported 1min, 5min, 15min, 30min, 60min
    # OutputSize = compact (latest 100 data points) / full (All data points)
    #supported_intervals = ['1min', '5min', '30min', '60min']
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=output_size)
    return data, meta_data

def normalize_data(df):
    # Normalize stock prices using the first row of the df
    return df / df.ix[0,:]

def plot_data(df, title = "Stock prices"):
    # Plot Stock prices (Adj Close)
    ax = df.plot(title=title, fontsize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()
    
def sharpe_ratio(k, mean, daily_rf, std):
    return k*((mean-daily_rf)/std)

def compute_daily_returns(df):
    return df.pct_change(1)[1:]

def get_portfolio_stats(port_val, daily_rf, samples_per_year=252):
    """Helper function to compute portfolio statistics
    Parameters:
    port_val: A dataframe object showing the portfolio value for each day
    daily_rf: Daily risk-free rate, assuming it does not change
    samples_per_year: Sampling frequency per year
    
    Returns:
    cr: Cumulative return
    adr: Average daily return
    sddr: Standard deviation of daily return
    sr: Sharpe ratio
    """
    cr = port_val.iloc[-1, 0]/port_val.iloc[0, 0] - 1

    daily_returns = compute_daily_returns(port_val)
    adr = daily_returns.iloc[:, 0].mean()
    sddr = daily_returns.iloc[:, 0].std()
    sr = sharpe_ratio(np.sqrt(samples_per_year), adr, daily_rf, sddr)

    return cr, adr, sddr, sr

if __name__ == "__main__":
    symbol="GOOG"
    
    intraday_df, intraday_df_meta = get_online_intraday_data(symbol, interval='1min')
    
    pdb.set_trace()