import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.stats import linregress
import numpy as np
import util
import os
import trendy as trdy
import pdb # REMOVE

daily_risk_free = 0.0

def get_bollinger_bands(df, window = 20):
        """
        @summary: Calculate upper and lower bollinger bands
        @param df: dataFrame containing stock data
        @param symbol: Symbol of the stock we want to calculate
        @param window (optional): Size of the window for "rolling" calculus
        """
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()
        
        upper_band = rolling_mean + 2*rolling_std
        lower_band = rolling_mean - 2*rolling_std
        
        return lower_band, upper_band
    
def get_bollinger_band_values(df, window=20):
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    bb_values = (df-rolling_mean)/(2*rolling_std)
    return bb_values

def get_rolling_mean(df, window=20):
    """
    @Summary: Get rolling mean for symbol in df
    """
    return df.rolling(window=window).mean()

def get_exponential_moving_average(df, window=20):
    # Get ema for symbol in df
    #return pd.ewma(df, span = window, min_periods = window - 1)
    return df.ewm(span = window, min_periods = window - 1).mean()

def get_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns

def get_volatility(daily_returns):
    return np.std(daily_returns)

def get_sharpe_ratio(daily_returns, k=252, daily_rf=0.0):
    sharpe_ratio = np.mean(daily_rets-daily_rf)/np.std(daily_rets)
    sharpe_ratio_annualized = np.sqrt(k)*sharpe_ratio
    return sharpe_ratio_annualized

def get_momentum(df, window=5):
    momentum = (df / df.shift(window)) - 1
    momentum.ix[0:N, :] = 0
    return momentum

def get_macd(df, fast_window=12, slow_window=26):
    EMAfast = get_exponential_moving_average(df, fast_window)
    EMAslow = get_exponential_moving_average(df, slow_window)
    MACD = EMAfast - EMAslow
    MACDsign = get_exponential_moving_average(MACD, 9)
    MACDdiff = MACD-MACDsign
    return MACD, MACDsign, MACDdiff

def get_price_sma(df, window=20):
    sma = get_rolling_mean(df, window)
    price_sma = df / sma
    return price_sma-1

def get_ohlc(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    num_dates = np.asarray([mdates.date2num(x) for x in df.index])
    values = df.values
    new_values = np.insert(values, 0, num_dates, axis=1)
    return new_values

def get_rsi(df, window=14):
    """
    compute the n period relative strength indicator
    https://matplotlib.org/1.5.3/examples/pylab_examples/finance_work2.html
    https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    """
    # Get the difference in price from the previous step
    delta = df.diff()
    delta = delta[1:]
    
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0]=0
    
    # Calculate the EWMA
    roll_up = get_exponential_moving_average(up, window=window)
    roll_down = get_exponential_moving_average(down.abs(), window=window)
    
    # Calculate RSI based on EWMA
    rsi_aux = roll_up/roll_down
    rsi = 100.0 - (100.0 / (1.0 + rsi_aux))

    return rsi

def get_trendlines(df, window=1/3.0, segments=4, charts=False):
    #gtrends, maxslope, minslope = trdy.gentrends(df, window=window, charts=charts)
    # Modifying the number of segments in trdy.segtrends, modifies the number of S/R lines returned
    strends = trdy.segtrends(df, segments=segments, charts = charts)
    #trdy.minitrends(df.values)
    #strends.append(gtrends)
    points = trdy.iterlines(df, window=20, charts = charts)
    return strends, points

def get_line_slope(line):
    x = [float(i) for i in range(len(line))]
    y = [float(i) for i in line.values]
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

def get_closest_trendline(point, trends, max_sep=0):
    #TODO: REMAKE FUNCTION SINCE IT DOESN'T WORK CORRECTLY
    # Finds the closest trendline (Support or Resistance) to the given point 
    # If the distance between them is <= max_separation*total_distance, else None
    total_min, total_max = 1000, 1000
    max_trendline, min_trendline = None, None
    date, price = point[0], point[1]
    for trendline in trends:
        max_price = trendline['Max Line'].loc[date]
        min_price = trendline['Min Line'].loc[date]
        if max_sep>0:
            msep = (max_price-min_price)*max_separation
            if abs(max_price-price)<=msep:
                slope = get_line_slope(trendline['Max Line'])
                return "Resistance", slope
            elif abs(price-min_price)<=msep:
                slope = get_line_slope(trendline['Min Line'])
                return "Support", slope
        else:
            max_dist = abs(max_price-price)
            min_dist = abs(min_price-price)
            if max_dist < total_max:
                max_trendline = trendline
                total_max = max_dist
            elif min_dist < total_min:
                min_trendline = trendline
                total_min = min_dist
    #pdb.set_trace()
    if max_sep>0:
        return None, 0
    else:
        if total_max < total_min:
            slope = get_line_slope(max_trendline['Max Line'])
            return "Resistance", slope
        elif total_min > total_max:
            slope = get_line_slope(min_trendline['Min Line'])
            return "Support", slope
        else:
            return None, 0
    
    
def plot_trendlines(title, xlabel, ylabel, price, trendlines, legend_loc="lower left"):
    lin1 = plt.plot(price, label="Price", color="blue")
    #lin2 = plt.plot(gen_trends['Max Line'], color="g")
    #lin3 = plt.plot(gen_trends['Min Line'], color="r")
    for seg_trend in trendlines:
        lin2=plt.plot(seg_trend['Max Line'], color="g")
        lin3=plt.plot(seg_trend['Min Line'], color="r")
    plt.xlabel(xlabel)
    plt.xticks(rotation=50)
    plt.ylabel(ylabel)
    plt.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Price'),
                   Line2D([0], [0], color='g', lw=1, label='Resistance'),
                   Line2D([0], [0], color='r', lw=1, label='Support')]

    plt.title(title)
    plt.legend(handles=legend_elements, loc=legend_loc, prop={'size':10})
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    
    

def plot_data(title,xlabel, ylabel, legend_loc="upper left", **kwargs):
    
    for arg in kwargs['kwargs']:
        #print arg,":", kwargs['kwargs'][arg]
        plt.plot(kwargs['kwargs'][arg], label=arg, linewidth=2.0)
        
    plt.xlabel(xlabel)
    plt.xticks(rotation=50)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc, prop={'size':10})
    plt.title(title)
    plt.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()

def plot_macd(title, xlabel, ylabel, MACD, MACDsign, MACDdiff,  legend_loc="upper right"):
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=50)
    plt.ylabel(ylabel)
    plt.plot(MACD, label="MACD", linewidth=2.0, color="blue")
    plt.plot(MACDsign, label="MACD Signal Line", linewidth=2.0, color="red")
    plt.axhline(y=0, linestyle='--', linewidth=0.6, color='black')
    plt.bar(MACDdiff[31:].keys(), MACDdiff[31:], alpha=0.3, color="c")
        
    plt.legend(loc=legend_loc, prop={'size':10})
    plt.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    
def plot_bolinger_bands(df, symbol, window = 20):
    
    mean = np.mean(df[symbol])
    
    rolling_mean = get_rolling_mean(df[symbol], window)
    rolling_std = pd.rolling_std(df[symbol], window=window)
    
    lower_band, upper_band = get_bollinger_bands(df[symbol], window)
    
    plt.plot(df[symbol], label=str(symbol), linewidth=1.2)
   
    plt.plot(rolling_mean,label="Rolling mean", color="red")
    plt.axhline(y=mean, linestyle='--', linewidth=0.5)

    plt.plot(upper_band, label = "upper Band", color="green", linewidth=0.8)
    plt.plot(lower_band, label = "lower Band", color="green", linewidth=0.8)
    
    plt.xlabel("Date")
    plt.xticks(rotation=50)
    plt.ylabel("Price")
    plt.legend(loc='lower left', prop={'size':10})
    title = "{} Bollinger bands".format(str(symbol))
    plt.title(title)
    plt.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    
def plot_rsi(title,xlabel, ylabel, rsi, legend_loc="lower right"):
    
    plt.plot(rsi, label="RSI (14)", color="y", linewidth=1.2)
    #plt.fill_between(r.date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor="red")
    #plt.fill_between(r.date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor="red")
    plt.axhline(y=70, linestyle='--', linewidth=0.6, color='black', label=">70 = overbought")
    plt.axhline(y=30, linestyle='--', linewidth=0.6, color='black', label="<30 = oversold")
    plt.xlabel(xlabel)
    plt.xticks(rotation=50)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc, prop={'size':10})
    plt.title(title)
    plt.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    
def plot_ohlc(title, xlabel, ylabel, ohlc, legend_loc='upper right', sma=False, macd=False):
    
   
    
    if type(macd) != bool:
        f, (ax1, ax2) = plt.subplots(2, sharex=True)
        #ax1.subplot2grid((8, 2), (1, 0), rowspan=6, colspan=4)
    else:
        ax1 = plt.subplot2grid((8, 2), (1, 0), rowspan=6, colspan=4)
    candlestick_ohlc(ax1, ohlc, width=0.4 , colorup='#77d879', colordown='#db3f3f')
    
    if type(sma) != bool:
        ax1.plot(sma,label="Rolling mean", color="blue")
    
    ax1.xaxis_date()
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M:%S'))
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
    
    if type(macd) != bool:
        # Plot MACD and MACDsign in different plot
        ax2.plot(macd[0], label="MACD", linewidth=2.0, color="blue")
        ax2.plot(macd[1], label="MACD Signal Line", linewidth=2.0, color="red")
        ax2.axhline(y=0, linestyle='-', linewidth=0.8, color='black')
        ax2.grid(True, color='black', linestyle='--', linewidth=0.2, alpha=0.8)
        
    
    plt.xlabel(xlabel)
    plt.xticks(rotation=50)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.legend(loc=legend_loc, prop={'size':10})
    
    filename = str(title).replace(" ", "_")
    filename+=".jpg"
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    
        
if __name__ == "__main__":
    symbols = ['SPY','AAPL', 'GOOG', 'IBM', 'XOM']
    dates = ['2012-01-01','2012-12-28']
    df = util.get_data(symbols, pd.date_range(dates[0], dates[1]))
    
    symbol = 'GOOG'
    
    #Plot Price/SMA
    price = df[symbol]
    sma = get_rolling_mean(price)
    price_sma = get_price_sma(price)
    
    price = util.normalize_data(price)
    sma = get_rolling_mean(price)
    
    plot_data(title=str(symbol)+" Price SMA", xlabel="Date", ylabel="Price (Normalized)", kwargs={'Price':price,'SMA':sma})
    
    #Plot Bollinger Bands
    plot_bolinger_bands(df, symbol)
    
    #Plot RSI
    prices = df[symbol]
    rsi = get_rsi(prices)
    
    plot_rsi(title=str(symbol)+" RSI", xlabel="Date", ylabel="RSI", rsi=rsi)
    
    #Plot MACD
    price = df[symbol]
    MACD, MACDsign, MACDdiff = get_macd(price)
    
    plot_macd("{} MACD".format(symbol), xlabel="Date", ylabel="Divergence", MACD=MACD, MACDsign=MACDsign, MACDdiff=MACDdiff)
    
    # Plot OHLC Candlesticks graph
    all_dates = ['2011-01-01','2012-12-28']
    obs_dates = ['2011-03-01','2011-03-28']
    df_aapl = util.get_all_data(symbol, all_dates)
    df = util.get_data([symbol], pd.date_range(all_dates[0], all_dates[1]))
    sma = get_rolling_mean(df[symbol])
    ohlc = get_ohlc(df_aapl[obs_dates[0]:obs_dates[1]])
    MACD, MACDsign, MACDdiff = get_macd(df[symbol], fast_window=12, slow_window=26)
    plot_ohlc("{} OHLC Candlesticks".format(symbol), "Date", "Price", ohlc, sma=sma[obs_dates[0]:obs_dates[1]], macd=[MACD[obs_dates[0]:obs_dates[1]], MACDsign[obs_dates[0]:obs_dates[1]]])
    
    # Plot Trendlines
    strends, points = get_trendlines(price)
    pdb.set_trace()
    plot_trendlines("GOOG Trendlines", "Date", "Price", price, strends)