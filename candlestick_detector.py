import pandas as pd
import matplotlib.pyplot as plt
import indicators as ind
import matplotlib.dates as mdates
import numpy as np
import util
import pdb

date, popen, high, low, pclose, volume = 0,1,2,3,4,5

def is_harami_cross(row1, row2):
    # row = [Date, Open, High, Low, Close, Volume]
    if not is_doji(row1) and is_doji(row2):
        if row2[low] >= min(row1[popen], row1[pclose]) and row2[high] <= max(row1[popen], row1[pclose]):
            #Harami Cross Found
            if row1[pclose]-row1[popen]<0:
                # Downward trend reversing into Upward trend
                signal = 1 # Bullish - BUY
            elif row1[pclose]-row1[popen]>0:
                # Upward trend reversing into Downward trend
                signal = -1 # Bearish - SHORT
            return True, signal
    return False, None

def is_abandoned_baby(row1, row2, row3):
    # row = [Date, Open, High, Low, Close, Volume]
    if not is_doji(row1) and not is_doji(row3) and is_doji(row2):
        if row2[low] > row1[high] and row2[low] > row3[high]:
            # Bearish Abandoned Baby
            # Upward trend reversing into Downward
            return True, -1 # Bearish - SHORT
        elif row2[high] < row1[low] and row2[high] < row3[low]:
            # Bullish Abandoned Baby
            # Downward trend reversing into Upward
            return True, 1 # Bullish - BUY
    return False, None
    
def is_doji(row, max_doji_width=0.001):
    return abs(row[pclose]-row[popen])<= max_doji_width  * max([row[pclose], row[popen]])

def is_morning_star_doji(row1, row2, row3):
    # row = [Date, Open, High, Low, Close, Volume]
    if not is_doji(row1) and not is_doji(row3) and is_doji(row2):
        if row1[pclose] - row1[popen] < 0:
            # First candle is a downtrend
            min_row1 = min(row1[popen],row1[pclose])
            if row2[popen] < min_row1:
                # Next day opens at a lower price, but trades in a very narrow range
                mid_first = min_row1 + abs(row1[popen]-row1[pclose])/2
                if row3[pclose] > row3[popen] and row3[pclose] >= mid_first:
                    # Last day reverses prices and closes >= middle of first candle body
                    return True, 1 # Bullish - BUY
    return False, None

def is_evening_star_doji(row1, row2, row3):
    # row = [Date, Open, High, Low, Close, Volume]
    if not is_doji(row1) and not is_doji(row3) and is_doji(row2):
        if row1[pclose] - row1[popen] > 0:
            # First candle is an uptrend
            max_row1 = max(row1[popen],row1[pclose])
            if row2[popen] > max_row1:
                # Next day opens at a higher price, but trades in a very narrow range
                mid_first = max_row1 - abs(row1[popen]-row1[pclose])/2
                if row3[pclose] < row3[popen] and row3[pclose] <= mid_first:
                    # Last day reverses prices and closes <= middle of first candle body
                    return True, -1 # Bearish - SHORT
    return False, None    


def pattern_search(ohlc, patterns):
    found_patterns = {pattern:[] for pattern in patterns}
    max_len = len(ohlc)-1 
    for i, row in enumerate(ohlc[:-1]):
        if i<max_len-1:
            if i>0:
                # Search for 3 candle patterns
                if "Evening Star" in patterns:
                    # Search for Evening Star patterns
                    found, signal = is_evening_star_doji(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_patterns['Evening Star'].append([ohlc[i-1][date], ohlc[i+1][date], signal])
                if "Morning Star" in patterns:
                    # Search for Morning Star patterns
                    found, signal = is_morning_star_doji(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_patterns['Morning Star'].append([ohlc[i-1][date], ohlc[i+1][date], signal])
                if "Abandoned Baby" in patterns:
                    # Search for Abandoned baby patterns
                    found, signal = is_abandoned_baby(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_patterns['Abandoned Baby'].append([ohlc[i-1][date], ohlc[i+1][date], signal])
                        
            # Search for 2 candle patterns
            if "Harami Cross" in patterns:
                # Search for Harumi Cross patterns
                found, signal = is_harami_cross(row, ohlc[i+1])
                if found:
                    #pdb.set_trace()
                    found_patterns['Harami Cross'].append([mdates.num2date(row[date]), mdates.num2date(ohlc[i+1][date]), signal])
    return found_patterns

def pattern_signals(ohlc, patterns):
    found_signals = []
    max_len = len(ohlc)-1 
    for i, row in enumerate(ohlc[:-1]):
        if i<max_len-1:
            if i>0:
                # Search for 3 candle patterns
                if "Evening Star" in patterns:
                    # Search for Evening Star patterns
                    found, signal = is_evening_star_doji(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_signals.append([mdates.num2date(ohlc[i+1][date]), signal])
                if "Morning Star" in patterns:
                    # Search for Morning Star patterns
                    found, signal = is_morning_star_doji(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_signals.append([mdates.num2date(ohlc[i+1][date]), signal])
                if "Abandoned Baby" in patterns:
                    # Search for Abandoned baby patterns
                    found, signal = is_abandoned_baby(ohlc[i-1], row, ohlc[i+1])
                    if found:
                        found_signals.append([mdates.num2date(ohlc[i+1][date]), signal])
                        
            # Search for 2 candle patterns
            if "Harami Cross" in patterns:
                # Search for Harumi Cross patterns
                found, signal = is_harami_cross(row, ohlc[i+1])
                if found:
                    #pdb.set_trace()
                    found_signals.append([mdates.num2date(ohlc[i+1][date]), signal])
    return found_signals
                
if __name__ == "__main__":
    symbols = ['SPY','AAPL', 'GOOG', 'IBM', 'XOM']
    obs_dates = ['2011-03-01','2011-04-01']
    symbol = "GOOG"
    df_aapl = util.get_all_data(symbol, obs_dates)
    ohlc = ind.get_ohlc(df_aapl)
    for i, row in enumerate(ohlc[:-1]):
        if is_harami_cross(row, ohlc[i+1])[0]:
            print "Found Harami Cross!"
            print mdates.num2date(row[0]), row[1:]
            print mdates.num2date(ohlc[i+1][0]), ohlc[i+1][1:]
    #pdb.set_trace()