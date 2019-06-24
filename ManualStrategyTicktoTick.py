import pandas as pd
import numpy as np
import datetime as dt
import util
from marketsim import market_simulator
import indicators as ind
import candlestick_detector as cdsd
import GeneticAlgorithm as ga
#from alpha_vantage.techindicators import TechIndicators
import pdb

#TODO: Optimize GA Algorithm

class ManualStrategyTicktoTick(object):
    
    def __init__(self):
        # Initialize BestPossibleStrategy
        self.df_order_signals = pd.DataFrame()
        self.df_trades = pd.DataFrame()
        self.position = 0
        
    def trade_strategy(self, price, ohlc, ga_train=False):
        """
        Create a dataframe of order signals that maximizes portfolio return
        Signals:
            buy  = 1
            hold = 0
            sell  = -1
        """
        symbol = price.dtypes.index[0]
        
        # Find Candlestick patterns and generate signals
        patterns_signal = pd.DataFrame(0, index=price.index, columns=[symbol])
        found_patterns = cdsd.pattern_signals(ohlc, ['Abandoned Baby', 'Morning Star', 'Evening Star', 'Harami Cross'])
        for pattern in found_patterns:
            patterns_signal.loc[pattern[0]] = pattern[1]

        # Find trendlines and iterlines points and generate signals
        trends, points_signal = ind.get_trendlines(price, charts=False)

        #Get SMA and generate signals
        sma = ind.get_price_sma(price, window=30)
        sma_signal = 1 * (sma < 0.0) + -1 * (sma > 0.0)

        #Get RSI and generate signals
        rsi = ind.get_rsi(price, window=14)
        rsi_signal = 1 * (rsi < 30.0) + -1 * (rsi > 70.0)

        #Get MACD and generate signals
        macd, macdsign, macddiff = ind.get_macd(price, fast_window=12, slow_window=26)
        macd_signal = 1 * (macd>macdsign) + -1 * (macd<macdsign)

        #Get Bollinger Bands and generate signals
        bb_values = ind.get_bollinger_band_values(price, window=30)
        bb_signal = 1 * (bb_values < -1)  + -1 * (bb_values > 1)
            
        #pdb.set_trace()
        
        #Combine signals
        if type(ga_train)==bool:
            signal = 1 * ( ((patterns_signal==1)*0.2 + (macd_signal==1)*0.2 + (sma_signal==1)*0.2 + (rsi_signal==1)*0.2 + (bb_signal==1)*0.2) + (points_signal==1)*0.4 >= 0.8) + -1 * ( ((patterns_signal==-1)*0.2 + (macd_signal==-1)*0.2 + (sma_signal==-1)*0.2 + (rsi_signal==-1)*0.2 + (bb_signal==-1)*0.2) + (points_signal==-1)*0.4 >= 0.8)
        else:
            signal = 1 * ( ((patterns_signal==1)*ga_train[0] + (macd_signal==1)*ga_train[1] + (sma_signal==1)*ga_train[2] + (rsi_signal==1)*ga_train[3] + (bb_signal==1)*ga_train[4]) + (points_signal==1)*ga_train[5] >= ga_train[6]) + -1 * ( ((patterns_signal==-1)*ga_train[0] + (macd_signal==-1)*ga_train[1] + (sma_signal==-1)*ga_train[2] + (rsi_signal==-1)*ga_train[3] + (bb_signal==-1)*ga_train[4]) + (points_signal==-1)*ga_train[5] >= ga_train[6])
            
        # Create an order series with 0 as default values
        self.df_order_signals = signal * 0

        # Keep track of net signals which are constrained to -1, 0, and 1
        net_signals = 0
        for date in self.df_order_signals.index:
            net_signals = self.df_order_signals.loc[:date].sum()
            # If net_signals is not long and signal is to buy
            if (net_signals<1).values[0] and (signal.loc[date]==1).values[0]:
                self.df_order_signals.loc[date]=1
                
            # If net_signals is not short and signal is to sell
            elif (net_signals > -1).values[0] and (signal.loc[date]==-1).values[0]:
                self.df_order_signals.loc[date]=-1
        
        if trends != None:
            trades = self.df_order_signals[self.df_order_signals[symbol]!=0]
            for i in range(len(trades)):
                date = trades.index[i]
                signal = trades.ix[i][0]
                #closest_trend, slope = ind.get_closest_trendline([date, price.loc[date][0]], trends, max_sep=0.3)
                closest_trend, slope = ind.get_closest_trendline([date, price.loc[date][0]], trends)
                if closest_trend != None:
                    if closest_trend == "Resistance":
                        #pdb.set_trace()
                        if signal==1 and slope>0:
                            self.df_order_signals.loc[date] = 0
                    elif closest_trend == "Support":
                        if signal==-1 and slope<0:
                            self.df_order_signals.loc[date] = 0

        #pdb.set_trace()
        
        
        # On the last day, close any open positions
        if self.df_order_signals.values.sum() == -1:
            # Found short position
            self.df_order_signals.ix[-1] = 1
        if self.df_order_signals.values.sum() == 1:
            # Found held position
            self.df_order_signals.ix[-1]=-1
            
        # Alternative close positions
        open_pos = self.df_order_signals.values.sum()
        if open_pos != 0:
            self.df_order_signals.ix[-1] = -1 * open_pos
        
        #pdb.set_trace()
            
        return self.df_order_signals
    
    def get_signal(self, price, ohlc, window=20, ga_train=False):
        symbol = price.dtypes.index[0]
        # Find Candlestick patterns and generate signals
        patterns_signal = pd.DataFrame(0, index=price.index, columns=[symbol])
        found_patterns = cdsd.pattern_signals(ohlc, ['Abandoned Baby', 'Morning Star', 'Evening Star', 'Harami Cross'])
        for pattern in found_patterns:
            patterns_signal.loc[pattern[0]] = pattern[1]

        # Find trendlines and iterlines points and generate signals
        trends, points_signal = ind.get_trendlines(price, charts=False)

        #Get SMA and generate signals
        sma = ind.get_price_sma(price, window=30)
        sma_signal = 1 * (sma < 0.0) + -1 * (sma > 0.0)

        #Get RSI and generate signals
        rsi = ind.get_rsi(price, window=14)
        rsi_signal = 1 * (rsi < 30.0) + -1 * (rsi > 70.0)

        #Get MACD and generate signals
        macd, macdsign, macddiff = ind.get_macd(price, fast_window=12, slow_window=26)
        macd_signal = 1 * (macd>macdsign) + -1 * (macd<macdsign)

        #Get Bollinger Bands and generate signals
        bb_values = ind.get_bollinger_band_values(price, window=30)
        bb_signal = 1 * (bb_values < -1)  + -1 * (bb_values > 1)
        
        pdb.set_trace()
    
    def get_trade(self, symbol, order, last_day=False):
        # Create the corresponding trade for given order
        
        
        order_signal = order.values[0]
        date = order.name
        
        if last_day:
            # Last day close open positions
            if self.position == 1:
                return (date, symbol, "SELL", 1000)
            elif self.position == -1:
                return (date, symbol, "BUY", 1000)
            elif self.position == 0:
                return None
        
        # Double-Down and Double-Up is Active
        if order_signal == 1:
            # Buy Order
            if self.position == 0:
                self.position = 1
                return (date, symbol, "BUY", 1000)
            elif self.position==-1:
                # Double-Up
                self.position = 1
                return (date, symbol, "BUY", 2000)                
        elif order_signal == -1:
            # Sell Order
            if self.position==0:
                self.position = -1
                return (date, symbol, "SELL", 1000)                
            elif self.position==1:
                # Double-Down
                self.position = -1
                return (date, symbol, "SELL", 2000)
            
        #if last_day:
            ## Last day, close open positions
            #last_trade = trades[-1]
            #trades[-1] = (last_trade[0], last_trade[1], last_trade[2], last_trade[3]-1000)
            #position = 0
        
    
    def testPolicy(self, symbol, dates=['2011-1-1','2011-12-31'], start_val=100000, ga_train=False):
        #Test a trading policy for a stock wthin a date range and output a trades dataframe.
        
        # Get data
        prices_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
        # Get Candlestick data
        df_aux = util.get_all_data(symbol, dates)
        ohlc = ind.get_ohlc(df_aux)
        
        #Generate order signals
        order_signals = self.trade_strategy(prices_df, ohlc, ga_train)
            
        
        # Remove 0 signals
        order_signals = order_signals[order_signals!=0.0]
        
        # Create trades dataframe
        trades=[]
        # Double-Down and Double-Up is Active
        position = 0
        for date in order_signals.index:
            if order_signals.loc[date].values == 1:
                # Buy Order
                if position == 0:
                    trades.append((date, symbol, "BUY", 1000))
                    position = 1
                elif position==-1:
                    # Double-Up
                    trades.append((date, symbol, "BUY", 2000))
                    position = 1
            elif order_signals.loc[date].values == -1:
                # Sell Order
                if position==0:
                    trades.append((date, symbol, "SELL", 1000))
                    position = -1
                elif position==1:
                    # Double-Down
                    trades.append((date, symbol, "SELL", 2000))
                    position = -1
            if date == order_signals.index[-1]:
                # Last day, close open positions
                last_trade = trades[-1]
                trades[-1] = (last_trade[0], last_trade[1], last_trade[2], last_trade[3]-1000)
                position = 0
                
        self.df_trades = pd.DataFrame(trades, columns=['Date', 'Symbol', 'Order', 'Shares'])
        self.df_trades.set_index('Date', inplace=True)
        
        #pdb.set_trace()
        
        return self.df_trades
        
if __name__ == "__main__":
    start_val=100000
    symbol="GOOG"
        
    #In-sample period
    dates = [dt.datetime(2011,1,1), dt.datetime(2011,12,31)]
    
    #Benchmark
    benchmark_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
    #Benchmark trades
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), symbol, "BUY", 1000), (benchmark_df.index.max(), symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    # Use GeneticAlgorithm to optimize signal weights
    params = False
    #gen_alg = ga.GeneticAlgorithm(symbol=symbol, dates=dates, start_val=start_val, verbose=True)
    #params, sharpe_ratio = gen_alg.start_ga()
    
    #pdb.set_trace()
    
    manual_strat = ManualStrategy()
    trades_df = manual_strat.testPolicy(symbol, dates=dates, start_val=start_val, ga_train=params)
    
    # Retrieve performance stats
    print ("Performances during training period (in-sample) for {}".format(symbol))
    print ("Date Range: {} to {}".format(dates[0], dates[1]))
    print (" ")
    
    #pdb.set_trace()
    market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=True)
    
    # Out-of-sample period
    dates = [dt.datetime(2012,1,1), dt.datetime(2012, 12, 31)]
    
    benchmark_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), symbol, "BUY", 1000), (benchmark_df.index.max(), symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    manual_strat = ManualStrategy()
    trades_df = manual_strat.testPolicy(symbol, dates=dates, start_val=start_val, ga_train=params)
    
    # Retrieve performance stats
    print ("Performances during testing period (out-of-sample) for {}".format(symbol))
    print ("Date Range: {} to {}".format(dates[0], dates[1]))
    print (" ")
    market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=False)