import util
import datetime as dt
import ManualStrategyTicktoTick as manstratt
import indicators as ind
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from marketsim import market_simulator
import GeneticAlgorithm as ga
import time
import pdb

class FakeTicker(object):
    
    def __init__(self, symbol, df, ticker_split=0.5):
        self.symbol = symbol
        self.df = df
        self.split_pos = int(ticker_split*len(df))
        self.training_df = df.ix[:-self.split_pos]
        self.testing_df = df.ix[-self.split_pos:]
        self.current_ticker = 0
        
    def get_training_data(self):
        return self.training_df
    
    def get_ticker_data(self):
        if self.current_ticker < len(self.testing_df):
            # Day is not the last, return current ticker data
           return self.testing_df.ix[self.current_ticker]
        return []
    
    def is_end_ticker(self):
        # Is ticker done?
        return self.current_ticker == len(self.testing_df)
    
    def next_ticker(self):
        # Advance ticker
        self.current_ticker += 1
           
           
def join_data_ticker(df, ticker_data, columns=['Adj Close']):
    pass
        

if __name__ == "__main__":
    symbol="GOOG"
    start_val = 100000
    testing_pctg = 0.5
    
    #In-sample period
    dates = [dt.datetime(2017,1,1), dt.datetime(2018,4,4)]
    strategy = manstratt.ManualStrategyTicktoTick()
    
    # Get data
    df = util.get_all_data(symbol, dates)
    
    pos = int(len(df.index)* testing_pctg)
    split_date = df.index[pos]
    
    # Use GeneticAlgorithm to optimize signal weights
    params = False
    gen_alg = ga.GeneticAlgorithm(symbol=symbol, dates=[dates[0], split_date], start_val=start_val, verbose=True)
    params, sharpe_ratio = gen_alg.start_ga()
    
    #pdb.set_trace()
    
    ft = FakeTicker(symbol, df, testing_pctg)
    #print (ft.get_training_data().shape)
    prices_df = ft.get_training_data()['Adj Close']
    ohlc = ind.get_ohlc(ft.get_training_data())
    
    trades = []
    ctr=0
    while not ft.is_end_ticker():
        start = time.time()
        # Get tick data
        ticker_data = ft.get_ticker_data()
        
        # Add new ticker to training data
        new_index = prices_df.index.insert(len(prices_df), ticker_data.name)
        prices_df = np.append(prices_df.values.flatten(), ticker_data['Adj Close'])
        prices_df = pd.DataFrame(prices_df, index=new_index, columns=[symbol])
        
        # Add new OHLC to ohlc array
        new_ohlc = [mdates.date2num(ticker_data.name), ticker_data['Open'], ticker_data['High'], ticker_data['Low'], ticker_data['Close'], ticker_data['Volume']]
        ohlc_list = ohlc.tolist()
        ohlc_list.append(new_ohlc)
        ohlc = np.asarray(ohlc_list)
        
        # Get order signal today
        order_signals = strategy.trade_strategy(prices_df, ohlc, ga_train = params)
        order_today = order_signals.loc[ticker_data.name]
        
        # Get trades
        trade = strategy.get_trade(symbol, order_today)
        if trade != None:
            trades.append(trade)
        
        #pdb.set_trace()
        #print (ft.get_ticker_data())
        ctr+=1
        end = time.time()
        print (str(ctr)+", took "+str(end-start)+" seconds...")
        
        # Go to next ticker
        ft.next_ticker()
    
    
    
    closing_trade = strategy.get_trade(symbol, ticker_data, True)
    if closing_trade != None:
        trades.append(closing_trade)
        
    # Check performance vs benchmark
    # Benchmark portfolio
    benchmark_trades_df = pd.DataFrame(data=[(split_date, symbol, "BUY", 1000), (prices_df.index.max(), symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    # Learned portfolio
    strategy_trades_df = pd.DataFrame(trades, columns=['Date','Symbol', 'Order', 'Shares'])
    strategy_trades_df.set_index('Date', inplace=True)
    
    # Retrieve performance stats
    print ("Performances during training period (in-sample) for {}".format(symbol))
    print ("Date Range: {} to {}".format(dates[0], dates[1]))
    print (" ")
    
    #strategy.get_signal(prices_df, ohlc)
    #pdb.set_trace()
    market_simulator(strategy_trades_df, benchmark_trades_df, start_val=start_val, insample=True)
    
        
    pdb.set_trace()
    
