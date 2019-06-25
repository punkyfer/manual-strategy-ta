import os
import pandas as pd
import datetime as dt
import util
from marketsim import market_simulator
import indicators as ind
import candlestick_detector as cdsd
import GeneticAlgorithm as ga
import ManualStrategy as ms
import ManualStrategyTicktoTick as mst

# TODO NOW: Fix marketsim error when called from interface
# TODO: Implement genetic algorithm selected case
# TODO: Make amount bought by benchmark relative to the start_val

if __name__ == "__main__":
    
    start_val=input("Choose initial ammount of funds (eg. 100000): ")
    
    stock_dict = util.get_stocks()
    stock_list = list(stock_dict.keys())

    print ("Choose stocks and dates for the training dataset")

    for i, sname in enumerate(stock_list):
        print ("{}. {}".format(i, sname))

    tr_symbol_ind = int(input("Choose a symbol (0-{}): ".format(len(stock_list)-1)))
    tr_symbol=stock_list[tr_symbol_ind]
        
    #In-sample period
    sdate, edate = util.get_dates(stock_dict[tr_symbol])
    print ("Choose dates from {:%Y/%m/%d} to {:%Y/%m/%d}".format(sdate, edate))
    tmp = input("Choose starting date (year/month/day): ").split('/')
    tr_sdate = dt.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]))

    tmp = input("Choose ending date (year/month/day): ").split('/')
    tr_edate = dt.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]))

    tr_dates = [tr_sdate, tr_edate]

    print ("Choose stocks and dates for the testing dataset")

    for i, sname in enumerate(stock_list):
        print ("{}. {}".format(i, sname))

    ts_symbol_ind = int(input("Choose a symbol (0-{}): ".format(len(stock_list)-1)))
    ts_symbol=stock_list[ts_symbol_ind]
        
    #Out-sample period
    sdate, edate = util.get_dates(stock_dict[ts_symbol])
    print ("Choose dates from {:%Y/%m/%d} to {:%Y/%m/%d}".format(sdate, edate))
    tmp = input("Choose starting date (year/month/day): ").split('/')
    ts_sdate = dt.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]))

    tmp = input("Choose ending date (year/month/day): ").split('/')
    ts_edate = dt.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]))
    
    ts_dates = [ts_sdate, ts_edate]
    
    #Benchmark
    benchmark_df = util.get_data([tr_symbol], pd.date_range(tr_dates[0], tr_dates[1]), addSPY=False).dropna()
        
    #Benchmark trades
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), tr_symbol, "BUY", 1000), (benchmark_df.index.max(), tr_symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)

    print ("Benchmark portfolio created...")
    
    use_ga = bool(int(input("Use genetic algorithm to optimize signal weights? (0-No,1-Yes): ")))
    # Use GeneticAlgorithm to optimize signal weights
    params = False
    #gen_alg = ga.GeneticAlgorithm(symbol=symbol, dates=dates, start_val=start_val, verbose=True)
    #params, sharpe_ratio = gen_alg.start_ga()
    
    #pdb.set_trace()
    print ("Strategy Generation")
    print ("0. Manual Strategy")
    print ("1. Manual Strategy Tick to Tick")
    ttt_strat = bool(int(input("Choose (0 or 1): ")))
    if ttt_strat:
        manual_strat = mst.ManualStrategyTicktoTick()
    else:
        manual_strat = ms.ManualStrategy()

    trades_df = manual_strat.testPolicy(tr_symbol, dates=tr_dates, start_val=start_val, ga_train=params)
    
    # Retrieve performance stats
    print ("Performances during training period (in-sample) for {}".format(tr_symbol))
    print ("Date Range: {} to {}".format(tr_dates[0], tr_dates[1]))
    print (" ")
    
    #pdb.set_trace()
    market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=True)
    
    # Out-of-sample period
    dates = [dt.datetime(2012,1,1), dt.datetime(2012, 12, 31)]
    
    benchmark_df = util.get_data([ts_symbol], pd.date_range(ts_dates[0], ts_dates[1]), addSPY=False).dropna()
        
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), ts_symbol, "BUY", 1000), (benchmark_df.index.max(), ts_symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    manual_strat = ManualStrategyTicktoTick()
    trades_df = manual_strat.testPolicy(ts_symbol, dates=ts_dates, start_val=start_val, ga_train=params)
    
    # Retrieve performance stats
    print ("Performances during testing period (out-of-sample) for {}".format(ts_symbol))
    print ("Date Range: {} to {}".format(ts_dates[0], ts_dates[1]))
    print (" ")
    market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=False)
