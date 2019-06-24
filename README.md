## Trading Bot Simulator

### How does it work?

**1.** First we choose the starting amount of money for the simulation

**2.** We select a stock and a date range for the training dataset (you can choose from the available datasets or add your own to /Data)

**3.** We select a stock and a date range for the testing dataset

**4.** An empty portfolio and a benchmark portfolio are then created for comparison, the benchmark portfolio buys all the stock it can on the start date and sells it on the end date of the dataset

**5.** (Optional) We run the genetic algorithm with the testing dataset to determine how much weight we should assign to the different parameters (rsi, macd, trend lines, candlestick patterns, ...) used to determine the BUY/HOLD/SELL signals, if not using the genetic algorithm, all parameters are weighted equally

**6.** Based on the training data, the different parameters and their respective weights we generate a series of trade orders (BUY/HOLD/SELL) to maximize portfolio return

**7.** Finally we run the market simulator with the portfolio created and the benchmark, both with the training dataset (InSample) and the testing dataset (OutSample) to compare their performance


### Dependencies
* scipy
* pandas
* matplotlib
* alpha_vantage
* mpl_finance

### Files

**Candlestick Detector:** is used to find candlestick patterns in a specified OHLC (Open-High-Low-Close) dataset, it can also generate order signals based on the patterns found

**Fake Ticker:** is used to simulate the way we obtain stock information in real time, it reads a historic dataset and returns the data tick by tick (row by row) sequentially until the last row

**Genetic Algorithm:** runs a genetic algorithm to determine the strength which should be assigned to each parameter (rsi, macd, trend lines, candlestick patterns, ...) used in the trade order creation

**Indicators:** contains functions both to calculate and plot a series of technical indicators (bollinger_bands, EMA, sharpe ratio, MACD, momentum,...) for a given dataset

**Manual Strategy:** generates a series of BUY/HOLD/SELL signals to maximize portfolio returns for a given dataset based on the available parameters and their generated weights

**Manual Strategy Tick to Tick:** works ver similar to **Manual Strategy** but instead of receiving a complete dataset it receives a **Fake Ticker** and reads the data tick by tick

**Marketsim:** receives the generated portfolio and the benchmark and it computes different statistics for the portfolios (*sharpe ratio*, *cumulative return*, *standard deviation*, ...), for both the training and testing dataset

**Trendy:** is used to find and (optionally) plot support, resistance and trend lines for a given dataset
