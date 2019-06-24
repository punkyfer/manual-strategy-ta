import marketsim as msim
import candlestick_detector as cdsd
import ManualStrategy as manstrat
import random
import pdb
import time
import util
import indicators as ind
import pandas as pd
#https://gist.github.com/josephmisiti/940cee03c97f031188ba7eac74d03a4f

class GeneticAlgorithm(object):
    
    def __init__(self, symbol, dates, start_val, verbose=False):
        # Initialize Genetic Algorithm
        self.POP_SIZE = 100
        self.DNA_SIZE = 8
        self.MAX_GENERATIONS = 500
        # Mutation probability = 1/mutation_chance
        self.mutation_chance = 100
        self.mutation_factor = 0.1
        self.population_replacement = 0.7
        self.best_population_replacement = 0.5
        
        self.verbose = verbose
        
        self.start_val = start_val
        self.symbol = symbol
        self.dates = dates
        self.fitness_dict = {}
        # Get data
        self.prices_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
        # Get Candlestick data
        self.df_aux = util.get_all_data(symbol, dates)
        self.ohlc = ind.get_ohlc(self.df_aux)
        self.patterns_signal, self.trends, self.points_signal, self.sma_signal, self.rsi_signal, self.macd_signal, self.bb_signal = self.get_signals(self.prices_df, self.ohlc)

    def get_signals(self, price, ohlc):
        # Find Candlestick patterns and generate signals
        patterns_signal = pd.DataFrame(0, index=price.index, columns=[self.symbol])
        found_patterns = cdsd.pattern_signals(ohlc, ['Abandoned Baby', 'Morning Star', 'Evening Star', 'Harami Cross'])
        for pattern in found_patterns:
                patterns_signal.loc[pattern[0]] = pattern[1]

        # Find trendlines and iterlines points and generate signals
        trends, points_signal = ind.get_trendlines(price, charts=False)
        
        #Get SMA and generate signals
        sma = ind.get_price_sma(price, window=30)
        # sma.fillna(0, inplace=True)
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
        
        return patterns_signal, trends, points_signal, sma_signal, rsi_signal, macd_signal, bb_signal

    def random_population(self):
        # Generate Random population
        population = []
        for i in range(self.POP_SIZE):
            dna = []
            for j in range(self.DNA_SIZE-1):
                dna.append(round(random.random(), 2))
            dna.append(round(sum(dna), 2))
            population.append(dna)
            
        return population

    def fitness(self, dna):
        """
        For each gene DNA, this function calculates the sharpe_ratio.
        """
        # TODO: FIX FITNESS SO IT CALCULATES sharpe_Ratio 'INHOUSE'
        
        if str(dna) not in self.fitness_dict.keys():
            
            if dna[-1]>sum(dna[:-1]):
                self.fitness_dict[str(dna)] = 0
            else:

                #trades_df = self.manual_strat.testPolicy_ga_train(symbol=self.symbol,prices_df = self.prices_df, ohlc=self.ohlc, ga_train=dna)
                #order_signals = self.manual_strat.trade_strategy(self.prices_df, self.ohlc, dna)
                                
                # Get Signals                
                patterns_signal, points_signal, sma_signal, rsi_signal, macd_signal, bb_signal = self.patterns_signal, self.points_signal, self.sma_signal, self.rsi_signal, self.macd_signal, self.bb_signal
                
                # Generate order signals
                signal = 1 * ( ((patterns_signal==1)*dna[0] + (macd_signal==1)*dna[1] + (sma_signal==1)*dna[2] + (rsi_signal==1)*dna[3] + (bb_signal==1)*dna[4]) + (points_signal==1)*dna[5] >= dna[6]) + -1 * ( ((patterns_signal==-1)*dna[0] + (macd_signal==-1)*dna[1] + (sma_signal==-1)*dna[2] + (rsi_signal==-1)*dna[3] + (bb_signal==-1)*dna[4]) + (points_signal==-1)*dna[5] >= dna[6])
                
                if len(signal[signal[self.symbol]!=0])==0:
                    self.fitness_dict[str(dna)] = 0
                    return 0
                
                order_signals = signal * 0

                # Keep track of net signals which are constrained to -1, 0, and 1
                net_signals = 0
                for date in order_signals.index:
                    net_signals = order_signals.loc[:date].sum()
                    # If net_signals is not long and signal is to buy
                    if (net_signals<1).values[0] and (signal.loc[date]==1).values[0]:
                        order_signals.loc[date]=1
                        
                    # If net_signals is not short and signal is to sell
                    elif (net_signals > -1).values[0] and (signal.loc[date]==-1).values[0]:
                        order_signals.loc[date]=-1
                        
                #pdb.set_trace()
                
                if self.trends != None:
                    trades = order_signals[order_signals[self.symbol]!=0]
                    for i in range(len(trades)):
                        date = trades.index[i]
                        signal = trades.ix[i][0]
                        #closest_trend, slope = ind.get_closest_trendline([date, price.loc[date][0]], trends, max_sep=0.3)
                        closest_trend, slope = ind.get_closest_trendline([date, self.prices_df.loc[date][0]], self.trends)
                        if closest_trend != None:
                            if closest_trend == "Resistance":
                                #pdb.set_trace()
                                if signal==1 and slope>0:
                                    order_signals.loc[date] = 0
                            elif closest_trend == "Support":
                                if signal==-1 and slope<0:
                                    order_signals.loc[date] = 0
                
                # Alternative close positions
                open_pos = order_signals.values.sum()
                order_signals.ix[-1] = -1 * open_pos


                # Remove 0 signals
                order_signals = order_signals[order_signals[self.symbol]!=0.0]
                
                #pdb.set_trace()
                
                # Create trades dataframe
                trades=[]
                #Double-Down and Double-Up
                position = 0
                for date in order_signals.index:
                    if order_signals.loc[date].values == 1:
                        # Buy Order
                        if position == 0:
                            trades.append((date, self.symbol, "BUY", 1000))
                            position = 1
                        elif position==-1:
                            # Double-Up
                            trades.append((date, self.symbol, "BUY", 2000))
                            position = 1
                    elif order_signals.loc[date].values == -1:
                        # Sell Order
                        if position==0:
                            trades.append((date, self.symbol, "SELL", 1000))
                            position = -1
                        elif position==1:
                            # Double-Down
                            trades.append((date, self.symbol, "SELL", 2000))
                            position = -1
                    if date == order_signals.index[-1]:
                        # Last day, close open positions
                        last_trade = trades[-1]
                        trades[-1] = (last_trade[0], last_trade[1], last_trade[2], last_trade[3]-1000)
                        position = 0
                        #pdb.set_trace()
                        
                trades_df = pd.DataFrame(trades, columns=['Date', 'Symbol', 'Order', 'Shares'])
                trades_df.set_index('Date', inplace=True)
                
                #pdb.set_trace()
                
                if len(trades_df)==0:
                    sharpe_ratio = 0
                else:

                    sharpe_ratio = msim.market_simulator_train_ga(trades_df, self.prices_df, start_val=self.start_val)

                self.fitness_dict[str(dna)] = sharpe_ratio
                
        return self.fitness_dict[str(dna)]
    
    def mutate(self, dna):
        """
        For each gene in the DNA, there is a 1/mutation_chance chance that it will be
        switched out with a random character. This ensures diversity in the
        population, and ensures that is difficult to get stuck in local minima.
        """
        new_dna = []
        for c in range(self.DNA_SIZE):
            if int(random.random()*self.mutation_chance)==1:
                if random.random() >= 0.5:
                    new_dna.append(round(dna[c]+self.mutation_factor, 2))
                else:
                    new_dna.append(round(dna[c]-self.mutation_factor, 2))
            else:
                new_dna.append(round(dna[c], 2))
        return new_dna
    
    def crossover(self, dna1, dna2):
        """
        Slices both dna1 and dna2 into two parts at a random index within their
        length and merges them. Both keep their initial sublist up to the crossover
        index, but their ends are swapped.
        """
        pos = int(random.random()*self.DNA_SIZE)
        return (dna1[:pos]+dna2[pos:], dna2[:pos]+dna1[pos:])
    
    def selection_wheel(self, weighted_population):
        """
        Chooses a random element from items, where items is a list of tuples in
        the form (item, weight). weight determines the probability of choosing its
        respective item. Note: this function is borrowed from ActiveState Recipes.
        """
        weight_total = sum((item[1] for item in weighted_population))
        n = random.uniform(0, weight_total)
        for item, weight in weighted_population:
            if n < weight:
                return item
            n = n - weight
        return item
        
    def weight_population(self, population):
        # Create population tuples (person, fitness)
        weighted_population = []
        for p in population:
            #pdb.set_trace()
            #start = time.time()
            fitness_value = self.fitness(p)
            #end = time.time()
            #print "Fitness took {}".format(str(end-start))
            weighted_population.append([p, fitness_value])
        return weighted_population
                
        
    def replacement(self, weighted_old, new_population):
        # Create new population by removing self.population_replacement% of the old one and adding the new one
        pos = int(self.population_replacement*len(weighted_old))
        weighted_old.sort(key=lambda x: x[1])
        old_pop = [x for x,y in weighted_old[pos:]]
        return (old_pop+new_population)
        
    def best_replacement(self, weighted_old, new_population):
        # Create new population by selecting the best from the old and the new
        # Calculate cutting point
        pos = int(self.best_population_replacement*len(weighted_old))
        # Sort old population by fitness and select the best
        weighted_old.sort(key=lambda x: x[1])
        old_pop = [x for x,y in weighted_old[pos:]]
        # Sort new population by fitness and select the best
        weighted_new = self.weight_population(new_population)
        weighted_new.sort(key=lambda x: x[1])
        new_pop = [x for x,y in weighted_new[pos:]]
        
        return (old_pop+new_pop), (weighted_old[pos:]+weighted_new[pos:])
    
    def start_ga(self):
        # Genetic Algorithm Main Loop
        
        # Create Initial Random Population
        population = self.random_population()
        
        # Weight population by their fitness score
        weighted_population = self.weight_population(population)
            
        # Simulate all of the generations
        for generation in range(self.MAX_GENERATIONS):
            sstart = time.time()
            # Select two random individuals, based on their fitness probabilites, cross
            # their genes over at a random point, mutate them, and add them to the new population
            new_population = []
            for x in range(int(len(population)/2)):
                # Select the parents        
                parent1 = self.selection_wheel(weighted_population)
                parent2 = self.selection_wheel(weighted_population)
                
                # Crossover     
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate and append to new population
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

            
            # Replace the old population with the new one
            #population=self.replacement(weighted_population, new_population)
            #weighted_population = self.weight_population(population)
            population, weighted_population = self.best_replacement(weighted_population, new_population)

            
            #if generation % 100 == 0:
            send = time.time()
            if self.verbose: print "Generation {} took {} seconds".format(str(generation), str(send-sstart))
            
        #pdb.set_trace()
        # After the loop is done, find the DNA strand with highest fitness from the resulting population
        weighted_population = self.weight_population(population)
        weighted_population.sort(key=lambda x: x[1])
        
        #pdb.set_trace()
        
        return weighted_population[-1]
    
if __name__ == "__main__":
    #TODO: Fix main method so we can use it to test the GA
    start_val=100000
    symbol="GOOG"
        
    #In-sample period
    dates = [dt.datetime(2011,1,1), dt.datetime(2011,12,31)]
    
    #Benchmark
    benchmark_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
    #Benchmark trades
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), symbol, "BUY", 1000), (benchmark_df.index.max(), symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    gen_alg = GeneticAlgorithm(symbol=symbol, dates=dates, start_val=start_val)
    params, sharpe_ratio = gen_alg.start_ga()
    
    pdb.set_trace()
    manual_strat = manstrat.ManualStrategy()
    trades_df = manual_strat.testPolicy(symbol, dates=dates, start_val=start_val)
    
    # Retrieve performance stats
    print ("Performances during training period (in-sample) for {}".format(symbol))
    print ("Date Range: {} to {}".format(dates[0], dates[1]))
    print (" ")
    
    #pdb.set_trace()
    msim.market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=True)
    
    # Out-of-sample period
    dates = [dt.datetime(2012,1,1), dt.datetime(2012, 12, 31)]
    
    benchmark_df = util.get_data([symbol], pd.date_range(dates[0], dates[1]), addSPY=False).dropna()
        
    benchmark_trades_df = pd.DataFrame(data=[(benchmark_df.index.min(), symbol, "BUY", 1000), (benchmark_df.index.max(), symbol, "SELL", 1000)], columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_trades_df.set_index('Date', inplace=True)
    
    manual_strat = manstrat.ManualStrategy()
    trades_df = manual_strat.testPolicy(symbol, dates=dates, start_val=start_val)
    
    # Retrieve performance stats
    print ("Performances during testing period (out-of-sample) for {}".format(symbol))
    print ("Date Range: {} to {}".format(dates[0], dates[1]))
    print (" ")
    msim.market_simulator(trades_df, benchmark_trades_df, start_val=start_val, insample=False)