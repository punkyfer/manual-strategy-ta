# https://github.com/dysonance/Trendy/blob/master/trendy.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pdb

def gentrends(df, window=1/3.0, charts=True):
    """
    Returns a Pandas dataframe with support and resistance lines.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """
    x = df.values.flatten()

    if window < 1:
        window = int(window * len(x))

    max1 = np.where(x == max(x))[0][0]  # find the index of the abs max
    min1 = np.where(x == min(x))[0][0]  # find the index of the abs min
    
    # First the max
    if max1 + window > len(x):
        max2 = max(x[0:(max1 - window)])
    else:
        max2 = max(x[(max1 + window):])

    # Now the min
    if min1 - window < 0:
        min2 = min(x[(min1 + window):])
    else:
        min2 = min(x[0:(min1 - window)])

    # Now find the indices of the secondary extrema
    max2 = np.where(x == max2)[0][0]  # find the index of the 2nd max
    min2 = np.where(x == min2)[0][0]  # find the index of the 2nd min

    # Create & extend the lines
    maxslope = (x[max1] - x[max2]) / (max1 - max2)  # slope between max points
    minslope = (x[min1] - x[min2]) / (min1 - min2)  # slope between min points
    a_max = x[max1] - (maxslope * max1)  # y-intercept for max trendline
    a_min = x[min1] - (minslope * min1)  # y-intercept for min trendline
    b_max = x[max1] + (maxslope * (len(x) - max1))  # extend to last data pt
    b_min = x[min1] + (minslope * (len(x) - min1))  # extend to last data point
    maxline = np.linspace(a_max, b_max, len(x))  # Y values between max's
    minline = np.linspace(a_min, b_min, len(x))  # Y values between min's

    # OUTPUT
    trends_aux = np.transpose(np.array((maxline, minline)))
    trends = pd.DataFrame(trends_aux, index=df.index, columns=['Max Line', 'Min Line'])
    for i, row in enumerate(trends_aux):
        trends.ix[i] = row
    

    

    if charts is True:
        plt.plot(df, label="Price", color="blue")
        plt.plot(trends['Max Line'], label="Resistance", color="green")
        plt.plot(trends['Min Line'], label="Support", color="red")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc="lower left")
        plt.xticks(rotation=50)
        plt.grid()
        #plt.show()
        plt.savefig("gentrends.jpg", bbox_inches='tight')
        plt.clf()
        plt.cla()

    return trends, maxslope, minslope

def segtrends(df, segments=2, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """
    trendlines = []
    x = df.values.flatten()
    y = np.array(x)

    # Implement trendlines
    segments = int(segments)
    maxima = np.ones(segments)
    minima = np.ones(segments)
    segsize = int(len(y)/segments)
    for i in range(1, segments+1):
        ind2 = i*segsize
        ind1 = ind2 - segsize
        maxima[i-1] = max(y[ind1:ind2])
        minima[i-1] = min(y[ind1:ind2])

    # Find the indexes of these maxima in the data
    x_maxima = np.ones(segments)
    x_minima = np.ones(segments)
    for i in range(0, segments):
        x_maxima[i] = np.where(y == maxima[i])[0][0]
        x_minima[i] = np.where(y == minima[i])[0][0]

    if charts:
        plt.plot(df, "blue")
        plt.grid(True)

    for i in range(0, segments-1):
        maxslope = (maxima[i+1] - maxima[i]) / (x_maxima[i+1] - x_maxima[i])
        a_max = maxima[i] - (maxslope * x_maxima[i])
        b_max = maxima[i] + (maxslope * (len(y) - x_maxima[i]))
        maxline = np.linspace(a_max, b_max, len(y))

        minslope = (minima[i+1] - minima[i]) / (x_minima[i+1] - x_minima[i])
        a_min = minima[i] - (minslope * x_minima[i])
        b_min = minima[i] + (minslope * (len(y) - x_minima[i]))
        minline = np.linspace(a_min, b_min, len(y))
        
        strends = pd.DataFrame(index=df.index, columns=["Max Line", "Min Line"])
        for i in range(len(maxline)):
            strends['Max Line'].ix[i] = maxline[i]
            strends['Min Line'].ix[i] = minline[i]
            
        trendlines.append(strends)

        if charts:
            plt.plot(strends['Max Line'], 'green')
            plt.plot(strends['Min Line'], 'red')

    if charts:
        #plt.show()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=50)
        plt.savefig("segtrends.jpg", bbox_inches='tight')
        plt.clf()
        plt.cla()
        

    # OUTPUT
    #return x_maxima, maxima, x_minima, minima
    return trendlines

def minitrends(x, window=20, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """
    #TODO: Fix this method since it doesn't work

    y = np.array(x)
    if window < 1:  # if window is given as fraction of data length
        window = float(window)
        window = int(window * len(y))
    x = np.arange(0, len(y))
    dy = y[window:] - y[:-window]
    crit = dy[:-1] * dy[1:] < 0


    # Find whether max's or min's
    maxi = (y[x[crit]] - y[x[crit] + window] > 0) & \
           (y[x[crit]] - y[x[crit] - window] > 0) * 1
    mini = (y[x[crit]] - y[x[crit] + window] < 0) & \
           (y[x[crit]] - y[x[crit] - window] < 0) * 1
    maxi = maxi.astype(float)
    mini = mini.astype(float)
    maxi[maxi == 0] = np.nan
    mini[mini == 0] = np.nan
    xmax = x[crit] * maxi
    xmax = xmax[~np.isnan(xmax)]
    xmax = xmax.astype(int)
    xmin = x[crit] * mini
    xmin = xmin[~np.isnan(xmin)]
    xmin = xmin.astype(int)

    # See if better max or min in region
    yMax = np.array([])
    xMax = np.array([])
    for i in xmax:
        indx = np.where(xmax == i)[0][0] + 1
        try:
            Y = y[i:xmax[indx]]
            yMax = np.append(yMax, Y.max())
            xMax = np.append(xMax, np.where(y == yMax[-1])[0][0])
        except:
            pass
    yMin = np.array([])
    xMin = np.array([])
    for i in xmin:
        indx = np.where(xmin == i)[0][0] + 1
        try:
            Y = y[i:xmin[indx]]
            yMin = np.append(yMin, Y.min())
            xMin = np.append(xMin, np.where(y == yMin[-1])[0][0])
        except:
            pass
    if y[-1] > yMax[-1]:
        yMax = np.append(yMax, y[-1])
        xMax = np.append(xMax, x[-1])
    if y[0] not in yMax:
        yMax = np.insert(yMax, 0, y[0])
        xMax = np.insert(xMax, 0, x[0])
    if y[-1] < yMin[-1]:
        yMin = np.append(yMin, y[-1])
        xMin = np.append(xMin, x[-1])
    if y[0] not in yMin:
        yMin = np.insert(yMin, 0, y[0])
        xMin = np.insert(xMin, 0, x[0])

    # Plot results if desired
    if charts is True:
        plt.plot(x, y)
        plt.plot(xMax, yMax, '-o')
        plt.plot(xMin, yMin, '-o')
        plt.grid(True)
        #plt.show()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=50)
        plt.savefig("minitrends.jpg", bbox_inches='tight')
        plt.clf()
        plt.cla()
    # Return arrays of critical points
    return xMax, yMax, xMin, yMin

def iterlines(df, window=30, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """
    #pdb.set_trace()
    try:
        symbol = df.dtypes.index[0]
    except:
        symbol = df.name


    x = df.values.flatten()
    n = len(x)
    if window < 1:
        window = int(window * n)
    sigs = np.zeros(n, dtype=float)

    i = window
    while i != n:
        if x[i] > max(x[i-window:i]): sigs[i] = 1
        elif x[i] < min(x[i-window:i]): sigs[i] = -1
        i += 1

    xmin = np.where(sigs == -1.0)[0]
    xmax = np.where(sigs == 1.0)[0]
    ymin = x[xmin]
    ymax = x[xmax]
    
    df_sigs = pd.DataFrame(0, index=df.index, columns=[symbol])
    # TODO: FIX THIS PART!!!!!
    #prev_sign = 0
    #for i in range(len(sigs)):
        #if sigs[i] == 0:
            #df_sigs.ix[i] = sigs[i]
        #else:
            #pdb.set_trace()
            #if prev_sign != 0:
                ## Sigs is inverted
                #if sigs[prev_sign] == -1 and sigs[i] == -1 and df.ix[i][0] < df.ix[prev_sign][0]:
                    ## We found a lower price point to buy
                    ##pdb.set_trace()
                    #df_sigs.ix[i]  = -1*sigs[i]
                    #df_sigs.ix[prev_sign] = 0
                    #prev_sign = i
                #elif sigs[prev_sign] == 1 and sigs[i] == 1 and df.ix[i][0] > df.ix[prev_sign][0]:
                    #df_sigs.ix[i] = -1*sigs[i]
                    #df_sigs.ix[prev_sign] = 0
                    #prev_sign = i
                #elif sigs[prev_sign] != sigs[i]:
                    #df_sigs.ix[i] = -1*sigs[i]
                    #prev_sign = i
            #else:
                #prev_sign = i
                #df_sigs.ix[i] = -1*sigs[i]
        
    for i in range(len(sigs)):
        if sigs[i] == 0:
            df_sigs.ix[i] = sigs[i]
        else:
            df_sigs.ix[i] = -1*sigs[i]
        
        
    if charts is True:
        plt.plot(df)
        plt.plot(df.index[xmin], ymin, 'go')
        plt.plot(df.index[xmax], ymax, 'ro')
        plt.grid(True)
        #plt.show()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=50)
        plt.savefig("iterlines.jpg", bbox_inches='tight')
        plt.clf()
        plt.cla()

    
            
    
    return df_sigs
