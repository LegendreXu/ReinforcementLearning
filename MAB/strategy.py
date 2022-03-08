import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def mom(data, initial_budget = 10000, n_stocks = 1, trading_cost = 0.0/1000, lookback = 120, rebalance = 20, risk_aversion = 1, America=False):
    
    res = []
    lis = data.keys()

    for item in lis:
        temp = data[item].close
        temp = temp.to_frame()
        temp.columns = [item]
        res.append(temp)
    
    df = pd.concat(res,axis=1)
    
    if America:
        df = df[df.index>'2016-01-01']
      
    df.dropna(0,inplace=True)
    data_last = df
    net_value = pd.DataFrame([0]*data_last.shape[0], index=data_last.index, columns=['net_value'])
    cash = pd.DataFrame([0]*data_last.shape[0], index=data_last.index,columns=['cash'])
    # Target Weight
    weight = pd.DataFrame(np.zeros(shape=data_last.shape),columns=data_last.columns, index=data_last.index)
    # Practical Position
    stock_pos = pd.DataFrame(np.zeros(shape=data_last.shape),columns=data_last.columns, index=data_last.index)



    for i in range(0,lookback):
        cash.iloc[i] = initial_budget
        net_value.iloc[i] = initial_budget


    for i in range(lookback, data_last.shape[0]):
    #for i in range(lookback, 160):
        today_price = data_last.iloc[i-1:i+1]
        today_ret = (today_price - today_price.shift()).iloc[1:]/today_price.shift().iloc[1]
        today_ret = today_ret.fillna(0)
        if (i-lookback)%rebalance == 0:
            temp = data_last.iloc[i-lookback:i-1].dropna(axis=1)
            temp_ret = (np.log(temp)).diff()[1:]
            ret = temp_ret.sum(axis=0)
            vol = temp_ret.std(axis=0)
            momentum = ret - risk_aversion * vol
            if (momentum.shape[0] >= n_stocks):
                idx_buy = momentum.nlargest(n_stocks).index
                weight.iloc[i][idx_buy] = 1.0/n_stocks
            else:
                n_mom = momentum.shape[0]
                idx_buy = momentum.nlargest(n_mom).index
                weight.iloc[i][idx_buy] = 1.0/n_mom
            if i > lookback:
                cash.iloc[i] = stock_pos.iloc[i-1][idx_pre].sum()*(1-trading_cost)+cash.iloc[i-1]
            else:
                cash.iloc[i] = cash.iloc[i-1]

            available_cash = cash.iloc[i].values[0]
            for stock_id in idx_buy:
                target_pos = weight.iloc[i][stock_id]*available_cash
                n_temp = np.floor(target_pos/(data_last.iloc[i-1][stock_id]*(1+trading_cost)))
                stock_pos.iloc[i][stock_id] = n_temp*data_last.iloc[i-1][stock_id]
                cash.iloc[i] -= n_temp*data_last.iloc[i-1][stock_id]*(1+trading_cost)
            stock_pos.iloc[i] = stock_pos.iloc[i] *(today_ret[stock_pos.columns]+1)
            idx_pre = idx_buy
        else:
            weight.iloc[i] = weight.iloc[i-1]
            stock_pos.iloc[i] = stock_pos.iloc[i-1]*(today_ret[stock_pos.columns]+1)
            cash.iloc[i] = cash.iloc[i-1]
        net_value.iloc[i] = cash.iloc[i] + sum(stock_pos.iloc[i])

    net_value/=initial_budget
    ret = np.log(net_value).diff().dropna()
    annual_ret = ret.mean().values[0]*252
    annual_vol = ret.std().values[0]*np.sqrt(252)
    sharpe = annual_ret/annual_vol
    s=(ret+1).cumprod()
    mdd_pct = (1-np.ptp(s)/s.max()).values[0]
    mdd_usd = (net_value.cummax()-net_value).max()
    mdd_usd = mdd_usd.values[0]
    r_d = ret[ret<0].dropna()
    down_deviation = np.sqrt(252)*(np.sqrt((0 - r_d)**2).sum())/r_d.shape[0]
    down_deviation = down_deviation.values[0]
    sortino = annual_ret/ down_deviation

    print("Annualized Return: ", annual_ret*100, "%")
    print("Annualized Volatility: ", annual_vol*100, "%")
    print("Downside Deviation: ", down_deviation*100, "%")
    print("Max Drawdown(in percentage): ", mdd_pct*100, "%")
    print("Max Drawdown(in dollars): ", mdd_usd)
    print("Sharpe Ratio: ", sharpe)
    print("Sortino Ratio: ", sortino)


    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    i = np.argmax(np.maximum.accumulate(net_value) - net_value) # end of the period
    j = np.argmax(net_value[:i]) # start of period
    ax.plot(net_value)
    ax.plot([net_value.index[i], net_value.index[j]], [net_value.iloc[i], net_value.iloc[j]], 'o', color='Red', markersize=10)
    tick_space = 250
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_space))
    plt.grid()
    plt.show()
    
def EW(data, America=False):
    
    res = []
    lis = data.keys()

    for item in lis:
        temp = data[item].close
        temp = temp.to_frame()
        temp.columns = [item]
        res.append(temp)
    
    df = pd.concat(res,axis=1)
    
    if America:
        df = df[df.index>'2016-01-01']
      
    df.dropna(0,inplace=True)
    data_last = df
    net_value = pd.DataFrame([0]*data_last.shape[0], index=data_last.index, columns=['net_value'])
    for i in range(data_last.shape[0]):
        net_value.iloc[i] = data_last.iloc[i].mean()

    ret = np.log(net_value).diff().dropna()
    annual_ret = ret.mean().values[0]*252
    annual_vol = ret.std().values[0]*np.sqrt(252)
    sharpe = annual_ret/annual_vol
    s=(ret+1).cumprod()
    mdd_pct = (1-np.ptp(s)/s.max()).values[0]
    mdd_usd = (net_value.cummax()-net_value).max()
    mdd_usd = mdd_usd.values[0]
    r_d = ret[ret<0].dropna()
    down_deviation = np.sqrt(252)*(np.sqrt((0 - r_d)**2).sum())/r_d.shape[0]
    down_deviation = down_deviation.values[0]
    sortino = annual_ret/ down_deviation

    print("Annualized Return: ", annual_ret*100, "%")
    print("Annualized Volatility: ", annual_vol*100, "%")
    print("Downside Deviation: ", down_deviation*100, "%")
    print("Max Drawdown(in percentage): ", mdd_pct*100, "%")
    print("Max Drawdown(in dollars): ", mdd_usd)
    print("Sharpe Ratio: ", sharpe)
    print("Sortino Ratio: ", sortino)


    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    i = np.argmax(np.maximum.accumulate(net_value) - net_value) # end of the period
    j = np.argmax(net_value[:i]) # start of period
    ax.plot(net_value)
    ax.plot([net_value.index[i], net_value.index[j]], [net_value.iloc[i], net_value.iloc[j]], 'o', color='Red', markersize=10)
    tick_space = 250
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_space))
    plt.grid()
    plt.show()