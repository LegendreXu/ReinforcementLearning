import time
import pickle
import pandas_datareader as pdr

res = {}

# Get  Data
lis = ['MSFT', 'AAPL', 'AMZN', 'FB', 'GOOG', 'JNJ',
            'JPM', 'V', 'PG', 'T', 'UNH', 'MA', 'HD', 'INTC', 'VZ',
            'KO', 'BAC', 'XOM', 'MRK', 'DIS', 'PFE', 'PEP', 'CMCSA',
            'CVX', 'ADBE', 'CSCO', 'NVDA', 'WMT', 'NFLX', 'CRM',
            'WFC', 'MCD', 'ABT', 'BMY', 'COST', 'BA', 'C', 'PM', 'NEE',
            'MDT', 'ABBV', 'PYPL', 'AMGN', 'TMO', 'LLY', 'HON', 'ACN', 'IBM']

for item in lis:
    data = pdr.get_data_alphavantage("NVDA", api_key='EnTeRYoUrApIKeYhErE')
    res[item] = data
    time.sleep(15)

#

file = open('USStockData.pickle','wb')
res_string  = pickle.dump(res,file)
file.close()
