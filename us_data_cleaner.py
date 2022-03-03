import pickle
import numpy as np
import pandas as pd

data = pd.read_pickle("USStockData.pickle")
res = []
lis = data.keys()

for item in lis:
    sample = data[item].close
    temp = sample.apply(np.log).diff()[1:]
    temp = temp.to_frame()
    temp.columns = [item]
    res.append(temp)

res_pd = pd.concat(res, axis=1)
res_pd.to_pickle('USStockReturn.pkl')
