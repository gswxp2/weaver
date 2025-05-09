import os
import pandas as pd
import json
import numpy as np
import datasets as ds
import pickle
from transformers import AutoTokenizer
data = []
data = pd.read_csv('BurstGPT_without_fails_2.csv')
# filter model = GPT-4
data = data[data['Model'] == 'GPT-4']
print(data['Request tokens'].mean())
print(data['Response tokens'].mean())
alldata = []
for index, row in data.iterrows():
    # prune too long
    if row['Request tokens'] + row['Response tokens'] >= 2000 or row["Response tokens"] <=3:
        continue
    alldata.append([row['Request tokens'], row['Response tokens']])
with open('burst.json', 'w') as f:
    json.dump(alldata, f)