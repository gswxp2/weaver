import os
import pandas as pd
import numpy as np
import json

data = pd.read_csv('azure_trace.csv')
# iter over the rows
alldata = []
for index, row in data.iterrows():
    # prune too long
    if row['ContextTokens'] + row['GeneratedTokens'] >= 2000:
        continue
    alldata.append([row['ContextTokens'], row['GeneratedTokens']])

with open('azure.json', 'w') as f:
    json.dump(alldata, f)
