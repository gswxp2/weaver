import os
import requests
# s = requests.Session()
# s.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
# url = 'https://openrouter.ai/api/frontend/models/find?'
# response = s.get(url)
# import json
# import time
# file_name = time.strftime("%Y%m%d-%H%M%S") + '.json'
# with open(file_name, 'w') as f:
#     json.dump(json.loads(response.text), f)
import json
file_name = "20241222-071829.json"

data = json.loads(open(file_name,"r").read())['data']
total_tokens = []
for model in data['analytics']:
    print(model, data['analytics'][model].keys())
    print(data['analytics'][model]['total_completion_tokens'])
    print(data['analytics'][model]['total_prompt_tokens'])
    completion_tokens = data['analytics'][model]['total_completion_tokens']
    prompt_tokens = data['analytics'][model]['total_prompt_tokens']
    total_tokens.append(completion_tokens + prompt_tokens)
# draw a cdf of total tokens
total_tokens.sort(reverse=True)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.ecdfplot(total_tokens)
plt.xlabel('Total Tokens')
plt.ylabel('CDF')
plt.show()
plt.savefig('total_tokens_cdf.png')
print(sum(total_tokens) / len(total_tokens))
# print top 5% models 
print(sum(total_tokens))
su = sum(total_tokens)
import numpy as np
prefix_sum = list(np.cumsum(total_tokens))
for idx in range(len(total_tokens)+1):
    if idx < 10 or idx % 5 ==0 :
        print(idx+1, prefix_sum[idx]/su)
