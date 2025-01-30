import pandas as pd

data = pd.read_csv('/Users/juhong/Library/CloudStorage/OneDrive-Personal/과외/김진하/pitchain/data/ratings_small.csv',encoding='utf-8')
print(data.shape[1])
print((data.shape[1]-1)//3)
print(data['memberId'])