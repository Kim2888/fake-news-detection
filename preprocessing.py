import pandas as pd

df = pd.read_csv('data/WELFake_Dataset.csv')

print(df.shape)        # 有多少行、多少列
print(df.columns)      # 列名叫什么
print(df.head())       # 前5行长什么样
print(df.isnull().sum()) # 有没有空值