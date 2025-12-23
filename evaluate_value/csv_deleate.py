import pandas as pd
import csv

df=pd.read_csv('output.csv')


# 2. データの確認 (最初の5行だけ表示)
print(df.head())

# 3. データの抽出 (特定の列だけ選ぶ)
names = df[df['product_name'].str.contains('USB')]
print(names)

