import pandas as pd
import numpy as np


data = [92, 60, 70, 80, 90, np.nan, 130, np.nan, 110, 120, np.nan]
series = pd.Series(data, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
# print(series)
# naujas = series.head(3)
# print(series.head(2))
# print(series.size)
# # print(series.dtype)
# print(series.values)
# print(series.index)
# print(series[3])
# print(series.iloc[3]) # rekomenduojamas naudoti, jeigu indeksas yra skaicius
# print(series.loc['c'])
# series = series + 5
# print(series)

# print(series.isnull())
# how_many_nulls = series.isnull().sum()
# print(f'Null reikšmių kiekis: {how_many_nulls}')
# # series = series.dropna()
# # print(series)
# series.fillna(series.mean(), inplace=True)  
# print(series)
# print(series.value_counts())
# print(series.unique())
# print(series.nunique())

# series = series.apply(lambda x: x * 2 if x > 100 else x + 10)
# print(series)
# series_map_dict = {50: 100, 60: 120, 70: 140, 80: 160, 90: 180, 100: 200, 110: 220, 120: 240}
# series = series.map(series_map_dict)
# # series = series.map(lambda x: x * 2 if x > 100 else x + 10)
# print(series)
print(series.sort_values())
test = pd.date_range('2023-05-10', periods=365, freq='D').strftime('%Y-%m-%d')
values = np.random.randint(0, 100, size=365)
df = pd.Series(values, index = test)
print(df.head())