import pandas as pd
import numpy as np


# data = [92, 60, 70, 80, 90, np.nan, 130, np.nan, 110, 120, np.nan]
# series = pd.Series(data, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
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
# print(series.sort_values())
# test = pd.date_range('2023-05-10', periods=365, freq='D').strftime('%Y-%m-%d')
# values = np.random.randint(0, 100, size=365)
# df = pd.Series(values, index = test)
# print(df.head())


data = {
    'vardas': ['Jonas', 'Petras', 'Ona', 'Marytė', np.nan, 'Tomas', 'Andrius', 'Rasa', 'Eglė', 'Rasa',np.nan],
    'amžius': [25, 30, 22, 28, 35, np.nan, 40, 29, 31, 27, 33],
    'miestas': ['Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys', 'Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys', np.nan], 
    'pajamos': [1500, 2000, 1800, 2200, np.nan, 2500, 2300, 2100, 1900, 2400, np.nan]
    }

df = pd.DataFrame(data)
# print(df)
# print(df.shape)
# df.info()
# info_df = df.describe(include='all')
# print(df['amžius'].median())
# print(df['amžius'])
# print(df[['amžius', 'pajamos']])
# print(df.iloc[1])
# print(df.loc[1])
# # filter by one column
# # print(df[df['amžius'] > 30])
# # filter by multiple columns
# print(df[(df['amžius'] > 30) & (df['pajamos'] > 2000)]) # AND
# print(df[(df['amžius'] > 30) | (df['pajamos'] > 2000)]) # OR

# naujas = inplace=False # is esmes padaro, kad metodas turetu return
# inplace=True # is esmes padaro, kad metodas turetu return None ir pakeistu originala

# print(pd.isnull(df).sum())

# df['amžius'].fillna(0, inplace=True)
# df['pajamos'].fillna(df['pajamos'].mean(), inplace=True)
# print(df)

# df.dropna(inplace=True, thresh=2)  # pašalina eilutes su NaN reikšmėmis
# df.dropna(axis=1,  thresh=10, inplace=True)  # pašalina stulpelius su NaN reikšmėmis
# df['amžius'].dropna(inplace=True)  # pašalina NaN reikšmes iš 'amžius' stulpelio
# print(df)

# df['naujas'] = df['amžius'] * 2  # sukuria naują stulpelį 'naujas', kurio reikšmės yra '
# df['naujas2'] = 5
# df['naujas3'] = [10,10,10,5,10,10,10,10,10,10,10]  # sukuria naują stulpelį 'naujas3' su fiksuotomis reikšmėmis
# print(df)

# df.drop(columns=['naujas2', 'naujas3'], inplace=True)  # pašalina stulpelius 'naujas2' ir 'naujas3'
# print(df)
# df.replace({'vardas': {np.nan: 'Nežinoma'}}, inplace=True)  # pakeičia NaN reikšmes stulpelyje 'vardas' į 'Nežinoma'
# print(df)

# print(df.groupby('miestas').mean(numeric_only=True) )  # grupuoja pagal 'miestas' stulpelį ir apskaičiuoja vidurkius
# for city, group in df.groupby('miestas'):
#     print(f'Miestas: {city}')
#     print(group)

# print(df.agg({
#     'amžius': ['mean', 'max', 'min'],
#     'pajamos': ['mean', 'sum']}))

# joined_df = pd.concat([df, df], ignore_index=True, axis=1)  # sujungia du DataFrame'us
# print(joined_df)
# df2 = pd.DataFrame({
#     'vardas': ['Jonas', 'Petras', 'Ona', 'Marytė', 'Tomas'],
#     'ugis': [180, 175, 160, 165, 170],
#     'svoris': [75, 70, 60, 65, 80]
# })
# merged_df = pd.merge(df, df2, on='vardas', how='left')  # sujungia du DataFrame'us pagal 'vardas' stulpelį
# print(merged_df)
df = pd.read_csv('data/titanic_train.csv')

print(df.head())
print(df.describe())