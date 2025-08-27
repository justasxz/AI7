import pandas as pd
import numpy as np

df = pd.read_csv('data/titanic_train.csv')
df.drop_duplicates