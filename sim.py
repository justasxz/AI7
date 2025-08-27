import pandas as pd
import numpy as np

df = pd.read_csv('data/titanic_train.csv')
print(df.isna())
# histogram of age
import matplotlib.pyplot as plt
import seaborn as sns
# sns.histplot(df['Age'].dropna(), kde=True)
# plt.show()
# histogram of passengerID
# sns.histplot(df['SibSp'], kde=True)
# plt.show()



# def detect_anomalies_iqr(data):
#     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1

#     # Define bounds for anomalies
#     lower_bound = Q1 - 2 * IQR # 20 - (1.5 *18) = -7
#     upper_bound = Q3 + 2 * IQR # 38 + (1.5 * 18) = 65

#     # Return the anomalies
#     anomalies = [x for x in data if x < lower_bound or x > upper_bound]
#     return anomalies

# anomalies = detect_anomalies_iqr(df['SibSp'].dropna())
# print("Anomalies in Age column:", anomalies)
