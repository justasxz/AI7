import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/titanic_train.csv')

df = df.drop(df.select_dtypes(include=['object']).columns, axis=1)
df.dropna(axis=1, thresh=0.5, inplace=True)
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1),df['Survived'], test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("X_train head:\n", X_train.head())
# print("X_test head:\n", X_test.head())

# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)
# print("y_train head:\n", y_train.head())
# print("y_test head:\n", y_test.head())

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10, verbose=2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(scores.mean())

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print("Predictions:", y_pred)
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # import numpy as np
# # y_pred = np.zeros_like(y_pred)  
# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)