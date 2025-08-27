import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('data/titanic_train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# one-hot encode categorical variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.set_index('PassengerId', inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
# label encode
le = LabelEncoder()
columns_to_encode = ['Sex']
for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
scaler = StandardScaler()
# exclude non-numeric and binary columns from scaling
columns_to_scale = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
# param_grid = {'n_neighbors': list(range(1, 31))}
k_values = list(range(2, 31))
mean_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    mean_scores.append(scores.mean())
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x=k_values, y=mean_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Hyperparameter Tuning')
plt.show()    
best_k = k_values[np.argmax(mean_scores)]
print(f'Best K: {best_k}')

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))
