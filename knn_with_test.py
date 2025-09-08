import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/titanic_train.csv')
def preprocess_data(df):
    # fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # one-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df.set_index('PassengerId', inplace=True)
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    # label encode
    le = LabelEncoder()
    columns_to_encode = ['Sex']
    for col in columns_to_encode:
        df[col] = le.fit_transform(df[col])
    return df

df = preprocess_data(df)
X = df.drop('Survived', axis=1)
y = df['Survived']



df_test = pd.read_csv('data/titanic_test.csv')
import seaborn as sns
import matplotlib.pyplot as plt
# sns.countplot(data=df, x='Survived')
# plt.show()
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# sns.histplot(df['Age'].dropna(), kde=True, bins=10)
# plt.show()

# columns_to_plot = df.select_dtypes(include=[np.number]).columns.tolist()

# fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(8, 5 * len(columns_to_plot)))

# for i, col in enumerate(columns_to_plot):
#     sns.histplot(df[col].dropna(), kde=True, bins=10, ax=axes[i, 0])
#     axes[i, 0].set_title(f'Histogram of {col}')
    
#     sns.boxplot(x=df[col], ax=axes[i, 1])
#     axes[i, 1].set_title(f'Boxplot of {col}')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# sns.histplot(df['Age'].dropna(), kde=True, bins=10, ax=axes[0,0])
# sns.histplot(df['Age'].dropna(), kde=True, bins=10)
# sns.boxplot(x=df['Age'], ax=axes[0,1])
# plt.tight_layout()
# plt.show()

# for col in columns_to_plot:
#     sns.histplot(df[col].dropna(), kde=True, bins=10)
#     plt.title(f'Histogram of {col}')
#     plt.show()
# sns.histplot(df['Fare'].dropna(), kde=True, bins=5)
# plt.show()


# X_test_final = preprocess_data(df_test)

# scaler = StandardScaler()
# # exclude non-numeric and binary columns from scaling
# columns_to_scale = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']

# X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
# X_test_final[columns_to_scale] = scaler.transform(X_test_final[columns_to_scale])
# # param_grid = {'n_neighbors': list(range(1, 31))}
# k_values = list(range(2, 31))
# mean_scores = []
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#     mean_scores.append(scores.mean())
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.lineplot(x=k_values, y=mean_scores)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Cross-Validated Accuracy')
# plt.title('KNN Hyperparameter Tuning')
# # plt.show()    
# best_k = k_values[np.argmax(mean_scores)]
# print(f'Best K: {best_k}')

# knn = KNeighborsClassifier(n_neighbors=best_k)
# knn.fit(X, y)

# # let's make test columns the same as train
# missing_cols_in_train = set(X_test_final.columns) - set(X.columns)
# print(missing_cols_in_train)
# missing_cols_in_test = set(X.columns) - set(X_test_final.columns)
# print(missing_cols_in_test)
# # drop missing columns in test
# X_test_final.drop(columns=missing_cols_in_train, inplace=True)

# y_pred = knn.predict(X_test_final)

# # let's create a submission file
# submission = pd.DataFrame({
#     'PassengerId': df_test['PassengerId'],
#     'Survived': y_pred
# })
# submission.to_csv('data/titanic_submission.csv', index=False)

# df['column'].fillna(df['column'].median(), inplace=True)
# df.fillna({'Column': df['Column'].median()}, inplace=True)
# df[col].method(value, inplace=True)
# df.method({col: value}, inplace=True)
# df[col] = df[col].method(value)