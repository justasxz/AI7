# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# import re
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree # tik atvaizdavimui
# import pandas as pd

 
# def ikelti_duomenis(failo_kelias):
#     try:
#         return pd.read_csv(failo_kelias, index_col=0)
#     except FileNotFoundError:
#         raise FileNotFoundError('Klaida, failas nerastas.')
 
# df = ikelti_duomenis('data/titanic_train.csv')
# df_test = ikelti_duomenis('data/titanic_test.csv')
 
 
# # kreipiniu istraukimas
# df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# df_test['Title'] = df_test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
 
 
# print(df.describe())
 
# # kreipiniu sugrupavimas, kas maziau uz 41 keliauja i other
# df["Title"] = df["Title"].where(df["Title"].map(df["Title"].value_counts()) >= 41, "Other")
# df_test["Title"] = df_test["Title"].where(df_test["Title"].map(df_test["Title"].value_counts()) >= 41, "Other")
 
# print(df['Title'].value_counts())
 
 
 
# # amziaus uzpildymas mediana pagal kreipini ir klase
# grupes_mediana = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
# df['Age'] = df['Age'].fillna(grupes_mediana)
 
# # amziaus uzpildymas mediana pagal kreipini
# title_mediana = df.groupby('Title')['Age'].transform('median')
# df['Age'] = df['Age'].fillna(title_mediana)
 
# # amziaus uzpildymas Age stulpelio mediana jei dar liko tusciu reiksmiu
# df['Age'] = df['Age'].fillna(df['Age'].median())
 
# # Užpildome trūkstamas amžiaus reikšmes testiniame rinkinyje
# df_test['Age'] = df_test['Age'].fillna(df['Age'].median())
 
# # amziaus grupavimas i kategorijas
# bins = [0, 17, 30, 65, df['Age'].max()]
# labels = ['vaikas', 'jaunas', 'suaugęs', 'senjoras']
# df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, ordered=True)
# df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=True, ordered=True)
 
# # papildomi laukeliai seimos dydis, ar vienas keliauja, kaina asmeniui
# df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
# df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# df['FarePerPerson'] = df['Fare'].fillna(df['Fare'].median()) / df['FamilySize'].replace(0, 1)
 
# df_test['FamilySize'] = df_test['SibSp'].fillna(0) + df_test['Parch'].fillna(0) + 1
# df_test['IsAlone'] = (df_test['FamilySize'] == 1).astype(int)
# df_test['FarePerPerson'] = df_test['Fare'].fillna(df_test['Fare'].median()) / df_test['FamilySize'].replace(0, 1)
 
 
# # ismetam nereikalingus laukelius
# df.drop(['Name', 'Ticket', 'Cabin', 'Age'], inplace=True, axis=1)
# df_test.drop(['Name', 'Ticket', 'Cabin', 'Age'], inplace=True, axis=1)
 
 
# # lyti pakeiciam i 0 ir 1
# df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype('Int64')
# df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}).astype('Int64')
 
# # embarked trukstamos reiksmes uzpildomos su mode (dazniausia reiksme)
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])
 
# # isskirstom stulpelius i bool reiksmes
# # df = pd.get_dummies(df, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
# # df_test = pd.get_dummies(df_test, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
# print(df.head())
 
# # suderinam train ir test stulpelius, kad butu tokie patys, neimam tik survived stulpelio
# X_cols = [c for c in df.columns if c != 'Survived']
# df_test = df_test.reindex(columns=X_cols, fill_value=0)
 
# # uzpildom likusius NAN skaitinese reikmese tu stulpeliu medianom
# df[X_cols] = df[X_cols].fillna(df[X_cols].median(numeric_only=True))
# df_test = df_test.fillna(df[X_cols].median(numeric_only=True))
 
 
 
# columns_with_outliers = ["Fare", "FarePerPerson",]
 
# for col in columns_with_outliers:
#     lower = df[col].quantile(0.01)   # 1 %
#     upper = df[col].quantile(0.95)   # 95 %
#     df[col] = df[col].clip(lower, upper)
 
# for col in columns_with_outliers:
#     lower = df[col].quantile(0.01)   # 1 %
#     upper = df[col].quantile(0.95)   # 95 %
#     df_test[col] = df_test[col].clip(lower, upper)
 
# X = df.drop('Survived', axis=1)
# y = df['Survived']
 
 
 
# # scaler = MinMaxScaler()

 
# x_scaled = X
# x_test_scaled = df_test
# # print(x_scaled) 
 
# # modelis = SVC(C=15)
# # gridsearchcv
# from sklearn.model_selection import GridSearchCV
# # param_grid = {'C': [0.1, 1, 10, 15, 20], 'gamma': [0.001, 0.01, 0.1, 1]}
# tree_param_grid = {'min_samples_split': [2, 5, 10],
#                     'min_samples_leaf': [1, 2, 4,6],
#                     'criterion': ['gini', 'entropy'],
#                     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None]}
# modelis = DecisionTreeClassifier(random_state=42)
# gcv = GridSearchCV(modelis, tree_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# gcv.fit(x_scaled, y)

# modelis = gcv.best_estimator_
# print("Best parameters:", gcv.best_params_)
# print("Best cross-validation score:", gcv.best_score_)
# # modelis = DecisionTreeClassifier(max_depth=3, random_state=42)



# # tree.plot_tree(modelis, filled=True, feature_names=X.columns, class_names=['Neišgyveno', 'Išgyveno'])
# # plt.show()

# spejimai = modelis.predict(x_test_scaled)
 
# submission = pd.DataFrame({
#     'PassengerId': df_test.index,
#     'Survived': spejimai
# })
# submission.to_csv('titanic_submission.csv', index=False)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
 
bazinis_modelis = DecisionTreeClassifier(max_depth=2, random_state=42)
# modelis = BaggingClassifier(estimator=bazinis_modelis, n_estimators=200, max_samples=0.8, max_features=1, random_state=42, n_jobs=-1)
modelis = AdaBoostClassifier(estimator=bazinis_modelis, n_estimators=200, random_state=42)
 
 
 
 
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
 
tree_param_grid = {'min_samples_split': [5, 7, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 4, 5, 6, 7, 8, 9],
                    'n_estimators': [50, 100, 200, 500]}
 
grid = GridSearchCV(
    estimator=rf,
    param_grid=tree_param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)
 
grid.fit(x_scaled, y)
modelis = grid.best_estimator_
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import re
from funkcijos import ikelti_duomenis
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
 
 
df = ikelti_duomenis('train.csv')
df_test = ikelti_duomenis('test.csv')
 
 
# kreipiniu istraukimas
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
 
 
print(df.describe())
 
# kreipiniu sugrupavimas, kas maziau uz 41 keliauja i other
df["Title"] = df["Title"].where(df["Title"].map(df["Title"].value_counts()) >= 41, "Other")
df_test["Title"] = df_test["Title"].where(df_test["Title"].map(df_test["Title"].value_counts()) >= 41, "Other")
 
print(df['Title'].value_counts())
 
 
 
# amziaus uzpildymas mediana pagal kreipini ir klase
grupes_mediana = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
df['Age'] = df['Age'].fillna(grupes_mediana)
 
# amziaus uzpildymas mediana pagal kreipini
title_mediana = df.groupby('Title')['Age'].transform('median')
df['Age'] = df['Age'].fillna(title_mediana)
 
# amziaus uzpildymas Age stulpelio mediana jei dar liko tusciu reiksmiu
df['Age'] = df['Age'].fillna(df['Age'].median())
 
# Užpildome trūkstamas amžiaus reikšmes testiniame rinkinyje
df_test['Age'] = df_test['Age'].fillna(df['Age'].median())
 
# amziaus grupavimas i kategorijas
bins = [0, 17, 30, 65, df['Age'].max()]
labels = ['vaikas', 'jaunas', 'suaugęs', 'senjoras']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, ordered=True)
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=True, ordered=True)
 
# papildomi laukeliai seimos dydis, ar vienas keliauja, kaina asmeniui
df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'].fillna(df['Fare'].median()) / df['FamilySize'].replace(0, 1)
 
df_test['FamilySize'] = df_test['SibSp'].fillna(0) + df_test['Parch'].fillna(0) + 1
df_test['IsAlone'] = (df_test['FamilySize'] == 1).astype(int)
df_test['FarePerPerson'] = df_test['Fare'].fillna(df_test['Fare'].median()) / df_test['FamilySize'].replace(0, 1)
 
 
# ismetam nereikalingus laukelius
df.drop(['Name', 'Ticket', 'Cabin', 'Age'], inplace=True, axis=1)
df_test.drop(['Name', 'Ticket', 'Cabin', 'Age'], inplace=True, axis=1)
 
 
# lyti pakeiciam i 0 ir 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype('Int64')
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}).astype('Int64')
 
# embarked trukstamos reiksmes uzpildomos su mode (dazniausia reiksme)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])
 
# isskirstom stulpelius i bool reiksmes
df = pd.get_dummies(df, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
print(df.head())
 
# suderinam train ir test stulpelius, kad butu tokie patys, neimam tik survived stulpelio
X_cols = [c for c in df.columns if c != 'Survived']
df_test = df_test.reindex(columns=X_cols, fill_value=0)
 
# uzpildom likusius NAN skaitinese reikmese tu stulpeliu medianom
df[X_cols] = df[X_cols].fillna(df[X_cols].median(numeric_only=True))
df_test = df_test.fillna(df[X_cols].median(numeric_only=True))
 
 
 
columns_with_outliers = ["Fare", "FarePerPerson",]
 
for col in columns_with_outliers:
    lower = df[col].quantile(0.01)   # 1 %
    upper = df[col].quantile(0.95)   # 95 %
    df[col] = df[col].clip(lower, upper)
 
for col in columns_with_outliers:
    lower = df[col].quantile(0.01)   # 1 %
    upper = df[col].quantile(0.95)   # 95 %
    df_test[col] = df_test[col].clip(lower, upper)
 
X = df.drop('Survived', axis=1)
y = df['Survived']
 
 
 
scaler = MinMaxScaler()
 
# Robustscaler 0.73684
# Standartscaler 0.76555
# MinMaxscaler 0.77511 ir modelis(SVC)
# MinMaxscaler 0.78229 ir modelis(SVC)
# MinMaxscaler 0.78708 ir modelis(SVC) ir dropintas Age stulpelis
 
 
x_scaled = scaler.fit_transform(X)
x_test_scaled = scaler.transform(df_test)
print(x_scaled)
 
 
# galimas_k = list(range(2, 50))
# visi_acc = []
# for k in galimas_k:
#     modelis_su_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) #n_jobs branduoliu sk -1 visus naudoja
#     dabartinis_acc = cross_val_score(estimator=modelis_su_k, X=x_scaled, y=y, cv=5, scoring='accuracy')
#     visi_acc.append(dabartinis_acc.mean())
 
# best_k = galimas_k[np.argmax(visi_acc)]
# print(f'Best K: {best_k}')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
 
bazinis_modelis = DecisionTreeClassifier(max_depth=2, random_state=42)
# modelis = BaggingClassifier(estimator=bazinis_modelis, n_estimators=200, max_samples=0.8, max_features=1, random_state=42, n_jobs=-1)
modelis = AdaBoostClassifier(estimator=bazinis_modelis, n_estimators=200, random_state=42)
 
 
 
 
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
 
tree_param_grid = {'min_samples_split': [5, 7, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 4, 5, 6, 7, 8, 9],
                    'n_estimators': [50, 100, 200, 500]}
 
grid = GridSearchCV(
    estimator=rf,
    param_grid=tree_param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)
 
grid.fit(x_scaled, y)
modelis = grid.best_estimator_
 
 
# base_tree = DecisionTreeClassifier(random_state=42)
 
# ada = AdaBoostClassifier(
#     estimator=base_tree,
#     random_state=42
# )
 
# param_grid = {
#     # base tree (accessed through 'estimator__')
#     "estimator__max_depth": [1, 2],
#     "estimator__min_samples_leaf": [1, 2, 5],
#     "estimator__max_features": [None, "sqrt"],
#     # AdaBoost
#     "n_estimators": [100, 200, 400],
#     "learning_rate": [0.3, 0.5, 1.0],
# }
 
# grid = GridSearchCV(
#     estimator=ada,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     scoring="accuracy"
# )
 
# grid.fit(x_scaled, y)
# modelis = grid.best_estimator_
# gridsearchcv
 
 
# param_grid = {'C': [0.1, 1, 10, 15, 20], 'gamma': [0.001, 0.01, 0.1, 1]}
# tree_param_grid = {'min_samples_split': [2, 5, 7, 10],
#                     'min_samples_leaf': [1, 2, 4],
#                     'criterion': ['gini', 'entropy'],
#                     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}
# modelis = DecisionTreeClassifier(random_state=42)
# gcv = GridSearchCV(modelis, tree_param_grid, cv=2, scoring='accuracy', n_jobs=-1)
 
# modelis.fit(x_scaled, y)
 
# modelis = gcv.best_estimator_
# print("Best parameters:", gcv.best_params_)
# print("Best cross-validation score:", gcv.best_score_)
# modelis = DecisionTreeClassifier(max_depth=3, random_state=42)
 
 
 
# tree.plot_tree(modelis, filled=True, feature_names=X.columns, class_names=['Neišgyveno', 'Išgyveno'])
# plt.show()
 
spejimai = modelis.predict(x_test_scaled)
 
modelis.fit(x_scaled, y)
spejimai = modelis.predict(x_test_scaled)
 
submission = pd.DataFrame({
    'PassengerId': df_test.index,
    'Survived': spejimai
})
submission.to_csv('titanic_submission.csv', index=False)
 