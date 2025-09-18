# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler
 
# df = pd.read_csv('data/Mall_Customers.csv', index_col='CustomerID')
  
# df= pd.get_dummies(df, columns=['Genre'], drop_first=True)
 
 
 
# scaler = StandardScaler()
# df_new = df.copy()
# df_new[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']] = scaler.fit_transform(df_new[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])
 

 
# # kmeans = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, random_state=42)
# from scipy.cluster.hierarchy import dendrogram, linkage

# np.random.seed(42)

# Z = linkage(df_new, method='ward')

# # predictions = kmeans.fit_predict(df_new)
# # pridedamas naujas stulpelis su predictions
# # df['new_col'] = predictions
 
# # print(df.head(20))
 
# # print(df['new_col'].value_counts())
 
# # df.to_csv('data/new_mall_cos.csv', index=False)
 
# # print(df['Age'].value_counts().mean())
# # df.info()
 
# # plot dendogram
# # print(Z)
# plt.figure(figsize=(10, 6))
# dendrogram(Z)
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()

# # let's cut and cluster
# from scipy.cluster.hierarchy import fcluster
# # let's choose the best distance to cut the dendrogram
# max_d = 11
# clusters = fcluster(Z, max_d, criterion='distance')
# df['cluster'] = clusters
# print(df['cluster'].value_counts())
# print(df)
# # klasteriu_vidutines_reiksmes =  df.groupby('new_col').mean().round(2)
# # print(klasteriu_vidutines_reiksmes)


