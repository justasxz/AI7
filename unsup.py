# # from sklearn.cluster import KMeans
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # generate synthetic data with make_blobs
# # from sklearn.datasets import make_blobs
# # data, labels = make_blobs(n_samples=300, centers=12, cluster_std=0.60, random_state=42)

# # # # show the data
# # # plt.scatter(data[:, 0], data[:, 1], s=30)
# # # plt.show()

# # # # use elbow method to find the optimal number of clusters
# # # inertia = []
# # # K = range(1, 15)
# # # for k in K:
# # #     kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
# # #     kmeans.fit(data)
# # #     inertia.append(kmeans.inertia_)

# # # # plot the elbow curve
# # # plt.figure(figsize=(8, 4))
# # # plt.plot(K, inertia, 'bx-')
# # # plt.xlabel('Number of clusters (k)')
# # # plt.ylabel('Inertia')
# # # plt.title('Elbow Method For Optimal k')
# # # plt.show()

# # # use silhouette score to find the optimal number of clusters
# # from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# # silhouette_scores = []
# # K = range(2, 15)
# # for k in K:
# #     kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
# #     labels = kmeans.fit_predict(data)
# #     silhouette_scores.append(silhouette_score(data, labels))
# # # plot the silhouette scores
# # plt.figure(figsize=(8, 4))
# # plt.plot(K, silhouette_scores, 'bx-')
# # plt.xlabel('Number of clusters (k)')
# # plt.ylabel('Silhouette Score')
# # plt.title('Silhouette Score For Optimal k')
# # plt.show()

# # # best k
# # best_k = K[np.argmax(silhouette_scores)]

# # # fit the KMeans model
# # kmeans = KMeans(n_clusters=8, init='random', n_init=10, max_iter=300, random_state=42)

# # predictions = kmeans.fit_predict(data)
# # centers = kmeans.cluster_centers_
# # print(centers)
# # print(predictions)

# # # combine predictions and data into dataframe
# # import pandas as pd
# # df = pd.DataFrame(data, columns=['x', 'y'])
# # df['cluster'] = predictions
# # print(df.head())

# # # make some assumptions about the clusters
# # cluster_summary = df.groupby('cluster').agg({
# #     'x': ['mean', 'std'],
# #     'y': ['mean', 'std'],
# #     'cluster': 'count'
# # }).reset_index()
# # cluster_summary.columns = ['cluster', 'x_mean', 'x_std', 'y_mean', 'y_std', 'count']
# # print(cluster_summary)


# # # # plot the clusters with sns
# # # # plt.figure(figsize=(10, 6))
# # # sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predictions, palette='Set1', s=50, alpha=0.6)
# # # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Centroids')
# # # # plt.title('KMeans Clustering')
# # # # plt.legend()
# # # plt.show()


# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler
 
# df = pd.read_csv('data/Mall_Customers.csv', index_col='CustomerID')
 
# # print(df.head())
# # df.info()
# # print(df.describe())
 
 
# df= pd.get_dummies(df, columns=['Genre'], drop_first=True)
 
 
 
# scaler = StandardScaler()
# df_new = df.copy()
# df_new[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']] = scaler.fit_transform(df_new[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])
 
# # use elbow method to find the optimal number of clusters
# # inertia = []
# # K = range(1, 15)
# # for k in K:
# #     kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
# #     kmeans.fit(df_new)
# #     inertia.append(kmeans.inertia_)
 
# # plt.figure(figsize=(8, 4))
# # plt.plot(K, inertia, 'bx-')
# # plt.xlabel('Number of clusters (k)')
# # plt.ylabel('Inertia')
# # plt.title('Elbow Method For Optimal k')
# # plt.show()
 
# silhouette_scores = []
# K = range(2, 15)
# for k in K:
#     kmeans = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=42)
#     labels = kmeans.fit_predict(df_new)
#     silhouette_scores.append(silhouette_score(df_new, labels))
 
# plt.figure(figsize=(8, 4))
# plt.plot(K, silhouette_scores, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score For Optimal k')
# # plt.show()
 
# kmeans = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, random_state=42)
 
# predictions = kmeans.fit_predict(df_new)
# # pridedamas naujas stulpelis su predictions
# df['new_col'] = predictions
 
# # print(df.head(20))
 
# # print(df['new_col'].value_counts())
 
# # df.to_csv('data/new_mall_cos.csv', index=False)
 
# # print(df['Age'].value_counts().mean())
# # df.info()
 
# klasteriu_vidutines_reiksmes =  df.groupby('new_col').mean().round(2)
# print(klasteriu_vidutines_reiksmes)
 
# #  2 - A
# #  1 - B
# #  3/4 - C
# #  0 - D
# #  5 - E
# #  4 - F

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# numeric_cols = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]

# # 1) How big each cluster is
# plt.figure()
# sns.countplot(data=df, x="new_col")
# plt.title("Cluster sizes")
# plt.xlabel("Cluster")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()


# plt.figure()
# sns.scatterplot(
#     data=df,
#     x="Annual Income (k$)",
#     y="Spending Score (1-100)",
#     hue="new_col",
#     alpha=0.9
# )
# plt.title("Income vs Spending by cluster")
# plt.tight_layout()
# plt.show()

# for col in numeric_cols:
#     plt.figure()
#     sns.boxplot(data=df, x="new_col", y=col)
#     plt.title(f"{col} by cluster")
#     plt.xlabel("Cluster")
#     plt.tight_layout()
#     plt.show()

# # pairplot
# sns.pairplot(df, hue="new_col", vars=numeric_cols)
# plt.suptitle("Pairplot of clusters", y=1.02)
# plt.tight_layout()
# plt.show()

# gmeans = df.groupby("new_col")[numeric_cols].mean()
# z = (gmeans - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0)

# plt.figure()
# sns.heatmap(z.round(2), annot=True, fmt=".2f", center=0, cmap="vlag")
# plt.title("Cluster profiles (z-scores: + = above average, - = below)")
# plt.ylabel("Cluster")
# plt.tight_layout()
# plt.show()