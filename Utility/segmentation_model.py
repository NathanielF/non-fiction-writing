import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nltk
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.cm as cm
import random
random.seed(30)

X, y = make_blobs(n_samples=10000, centers=5, n_features=6,
                 random_state=0)

df_products = pd.DataFrame(data=X,
                           columns=['product_desc_' + str(x) for x in range(0, 6)])
df_products['Product_Class'] = y
df_products['Product_ID'] = df_products.index

X, y = make_blobs(n_samples=1000, centers=10, n_features=8,
                 random_state=0)

df_customer = pd.DataFrame(data=X,
                           columns=['customer_desc_' + str(x) for x in range(0, 8)])

df_customer['Customer_Class'] = y
df_customer['Customer_ID'] = df_customer.index

day1 = pd.DataFrame(zip([random.randint(0, 1000) for x in range(0, 500)],
                [random.randint(0, 10000) for x in range(0, 500)]), columns=['Customer_ID', 'Product_ID'])

day2 = pd.DataFrame(zip([random.randint(0, 1000) for x in range(0, 500)],
                [random.randint(0, 10000) for x in range(0, 500)]), columns=['Customer_ID', 'Product_ID'])

purchases = pd.concat([day1, day2], axis=0, ignore_index=True)

df_purchases = None
for purchase in range(0, len(purchases)):
    cust_id = purchases['Customer_ID'][purchase]
    prod_id = purchases['Product_ID'][purchase]
    cust_features = df_customer[df_customer['Customer_ID'] == cust_id]
    cust_features.reset_index(inplace=True)
    prod_features = df_products[df_products['Product_ID'] == prod_id]
    prod_features.reset_index(inplace=True)
    temp = pd.concat([prod_features, cust_features], axis=1)
    if df_purchases is None:
        df_purchases = pd.concat([prod_features, cust_features], axis=1)
    else:
        df_purchases = df_purchases.append(pd.concat([prod_features, cust_features], axis=1))
df_purchases.reset_index(inplace=True)

matrix = df_purchases[[x for x in df_purchases.columns if 'product_desc' in x]]
for n_clusters in range(3,10):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

kmeans = KMeans(init='k-means++', n_clusters=5, n_init=30)
kmeans.fit(matrix)
clusters = kmeans.predict(matrix)

def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
    #____________________________
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        #___________________________________________________________________________________
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        #____________________________________________________________________
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        #______________________________________
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    plt.title('Silhouette Scores per Cluster')

# define individual silouhette scores
sample_silhouette_values = silhouette_samples(matrix, clusters)
#__________________
# and do the graph
graph_component_silhouette(5, [-0.07, 1], len(X), sample_silhouette_values, clusters)



