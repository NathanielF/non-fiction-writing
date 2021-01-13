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

from sklearn.preprocessing import StandardScaler
X = df_customer[[x for x in df_purchases.columns if 'customer_desc' in x]]
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=30)
kmeans.fit(X)
clusters = kmeans.predict(X)
X['cluster'] = clusters
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_pca = pd.DataFrame(sklearn_pca.fit_transform(X[[col for col in X.columns if
                                         not 'cluster' in col]]))
X = pd.concat([X, Y_pca], axis=1)
X[[0, 1, 'cluster']].plot.scatter(x=0,
                      y=1,
                      c='cluster',
                      colormap='viridis')
plt.title("Principle Components Representation - Coloured by inferred Clusters")
plt.ylabel("Principle Component 1")
plt.xlabel("Principle Component 2")
plt.style.use('default')
plt.show()

X_std = StandardScaler().fit_transform(X)
## Covariance Decomposition
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

## Correlation Decomposition
cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

## svd
u,s,v = np.linalg.svd(X_std.T)
u

## Explained Variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(var_exp)
print(cum_var_exp)

plt.figure(figsize=(8, 8))

plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.style.use('default')

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Call the function. Use only the 2 PCs.
myplot(X[[0, 1]],np.transpose(sklearn_pca.components_[0:2, :]))
plt.show()

import seaborn as sns
ax = sns.heatmap(sklearn_pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,sklearn_pca.n_components_+1)],
                 xticklabels=['customer_desc_0', 'customer_desc_1', 'customer_desc_2',
       'customer_desc_3', 'customer_desc_4', 'customer_desc_5',
       'customer_desc_6', 'customer_desc_7'],
                 #cbar_kws={"orientation": "horizontal"}
                 )
ax.set_aspect("equal")
plt.title("Component Weightings of the original observable variables")
plt.style.use('default')