#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:02:28 2018

@author: demaggis
"""

# Data and Packages

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------------------------------------------------------------

## Importing the data

data = pd.read_csv('wines_properties.csv',sep=',')

# dropping missing values 
data.dropna(how = "all", inplace=True)
data.head()

# dropping categorical variable 'Customer_segment'
X = data.iloc[:, 0:13]
X.head()

# ---------------------------------------------------------------------------------------------------

#############################  1. PCA with the Circle of Correlations  ##############################

# Standardizing
X_s = StandardScaler().fit_transform(X)

## A. PCA from Scratch

# eigenvalues and eigenvectors
mean_vector = np.mean(X_s, axis = 0)
N = X_s.shape[0]
covariance_matrix = (X_s - mean_vector).T.dot( (X_s - mean_vector) ) / (N - 1) #unbiased 
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values, eigen_vectors

correlation_matrix = np.corrcoef(X_s.T)
pd.DataFrame(correlation_matrix)

eigen_values_corr, eigen_vectors_corr = np.linalg.eig(correlation_matrix)
eigen_vectors
eigen_vectors_corr

# ratio of explained variance 
tot_eig_vals = sum(eigen_values)
tot_eig_vals
sorted_eigenvalues = sorted(eigen_values, reverse=True)
variance_explained = [ (i / tot_eig_vals)*100 for i in sorted_eigenvalues ]
variance_explained

# Sorting the eigenvectors based on the eigenvalues 
eigen_vectors_values = [ ( np.abs(eigen_values[i]), eigen_vectors[:, i] ) 
                        for i in range(len(eigen_values)) ]

eigen_vectors_values.sort()
eigen_vectors_values.reverse()
eigen_vectors_values[0][1].reshape(len(eigen_vectors_values), -1)

# top-2 eigenvectors matrix 
top2_eigenvectors = np.hstack( ( eigen_vectors_values[0][1].reshape(13, -1), 
                             eigen_vectors_values[1][1].reshape(13, -1) ) )

# first 2 PCs
top2_eigenvectors
# new Y
Y_ = X_s.dot(top2_eigenvectors)

## B. PCA with the simple command --> counterproof

my_pca = PCA(n_components=2)
Y = my_pca.fit_transform(X_s)
PCs = my_pca.components_

# Verifying the outputs (if they match)

for i in range(top2_eigenvectors.shape[0]):
    print(abs(round(top2_eigenvectors[i,0],5)) == abs(round(PCs.T[i,0],5)))
    print(abs(round(top2_eigenvectors[i,1],5)) == abs(round(PCs.T[i,1],5)))
    # oK!

# N.B. 'top2_eigenvectors' and 'PCs' have opposite signs (that's why we compare their absolute values), but this is not important since:
#       the variance does not change and the weights also change the sign, thus not altering the PCs interpretation


# Plotting
fig = plt.figure(figsize=(5,5))
plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
           PCs[0,:], PCs[1,:], 
           angles='xy', scale_units='xy', scale=1)

# Labels based on feature names 
feature_names = [data.columns[i] for i in range(PCs.shape[1])]
for i,j,z in zip(PCs[1,:]+0.02, PCs[0,:]+0.02, feature_names):
    plt.text(j, i, z, ha='center', va='center')

# Unit circle
circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
plt.gca().add_artist(circle)

#axis
plt.axis('equal')
plt.xlim([-1.0,1.0])
plt.ylim([-1.0,1.0])
plt.xlabel('PC 0')
plt.ylabel('PC 1')

plt.show()

#########################################  Interpretation  ##########################################

# The circle of correlations is used to give a visual representation of the initial variables in the factor space. 
# PCA analysis identifies two uncorrelated dimensions within the data which - in our example- are referenced to as
# PC 0 and PC 1. 
# The nature of PCA analysis requires that variables which are perfectly correlated with one dimension are independent
# of the other: this implies that variables correlated with PC 0 are not correlated with PC 1 and vice versa. 
# In our case, the variables  “Malic Acid”, “Color Intensity”, “Magnesium”, “Alcol”, “Proline”, “Ash”, 
# are negatively correlated with PC1, while the variables “Hue” and “OD280” are positively correlated 
# (this is not a case of perfect correlation) with PC 1.
# The variables “Non Flavanoid”, “Phenols”, “Ash Alcanity” are perfectly correlated in a negative way with PC 0, 
# while “Flavanoids”, “Proanthocyanins” and “Total Phenols ” are perfectly correlated in a positive way with PC 0. 
#In addition, “Malic Acid” is weakly correlated in a negative with both factors.

# ---------------------------------------------------------------------------------------------------

####################################  2. Hierarchical Clusters  #####################################

#Dendogram

dendogram = dendrogram(linkage(X_s, method='complete', metric='euclidean'))
plt.title('Dendogram')
plt.ylabel('Euclidean distance')

# ---------------------------------------------------------------------------------------------------

##################################  3. K-Means Clusters Analysis  ###################################

# 3.A Silhouette analysis --> selected n. of clusters: 3

km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(Y)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(Y, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

#########################################  Interpretation  ##########################################

# Silhouette analysis shows us the distance between the resulting 3 clusters. 
# In particular, it measures how different each observation in one cluster is from the observations assigned to the other clusters. 
# The closer the value of the silhouette is to one, the farther the observations are from the neighboring clusters. 
# In our case, the three cluster have a significant degree of difference as their silhouette is close enough to one. 
# Furthermore, the thickness of the clusters represents the number of obs contained in each one of them: 
# the three clusters here represented have the same thickness and thus are similar in size.

# ---------------------------------------------------------------------------------------------------

# 3.B Scatter plot 

# cluster 1
plt.scatter(Y[y_km == 0, 0],
            Y[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
# cluster 2
plt.scatter(Y[y_km == 1, 0],
            Y[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')
# cluster 3
plt.scatter(Y[y_km == 2, 0],
            Y[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3')
# centroids 
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

# Bad clustering --> n. clusters: 2

km2 = KMeans(n_clusters=2, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km2 = km2.fit_predict(Y)

cluster_labels = np.unique(y_km2)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(Y, y_km2, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km2 == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

# Scatter plot

# cluster 1
plt.scatter(Y[y_km2 == 0, 0],
            Y[y_km2 == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
# cluster 2
plt.scatter(Y[y_km2 == 1, 0],
            Y[y_km2 == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')

# centroids 
plt.scatter(km2.cluster_centers_[:, 0],
            km2.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------

# 3.C For each cluster, which “original” variables (ex ante the PCA) are more important?

# As per below codes (from line 315 to line 364), we computed the mean values of each one of the original variable 
# (the 13 ones) per cluster and then we plotted the distances among these mean values per pair of clusters so as 
# to look at how they differ from each other.
# As evidenced in the graph above, therefore, the most important variables before PCA are the ones that mainly contribute to
# differentiating each cluster from one another, as they take on values which are significantly different in the 
# three clusters (the ones having negative/positive peaks).
# In detail, the most significant variables by looking at the plotted pair-clusters distances are: 
# Alcohol, Ash_Alcalinity, total_phenols and Color_intensity.

# ---------------------------------------------------------------------------------------------------

# most important variables for each cluster
cl_1 = []
cl_2 = []
cl_3 = []

for i in range(len(y_km)):
    if y_km[i] == 0:
        cl_1.append(np.array(X_s[i,]))
    elif y_km[i] == 1:
        cl_2.append(np.array(X_s[i,]))
    else:
        cl_3.append(np.array(X_s[i,]))

# Getting the mean values of each "original" variable (the 13 ones) per cluster

#cluster 1
var_cl1 = []
var_mean_cl1 = []
        
for i in range(X_s.shape[1]):
    for l in range(len(cl_1)):
        var_cl1.append(cl_1[l][i])
    var_mean_cl1.append(np.mean(var_cl1))
        
#cluster 2
var_cl2 = []
var_mean_cl2 = []
        
for i in range(X_s.shape[1]):
    for l in range(len(cl_2)):
        var_cl2.append(cl_2[l][i])
    var_mean_cl2.append(np.mean(var_cl2))

#cluster 3
var_cl3 = []
var_mean_cl3 = []
        
for i in range(X_s.shape[1]):
    for l in range(len(cl_3)):
        var_cl3.append(cl_3[l][i])
    var_mean_cl3.append(np.mean(var_cl3))
    
# compare the mean value among clusters: the higher the difference, the higher the "between variability" (between clusters)

diff_12 = [var_mean_cl1[i] - var_mean_cl2[i] for i in range(len(var_mean_cl1))]
diff_13 = [var_mean_cl1[i] - var_mean_cl3[i] for i in range(len(var_mean_cl1))]
diff_23 = [var_mean_cl2[i] - var_mean_cl3[i] for i in range(len(var_mean_cl1))]
diff = np.c_[diff_12,diff_13,diff_23]
plt.plot(diff)
plt.legend(['diff_Cl1_Cl2','diff_Cl1_Cl3','diff_Cl2_Cl3'])
plt.title('Differences of Variables mean values between Clusters')

# ---------------------------------------------------------------------------------------------------
        
# 3.D Using both the information of barycenters and of PCA, give an interpretation to each cluster

# Our analysis evidences the presence of three well differentiated clusters, each of which corresponds - in our interpretation-  
# to three different qualities of wines.
# The cluster are characterised by a good degree of variation across them, as we can infer from the distance 
# among the centroids of each cluster. 
# In addition, the clusters display a sufficient level of homogeneity within them, as shown by their low within-variance.
        
# ---------------------------------------------------------------------------------------------------

##################################  4. Optimal K-Means Silhouette  ###################################

def Optimal_k_means(dataset):
    
    from sklearn.metrics import silhouette_score

    max_clust = 10 # it can be modified by the user if we insert it as an input of the function
    
    my_pca = PCA(n_components=2) 
    Y = my_pca.fit_transform(X)
    kms = {}
    max = -1
    opt_k = 0
     
    for i in range(max_clust - 1):
        km = KMeans(n_clusters=(i+2), 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
        y_km = km.fit_predict(Y)
        kms[i+2] = km
        silhouette_avg = silhouette_score(Y, y_km)
        if silhouette_avg > max:
            max = silhouette_avg
            opt_k = i+2
    y_km = kms[opt_k].fit_predict(Y)
    col = ['yellow','green','brown','green','pink','blue','violet','orange','black','ray']
    for i in range(opt_k):
        plt.scatter(Y[y_km ==i , 0],
                    Y[y_km ==i , 1],
                    s=50, c=col[i],
                    marker = 'o', edgecolor='black',
                    label = 'cluster' + str(i+1))
    plt.scatter(kms[opt_k].cluster_centers_[:,0],
                kms[opt_k].cluster_centers_[:,1],
                s=250, marker='*',
                c='red', edgecolor='black',
                label = 'centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return opt_k

Optimal_k_means(X_s)

# ---------------------------------------------------------------------------------------------------

##################################  5. Circle of Correlation  ###################################

def circle_correlation(X):
    
    X.dropna(how = "all", inplace=True)
    
    #standardizing
    from sklearn.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X)

    my_pca = PCA()
    Y = my_pca.fit_transform(X_s)
    PCs = my_pca.components_

    for i in range(PCs.shape[1]):
        for l in range(PCs.shape[1]):
            if i==l:
                continue
            # Plotting
            else:
                fig = plt.figure(figsize=(3,3))
                plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
                   PCs[i,:], PCs[l,:], 
                   angles='xy', scale_units='xy', scale=1)
                # Labels based on feature names 
                feature_names = [X.columns[i] for i in range(PCs.shape[1])]
                for a,j,z in zip(PCs[l,:]+0.02, PCs[i,:]+0.02, feature_names):
                    plt.text(j, a, z, ha='center', va='center')

                # Unit circle
                circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)

                #axis
                plt.axis('equal')
                plt.xlim([-1.0,1.0])
                plt.ylim([-1.0,1.0])
                plt.xlabel('PC {0}'.format(i))
                plt.ylabel('PC {0}'.format(l))

                plt.show()


circle_correlation(X)

