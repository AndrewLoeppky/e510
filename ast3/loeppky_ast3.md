---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Assignment 3

Andrew Loeppky

EOSC 510

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from minisom import MiniSom
```

## Problem 1 

You are given two timeseries $x_1$ and $x_2$ (`data.mat` or `data.csv`), each one containing 310 points in time. For each
timeseries, you'd like to investigate how the characteristic temporal patterns change in time, so you decide to
use the singular spectrum analysis (applied on each timeseries separately) with a total lag of $L=50$ days.

Tasks:
1) Plot the timeseries and create the lagged matrix for each of the timeseries. Show (in symbolic matrix form)
how your lagged matrix looks like for $x_1$.

$$
X =
\left[ {\begin{array}{cc}
    x_1(0) & x_1(1) & x_1(2) & ... & x_1(N-L+1)\\
    x_1(1) & x_1(2) & x_1(3) & ... & x_1(N-L+2) \\
    \vdots & \vdots & \vdots & \ddots & \vdots\\
    x_1(L) & x_1(L+1) & x_1(L+2) & ... & X_1(N)
\end{array} } \right]
$$

```{code-cell} ipython3
# get the data and plot it
data_in = pd.read_csv("data.csv")

fig, ax = plt.subplots(figsize=(15,3))
ax.plot(data_in["t"], data_in["x1"], label="$x_1$")
ax.plot(data_in["t"], data_in["x2"], label="$x_2$")
ax.set_ylabel("t")
ax.set_title("Plot of Raw data x1 and x2")
ax.legend();
```

```{code-cell} ipython3
# create the lagged matrix for each variable
L = 50
n_obs = np.shape(data_in["x1"])[0] - L + 1
SSA_input_x1 = np.empty((n_obs, L))
SSA_input_x2 = np.empty((n_obs, L))

for kk in range(n_obs):
    SSA_input_x1[kk,:] = data_in["x1"][kk:kk+L] 
    SSA_input_x2[kk,:] = data_in["x2"][kk:kk+L]
```

2) Perform SSA. Plot the eigenvectors and PCs of the most important modes (decide yourself how many modes
are important) for $x_1$ and $x_2$. Hint: in SSA we are interested in the pairs of modes.

```{code-cell} ipython3
#do SSA (PCA on the lagged matrix)

# x1
n_modes_x1 = np.min(np.shape(SSA_input_x1))
pca_x1 = PCA(n_components = n_modes_x1)
PCs_x1 = pca_x1.fit_transform(SSA_input_x1)
eigvecs_x1 = pca_x1.components_
fracVar_x1 = pca_x1.explained_variance_ratio_

# x2
n_modes_x2 = np.min(np.shape(SSA_input_x2))
pca_x2 = PCA(n_components = n_modes_x2)
PCs_x2 = pca_x2.fit_transform(SSA_input_x2)
eigvecs_x2 = pca_x2.components_
fracVar_x2 = pca_x2.explained_variance_ratio_
```

```{code-cell} ipython3
def plot_pcs(fracVar, title):
    """
    Plots the fractional variance contained within each pc
    
    stolen from Sam Anderson's SSA tutorial
    """
    plt.figure(figsize=(10,3))

    plt.subplot(1,2,1)
    plt.scatter(range(len(fracVar)),fracVar,s = 75, edgecolor = 'k')
    plt.xlabel('Mode Number')
    plt.ylabel('Fraction Variance')
    #plt.title('Variance Explained by All Modes', fontsize = 24)
    plt.title(title, fontsize = 24)

    plt.subplot(1,2,2)
    n_modes_show = 10
    plt.scatter(range(n_modes_show),fracVar[:n_modes_show],s = 100, edgecolor = 'k')
    plt.xlabel('Mode Number', fontsize = 12)
    plt.ylabel('Fraction Variance', fontsize = 12)
    
    

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
plot_pcs(fracVar_x1, "$x_1$")
plot_pcs(fracVar_x2, "$x_2$")
```

Based on the above plots, I want to keep the first 4 modes for $x_1$ and the first 6 for $x_2$

```{code-cell} ipython3
#plot the first n modes and PCs -- choose a value of 'n' from the variance explained figure!
def plot_PCs(PCs, eigvecs, n, title):
    """
    plots each PC and eigenvector for n modes (decide how many to keep)
    """
    plt.figure(figsize=(20,5*n))
    plt.suptitle(title, fontsize=40)
    for kk in range(n):

        plt.subplot(n,2,kk*2+1)
        plt.plot(eigvecs[kk])
        plt.ylim((np.min(eigvecs),np.max(eigvecs)))
        plt.xlabel('Day', fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.title('Eigenvector of Mode #'+str(kk+1), fontsize = 24)

        plt.subplot(n,2,(kk+1)*2)
        plt.plot(PCs[:,kk])
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.title('PCs of Mode #' + str(kk+1), fontsize = 24)
        plt.xlabel('Day', fontsize = 20)

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
plot_PCs(PCs_x1, eigvecs_x1, 4, "=" * 20 + "  $x_1$  " + "=" * 20)  # for x1
plot_PCs(PCs_x2, eigvecs_x2, 6, "=" * 20 + "  $x_2$  " + "=" * 20)  # for x2
```

3) How much variance is carried by the dominant signals (signals of different frequencies) in $x_1$ and how much
in $x_2$? Note that in SSA, a signal of given frequency is usually captured by two modes.
> for $x_1$, the frist two modes carry 74\% of the variance

>for $x_2$, the first two modes carry 57\% of the variance

```{code-cell} ipython3
# first two modes of x1
print(sum(fracVar_x1[:2]))
print(sum(fracVar_x2[:2]))
```

## Problem 2

You are given a data (data_problem2.mat or data_problem2.csv) that contains one year of normalized daily
streamflow from 194 rivers in Alberta, Canada (i.e. there are 194 stations, each with 365 days of normalized
streamflow). The locations of each station are given by a latitude/longitude coordinate pair in stationLon.mat
and stationLat.mat (or stationLon.csv and stationLat.csv). ABlon.csv and ABlat.csv give coordinates of the
Alberta border for plotting (e.g. plt.plot(lon,lat) or figure; plot(lon, lat) will plot the border).
Following the guidelines below, perform two types of clustering to investigate how to cluster these stations
across the region on the basis of similarity in their streamflow regimes.

*Note: apply PCA on the data first (m=365, n=194) and then perform clustering (hierarchical
clustering and SOM) on the first few modes only. Most likely the first 3 modes will be enough to
keep. In the final plots, make sure that you reconstruct the data (streamflow) from the clustered
PC modes, as was done in the Tutorial example on SST dataset.*

```{code-cell} ipython3
# get the data
sflow = pd.read_csv("data_problem2.csv")#, header=None)
lats = np.array(pd.read_csv("ABlat.csv", header=None)).flatten()
lons = np.array(pd.read_csv("ABlon.csv", header=None)).flatten()

stationlats = np.array(pd.read_csv("stationLat.csv", header=None)).flatten()
stationlons = np.array(pd.read_csv("stationLon.csv", header=None)).flatten()

# show the stations
fig, ax = plt.subplots()
ax.scatter(stationlons, stationlats, marker="x")
ax.plot(lons, lats, color="k")
ax.set_aspect("equal")
```

Tasks:
1) Perform the hierarchical clustering (if using Matlab, use Ward's method) on the data. Plot the dendrogram.

```{code-cell} ipython3
# do PCA
n_modes = np.min(np.shape(sflow))
pca = PCA(n_components = n_modes)
PCs = pca.fit_transform(sflow)
eigvecs = pca.components_
fracVar = pca.explained_variance_ratio_

# define the inverse transform 
pca2 = PCA(n_components=2)
PCs2 = pca2.fit_transform(sflow)

# how many modes to keep?
plot_pcs(fracVar, " ")
```

```{code-cell} ipython3
# plot the eigvecs and PCs
plot_PCs(PCs, eigvecs, 2, " ")
```

```{code-cell} ipython3
# plot PC-space -- can you see any clusters?

plt.scatter(PCs[:,0],PCs[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC Space');
```

```{code-cell} ipython3
data = PCs[:,:2]
linked = linkage(data,'ward')
#labelList = range(1, len(frequency))

plt.figure(figsize=(10, 7))
dendrogram(linked, 
           orientation='top', 
           distance_sort='descending',
           truncate_mode='lastp',
           p=30)
plt.title('Dendrogram')
plt.show()
```

2) Choose two possible options for the optimal number of clusters (k) from the dendrogram and provide some
explanation on why you chose those k.


For each choice of k:

**a)** plot the mean streamflow pattern of each cluster


**b)** plot the clusters on the map of Alberta (lat/lon scatter-plot), i.e. color each station's location (can use a filled
circle as a marker) according to the cluster to which it belongs. Discuss what you think are two key differences between your results for two different choices of k

*by inspection, it appears that there are 2 somewhat distinct clusters, possibly 3*

<img src="possible_clusters1.png" width="500">
<img src="possible_clusters2.png" width="500">

```{code-cell} ipython3
######################
##### 2 clusters #####
######################

#now cluster
n_clusters = 2
cluster = AgglomerativeClustering(n_clusters=n_clusters, 
                                  affinity='euclidean', 
                                  linkage='ward')
cn = cluster.fit_predict(data)

#find mean pattern of each cluster
cluster_pattern = np.empty((n_clusters,np.shape(sflow)[1]))
cluster_pattern_PC = np.empty((n_clusters,np.shape(data)[1]))

for cluster_num in range(n_clusters):
    inds = np.argwhere(cn==cluster_num)
    cluster_pattern_PC[cluster_num,:] = np.mean(data[inds,:],axis=0)
    cluster_pattern[cluster_num,:] = np.mean(sflow.iloc[np.squeeze(inds),:],axis=0)
    
    
## get the reconstructed data (not right yet)

# 0th cluster
#pattern1 = cluster_pattern[0][0] * eigvecs[0] + cluster_pattern[0][1] * eigvecs[1]

# 1th cluster
#pattern2 = cluster_pattern[1][0] * eigvecs[0] + cluster_pattern[1][1] * eigvecs[1]
```

```{code-cell} ipython3
# plot the mean stramflow pattern of each cluster
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.scatter(PCs[:,0],PCs[:,1],c=cn)
plt.scatter(cluster_pattern_PC[:,0],cluster_pattern_PC[:,1],marker='*',c='red',s=200)
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.title('Clusters in PC-Space')

plt.subplot(122)
plt.plot(cluster_pattern.T)
# plt.plot(sflow_rec[:n_clusters].T)
plt.xlabel('Time')
plt.ylabel('Normalized Power')
plt.title('Streamflow Patterns')
plt.legend(['Cluster ' + str(ii) for ii in range(n_clusters)])

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# show the stations
fig, ax = plt.subplots()

for cluster_num in range(n_clusters):
    inds = np.argwhere(cn==cluster_num)
    #cluster_pattern_PC[cluster_num,:] = np.mean(data[inds,:],axis=0)
    #cluster_pattern[cluster_num,:] = np.mean(data[inds,:],axis=0)
    ax.scatter(stationlons[inds], stationlats[inds])

ax.plot(lons, lats, color="k")
ax.set_aspect("equal")
ax.set_title(f"Streamflow Stations in Alberta, {n_clusters} Clusters")
ax.set_xlabel("lon")
ax.set_ylabel("lat")
```

```{code-cell} ipython3
######################
##### 3 clusters #####
######################

#now cluster
n_clusters = 3
cluster = AgglomerativeClustering(n_clusters=n_clusters, 
                                  affinity='euclidean', 
                                  linkage='ward')
cn = cluster.fit_predict(data)

#find mean pattern of each cluster
cluster_pattern = np.empty((n_clusters,np.shape(sflow)[1]))
cluster_pattern_PC = np.empty((n_clusters,np.shape(data)[1]))

for cluster_num in range(n_clusters):
    inds = np.argwhere(cn==cluster_num)
    cluster_pattern_PC[cluster_num,:] = np.mean(data[inds,:],axis=0)
    cluster_pattern[cluster_num,:] = np.mean(sflow.iloc[np.squeeze(inds),:],axis=0)
    
    
# plot the mean stramflow pattern of each cluster
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.scatter(PCs[:,0],PCs[:,1],c=cn)
plt.scatter(cluster_pattern_PC[:,0],cluster_pattern_PC[:,1],marker='*',c='red',s=200)
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.title('Clusters in PC-Space')

plt.subplot(122)
plt.plot(cluster_pattern.T)
# plt.plot(sflow_rec[:n_clusters].T)
plt.xlabel('Time')
plt.ylabel('Normalized Power')
plt.title('Streamflow Patterns')
plt.legend(['Cluster ' + str(ii) for ii in range(n_clusters)])

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# show the stations
fig, ax = plt.subplots()

for cluster_num in range(n_clusters):
    
    inds = np.argwhere(cn==cluster_num)
    #cluster_pattern_PC[cluster_num,:] = np.mean(data[inds,:],axis=0)
    #cluster_pattern[cluster_num,:] = np.mean(data[inds,:],axis=0)
    ax.scatter(stationlons[inds], stationlats[inds])

ax.plot(lons, lats, color="k")
ax.set_aspect("equal")
ax.set_title(f"Streamflow Stations in Alberta, {n_clusters} Clusters")
ax.set_xlabel("lon")
ax.set_ylabel("lat");
```

**interpretation**

With 2 clusters, we identify 2 patterns, one where streamflow shows a strong spike in the spring and another where the peak is later and flatter. 

With 3 clusters, we identify an intermediate mode where both spring runoff and the longer-period summer signal are apparent.

+++

## Problem 3

Perform clustering using a 3 x 2 SOM, and plot the resulting streamflow patterns as a 3 x 2 SOM. Plot the
locations of the stations, coloured according to the cluster (BMU) to which they belong. What is the frequency of
each cluster?

```{code-cell} ipython3
# re-import the data
P = data.T

# normalize it
P_mean = np.mean(data)
P_std = np.std(data)
P -= P_mean
P /= P_std

P.shape
```

```{code-cell} ipython3
#run SOM -- this code creates/trains the SOM and calculates stats of interest
nx = 3
ny = 2

#make, initialize, and train the SOM
data = P#.get_values()
som = MiniSom(nx, ny, data.shape[1], sigma=1, learning_rate=0.5) # initialization of (ny x nx) SOM
som.pca_weights_init(data)
som.train_random(data, 500) # trains the SOM 

qnt = som.quantization(data) #this is the pattern of the BMU of each observation (ie: has same size as data input to SOM)
bmu_patterns = som.get_weights() #this is the pattern of each BMU; size = (nx, ny, len(data[0]))
QE = som.quantization_error(data) #quantization error of map
TE = som.topographic_error(data) #topographic error of map

#calculate the BMU of each observation
bmus = []
for kk in range(len(data)):
    bmus.append(som.winner(data[kk]))
    
#inds gives the sequential coordinates of each SOM node (useful for plotting)
inds = []
for ii in range(ny):
    for jj in range(nx):
        inds.append((ii,jj))

        
# compute the frequency of each BMU
freq = np.zeros((nx,ny))
for bmu in bmus:
    freq[bmu[0]][bmu[1]]+=1
freq/=len(data)
```

```{code-cell} ipython3
# show the stations
fig, ax = plt.subplots()

# shameful code
for i, bmu in enumerate(bmus):
    if str(bmu) == "(0, 0)":        
        ax.scatter(stationlons[i], stationlats[i], color="blue", alpha=0.7)
    elif str(bmu) == "(0, 1)":        
        ax.scatter(stationlons[i], stationlats[i], color="red", alpha=0.7)
    elif str(bmu) == "(1, 0)":        
        ax.scatter(stationlons[i], stationlats[i], color="green", alpha=0.7)
    elif str(bmu) == "(1, 1)":        
        ax.scatter(stationlons[i], stationlats[i], color="orange", alpha=0.7)
    elif str(bmu) == "(2, 0)":        
        ax.scatter(stationlons[i], stationlats[i], color="purple", alpha=0.7)
    elif str(bmu) == "(2, 1)":        
        ax.scatter(stationlons[i], stationlats[i], color="yellow", alpha=0.7)
    
ax.plot(lons, lats, color="k")
ax.set_aspect("equal")
ax.set_title(f"Streamflow Stations in Alberta, {nx} x {ny} SOM")
ax.set_xlabel("lon")
ax.set_ylabel("lat");
```

```{code-cell} ipython3
# what is the frequency of each BMU?
print(freq)
```

## Problem 4

Perform clustering using 2 x 2 SOM, and plot the SOM patterns, locations of stations coloured by BMU, and
frequency of each cluster as in the case with 3 x 2 SOM. Discuss what you think are two key differences
between your results from 3 x 2 SOM and this SOM. 

```{code-cell} ipython3

```

```{code-cell} ipython3
# re-import the data
P = data.T

# normalize it
P_mean = np.mean(data)
P_std = np.std(data)
P -= P_mean
P /= P_std

P.shape
```

```{code-cell} ipython3
#run SOM -- this code creates/trains the SOM and calculates stats of interest
nx = 2
ny = 2

#make, initialize, and train the SOM
data = P#.get_values()
som = MiniSom(nx, ny, data.shape[1], sigma=1, learning_rate=0.5) # initialization of (ny x nx) SOM
som.pca_weights_init(data)
som.train_random(data, 500) # trains the SOM 

qnt = som.quantization(data) #this is the pattern of the BMU of each observation (ie: has same size as data input to SOM)
bmu_patterns = som.get_weights() #this is the pattern of each BMU; size = (nx, ny, len(data[0]))
QE = som.quantization_error(data) #quantization error of map
TE = som.topographic_error(data) #topographic error of map

#calculate the BMU of each observation
bmus = []
for kk in range(len(data)):
    bmus.append(som.winner(data[kk]))
    
#inds gives the sequential coordinates of each SOM node (useful for plotting)
inds = []
for ii in range(ny):
    for jj in range(nx):
        inds.append((ii,jj))

        
# compute the frequency of each BMU
freq = np.zeros((nx,ny))
for bmu in bmus:
    freq[bmu[0]][bmu[1]]+=1
freq/=len(data)
```

```{code-cell} ipython3
# show the stations
fig, ax = plt.subplots()

# shameful code
for i, bmu in enumerate(bmus):
    if str(bmu) == "(0, 0)":        
        ax.scatter(stationlons[i], stationlats[i], color="blue", alpha=0.7)
    elif str(bmu) == "(0, 1)":        
        ax.scatter(stationlons[i], stationlats[i], color="red", alpha=0.7)
    elif str(bmu) == "(1, 0)":        
        ax.scatter(stationlons[i], stationlats[i], color="green", alpha=0.7)
    elif str(bmu) == "(1, 1)":        
        ax.scatter(stationlons[i], stationlats[i], color="orange", alpha=0.7)
    
ax.plot(lons, lats, color="k")
ax.set_aspect("equal")
ax.set_title(f"Streamflow Stations in Alberta, {nx} x {ny} SOM")
ax.set_xlabel("lon")
ax.set_ylabel("lat");
```

```{code-cell} ipython3
print(freq)
```

```{code-cell} ipython3

```
