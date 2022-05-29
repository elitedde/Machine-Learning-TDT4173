import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
 
class KMeans:
     
    def __init__(self):
        
        self.clusters = [] 
        self.vett_k = [] #range of k
        self.final_k = 0 #selected k that performs well
        self.labels = [] 
        pass
    
    def recalculate_clusters(self, X, init_centroids, k):
        
        #given the centroids, clusters are recalculated
        clusters = {}
        self.labels = np.zeros(X.shape[0])
        for i in range(k):
            clusters[i]=[]
     
        for i in range(X.shape[0]): 
            dist = [] 
            for j in range(k):
                dist.append(euclidean_distance(X.loc[i], init_centroids[j]))
            clusters[dist.index(min(dist))].append(X.loc[i,:]) 
            self.labels[i] = dist.index(min(dist)) 
        return clusters
 
    def difference(self, prev, next, k):
        
        #methods used to verify the difference between new centroids and previous ones
        diff = 0
        for i in range(k):
            diff += np.linalg.norm(prev[i]-next[i])
        return diff
    
    def fit(self, X):
        
        self.vett_k = list([2,10])
 
     
    def predict(self, X):
        
        max = -1.0
        actual_k = 0
        actual_clusters = []
        actual_label = np.zeros(X.shape[0])
        for k in self.vett_k:
            for n in range(4):
                centroids = self.random_centroid(X,k)
                diff = 100
                while(diff>0.0001):
                    self.clusters = self.recalculate_clusters(X, centroids, k)
                    new_centroids = self.get_centroids(k)
                    diff = self.difference(centroids, new_centroids, k)
                    centroids = new_centroids
                value = euclidean_silhouette(X, np.array(self.labels, dtype=np.int32))
                if(value > max):
                    max = value
                    actual_label = self.labels
                    actual_k = k
                    actual_clusters = self.clusters
        self.clusters = actual_clusters
        self.final_k = actual_k
        return np.array(actual_label, dtype=np.int32)
        raise NotImplementedError()
     
    def random_centroid(self,X,k):
        #compute new random centroids that are taken from the dataset
        random_idx=[np.random.randint(X.shape[0]) for i in range(k)]
        centroids = np.zeros((k, X.shape[1]))
        cont=0
        for i in random_idx:
            centroids[cont] = X.loc[i,:]
            cont+=1
        return centroids
     
    def get_centroids(self, k):
        centroids = {}
        for i in range(k):
            centroids[i]= np.average(self.clusters[i], axis = 0)
        data=list(centroids.values())
        return np.array(data)
        raise NotImplementedError()
         
    def getK(self):
        return self.final_k
     
     
     
# --- Some utility functions
 
def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
     
    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments
     
    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
     
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
         
    return distortion
 
def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points
     
    Note: by passing "y=0.0", it will compute the euclidean norm
     
    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points
             
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)
 
def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points
     
    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.
             
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])
 
def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance
     
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257 
        - https://en.wikipedia.org/wiki/Silhouette_(clustering) 
     
    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments
     
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
     
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
     
    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
     
    return np.mean((b - a) / np.maximum(a, b))