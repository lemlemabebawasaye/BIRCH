# Import the Necessary Libraries
import numpy as np
import math

# Compute Cluster Statistics (N, LS, SS)
class CFNode:
    def __init__(self, point):
        self.N = 1
        self.LS = point
        self.SS = (point[0] ** 2, point[1] ** 2)

    def update(self, point):
        self.N += 1
        self.LS = (self.LS[0] + point[0], self.LS[1] + point[1])
        self.SS = (self.SS[0] + point[0] ** 2, self.SS[1] + point[1] ** 2)

# Define a Leaf Node for Clustering with Insertion Logic Based on a Threshold
class CFLeaf:
    def __init__(self, threshold):
        self.threshold = threshold
        self.points = []
        self.cluster_features = None
        self.radius = 0

    def insert(self, point):
        if not self.points:
            self.points.append(point)
            self.cluster_features = CFNode(point)
            return

        new_LS = self.cluster_features.LS[0] + point[0], self.cluster_features.LS[1] + point[1]
        new_SS = self.cluster_features.SS[0] + point[0] ** 2, self.cluster_features.SS[1] + point[1] ** 2
        new_N = self.cluster_features.N + 1
        new_radius = math.sqrt((new_SS[0] / new_N) - (new_LS[0] / new_N) ** 2), math.sqrt((new_SS[1] / new_N) - (new_LS[1] / new_N) ** 2)
        self.radius = new_radius

        if new_radius[0] > self.threshold or new_radius[1] > self.threshold:
            return False
        else:
            self.cluster_features.update(point)
            self.points.append(point)
            return True

    # Method to compute the centroid of this leaf
    def compute_centroid(self):
        if self.cluster_features.N > 0:
            return (self.cluster_features.LS[0] / self.cluster_features.N,
                    self.cluster_features.LS[1] / self.cluster_features.N)
        else:
            return None  # In case there are no points in the cluster

# Define Branch Node for BIRCH Clustering to Manage Child Nodes and Handle Point Insertions        
class CFBranch:
    def __init__(self, branching_factor, threshold):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.children = []

    def insert(self, point):
        for child in self.children:
            if isinstance(child, CFLeaf):
                if child.insert(point):
                    return True
            else:
                if child.insert(point):
                    return True
        new_leaf = CFLeaf(self.threshold)
        new_leaf.insert(point)
        self.children.append(new_leaf)
        if len(self.children) > self.branching_factor:
            new_branch = CFBranch(self.branching_factor, self.threshold)
            new_branch.children = self.children[self.branching_factor:]
            self.children = self.children[:self.branching_factor]
            self.children.append(new_branch)
        return True
    
# BIRCH Clustering Initialization: Building the CF Tree with Given Parameters
def birch_cluster(points, branching_factor, threshold):
    root = CFBranch(branching_factor, threshold)
    for point in points:
        root.insert(point)
    return root


# Global Clustering range by building a smaller CF tree
# Extracting Leaf Node Centroids from the CF Tree
def collect_leaf_centroids(node):
    centroids = []
    if isinstance(node, CFLeaf):
        centroid = node.compute_centroid()
        if centroid is not None:
            centroids.append(centroid)
    elif isinstance(node, CFBranch):
        for child in node.children:
            centroids.extend(collect_leaf_centroids(child))
    return centroids

# K-Means Algorithm
# Function to calculate distance
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def k_means(data, num_clusters, array_centroids, max_iterations):
  diff = 1
  j = 0

  while diff != 0 and j < max_iterations:
      clusters = []
      for point in data:
          distances = [distance(point, centroid) for centroid in array_centroids]
          cluster = distances.index(min(distances))
          clusters.append(cluster)

      centroids_new = np.zeros(array_centroids.shape)
      for i in range(num_clusters):
          points = [data[j] for j in range(len(data)) if clusters[j] == i]
          centroids_new[i] = np.mean(points, axis=0) if points else array_centroids[i]

      diff = np.linalg.norm(centroids_new - array_centroids)
      array_centroids = centroids_new
      j += 1
  return clusters