from kmeansCluster import kMeansCluster
from sklearn.datasets.samples_generator import make_blobs

X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)

kMeans = kMeansCluster(k = 4, max_iterations = 100)
hypotheses, centroids = kMeans.fit(X)

print('centroids')
print(centroids)
print('cluster_assignments')
print(cluster_assignments)
print('hypothese labels')
print(hypotheses)

