from cluster import cluster
import numpy as np

class kMeansCluster(cluster):

    def __init__(self, k = 2, max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations

    """
    Return Eclud distance between two points.
    p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
    """
    def distance(self, p1, p2):
        tmp = np.sum((p1 - p2) ** 2)
        return np.sqrt(tmp)

    # Generate k center within the range of data set. #
    def rand_center(self, Xin):
        n = Xin.shape[1] # features
        centroids = np.zeros((self.k, n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(Xin[:,i]), np.max(Xin[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(self.k)
        return centroids
    
    # if centroids not changed, we say 'converged'
    def converged(self, centroids1, centroids2):
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return (set1 == set2)    
    
    def fit(self, Xin):
        n = Xin.shape[0]
        centroids = self.rand_center(Xin)
        label = np.zeros(n, dtype=np.int)
        
        iter = 0
        converged = False      
        
        while not converged and iter < self.max_iterations:
            old_centroids = np.copy(centroids)

            for i in range(n):
                dist_min, index_min = np.inf, -1

                # index of closest centroid to x #
                for j in range(self.k):
                    dist = self.distance(Xin[i], centroids[j])
                    if dist < dist_min:
                        dist_min, index_min = dist, j
                        label[i] = j
                
            for m in range(self.k):
                centroids[m] = np.mean(Xin[label == m], axis=0)

            converged = self.converged(old_centroids, centroids)
            iter += 1
            
        return label, centroids

if __name__ == '__main__':
    test = np.array([ [0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10] ])    

    cluster = kMeansCluster()
    label, centroids = cluster.fit(test)

    print(label)
    print(centroids)



    