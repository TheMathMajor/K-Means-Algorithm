import numpy as np
def compute_d2_slow(data, centers, D):
    for i in range(data.shape[0]):
        for j in range(centers.shape[0]):
            dist = 0
            for k in range(centers.shape[1]):
                dist += (centers[j,k] - data[i,k])**2
            D[i,j] = dist


def compute_d2_fast(data, centers, D):
    A = -2 * np.dot(data, centers.transpose())
    x = (data*data).sum(axis = 1)
    #vector of dot products of all data points 1-by-n
    X = np.array([x] * centers.shape[0]).transpose()
    #expand into matrix and transpose n-by-k
    c = (centers*centers).sum(axis = 1)
    #vector of dot products of all centers
    C = np.array([c] * D.shape[0])
    #expand into matrix n-by-k
    B = X + A + C
    D[:,:] = B[:,:]


def kmeans(data, k, alg, tol=0.00001, maxiter=100):
	# data -- n*d data matrix (each row is a data point)
	# this matrix should be in single precision in order to consume less memory

	# k -- numbers of clusters to be found

	# alg:
	# 'slow' -- using BLAS-1
	# 'fast' -- using BLAS-3

	# tol -- If the absolute difference between the cost functions of two successive iterations is less than tol, stop.
	# maxiter -- Maximum number of iterations

	# Return values:
	# labels -- a vector of cluster assignments
	# it -- number of iterations
	# centers -- cluster centroids, a k*d matrix
	# min_dists -- n-vector containing squared Euclidean distances between each data point and its closest centroid

	n, d = data.shape
	sampling = np.random.randint(0, n, k)
	centers = data[sampling, :]

	old_e = float('inf')
	old_centers = np.zeros((k, d), dtype=np.float32)
	sizes = np.zeros(k, dtype=np.uint32)

	D = np.zeros((n, k), dtype=np.float32)
	
	for it in xrange(maxiter):
		old_centers[:] = centers

		if alg == 'fast':
			compute_d2_fast(data, centers, D)
		else: # alg == 'slow'
			compute_d2_slow(data, centers, D)

		labels = np.nanargmin(D, axis=1)
		min_dists = np.nanmin(D, axis=1)
		min_dists[min_dists < 0.0] = 0.0

		centers[:, :] = 0.0
		sizes[:] = 0
		for i in xrange(n):
			assignment = labels[i]
			sizes[assignment] += 1
			centers[assignment, :] += data[i, :]

		for j in xrange(k):
			if sizes[j] > 0:
				centers[j, :] /= sizes[j]
			else:
				centers[j, :] = np.nan

		e = float(np.sqrt(np.sum(min_dists) / n))
		print 'Iteration:', it, ', Error:', e
		if it > 0 and abs(e - old_e) <= tol:
			break
		old_e = e

	return labels, it, centers, min_dists
