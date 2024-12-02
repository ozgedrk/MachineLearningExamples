from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X, _ = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.6, random_state = 42)

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.title("Ornek Veri")




