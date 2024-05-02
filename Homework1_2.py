import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import datasets


def initialize_centroids(x, k):
    indices = np.random.choice(len(x), k, replace=False)
    return X[indices]


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def closest_centroid(x, centroids):
    distances = [distance(x, centroid) for centroid in centroids]
    return np.argmin(distances)


def update_centroids(X, clusters , k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[clusters == i]
        centroids[i] = np.mean(cluster_points, axis=0)
    return centroids


def k_means(X, k, max_iterations=1000):
    centroids = initialize_centroids(X, k)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1])
    centroid_points = ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red')
    figure_title = ax.text(0.5, 1.05, '', ha='center', va='bottom', transform=ax.transAxes)

    def update(frame):
        nonlocal centroids
        nonlocal scatter
        nonlocal centroid_points

        clusters = np.array([closest_centroid(x, centroids) for x in X])
        new_centroids = update_centroids(X, clusters, k)

        scatter.remove()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters)

        centroid_points.remove()
        centroid_points = ax.scatter(new_centroids[:, 0], new_centroids[:, 1], marker='x', c='red')

        centroids = new_centroids

        figure_title.set_text('Step {}'.format(frame + 1))

        return scatter, centroid_points, figure_title

    ani = FuncAnimation(fig, update, frames=max_iterations, blit=True, interval=500)

    plt.show()
    return centroids, clusters


iris = datasets.load_iris()
X = iris.data[:, :2]
k = 3

centroids, clusters = k_means(X, k)
