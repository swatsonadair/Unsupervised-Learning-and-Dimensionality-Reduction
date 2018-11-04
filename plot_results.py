import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import decomposition
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_samples, silhouette_score


valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)


def plot_results(results, trends_index=None, constant_index=None, constant_value=None, x_axis_index=None, x_axis_label=None, y_axis_index=None, y_axis_label=None, trend_labels=None, title=None, filename=None):
    
    COLORS = ['.b-', '.g-', '.m-', '.y-']

    x_axis = np.sort(np.unique(results[:,x_axis_index]).astype('int'))
    
    if constant_index:
        for i, constant in enumerate(constant_index):
            results = results[(results[:,constant] == constant_value[i])]

    if trends_index:
        trends = np.unique(results[:,trends_index])
        c = 0
        for index, trend in enumerate(trends):
            data = results[(results[:,trends_index] == trend)][:,y_axis_index]
            if len(trends) > 4:
                i = np.arange(len(data))
                plt.bar(i + 0.1 * (index + 1), data[:,0].astype('float'), 0.1, label=trend.astype('int'))
            else:
                plt.plot(x_axis, data[:,0].astype('float'), COLORS[c%4], label=trend)
            c += 1
    else:
        c = 0
        for index, trend in enumerate(y_axis_index):
            data = results[:,trend]
            plt.plot(x_axis, data.astype('float'), COLORS[c%4], label=trend_labels[index])
            c += 1

    # X-axis
    plt.xlabel(x_axis_label)

    # Y-axis
    plt.ylabel(y_axis_label)
    
    # Save graph
    plt.grid()
    plt.title(title)
    plt.legend(loc="best", borderaxespad=0.)
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


def plot_data(method, X, y, title, filename):

    fig, (ax1) = plt.subplots(1, 1)

    n_labels = len(y)

    if method == 'pca':
        t = decomposition.PCA(n_components=2)
        X = t.fit_transform(X)
    elif method == 'ica':
        t = decomposition.FastICA(n_components=2, whiten=True)
        X = t.fit_transform(X)
    elif method == 'rp':
        t = GaussianRandomProjection(n_components=2)
        X = t.fit_transform(X)

    np.random.seed(20)
    for label in np.unique(y):
        ax1.scatter(X[y == label, 0], X[y == label, 1], color=np.random.rand(3), linewidths=1)

    ax1.set_title(title)
    ax1.grid()
    plt.tight_layout()

    plt.savefig('/' . join(['output', filename]))
    plt.close("all")
    

def plot_cluster_scores(X, cluster_labels, n_clusters, title, filename):

    fig, (ax1) = plt.subplots(1, 1)
    
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title(title)
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster Label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Set the axis ticks
    ax1.set_yticks([]) 

    plt.savefig('/' . join(['output', filename]))
    plt.close("all")


def plot_categorical(results, x_axis_label=None, y_axis_label=None, title=None, filename=None):
    i = np.arange(len(results[:,0]))
    plt.bar(i + 0.25, results[:,0], 0.25, label='score', color='b')
    plt.bar(i + 0.25 * 2, results[:,1], 0.25, label='normalized', color='g')
            
    # X-axis
    plt.xlabel(x_axis_label)
    plt.xticks([]) 

    # Y-axis
    plt.ylabel(y_axis_label)
    
    # Save graph
    plt.grid()
    plt.title(title)
    plt.legend(loc="best", borderaxespad=0.)
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


