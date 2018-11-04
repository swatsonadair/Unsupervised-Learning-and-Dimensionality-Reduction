from time import time
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans

from load_datasets import *
from plot_results import *


random_state = 24
np.random.seed(random_state)


def optimize_k_means(dataset, min_clusters, max_clusters):

    if '-' in dataset:
        X, y = load_reduced(dataset)
    else:
        X, y = load_dataset(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_clusters in range(min_clusters, max_clusters):
        print('n_clusters: ', n_clusters)
        for init in ['k-means++', 'random']:

            scores = []
            estimator = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=random_state)

            t0 = time()
            estimator.fit(data)

            scores.append(n_clusters)
            scores.append(init)
            scores.append(time() - t0)
            scores.append(estimator.inertia_)
            scores.append(metrics.homogeneity_score(labels, estimator.labels_))
            scores.append(metrics.completeness_score(labels, estimator.labels_))
            scores.append(metrics.v_measure_score(labels, estimator.labels_))
            scores.append(metrics.adjusted_rand_score(labels, estimator.labels_))
            scores.append(metrics.adjusted_mutual_info_score(labels,  estimator.labels_))
            scores.append(metrics.silhouette_score(data, estimator.labels_))

            results.append(scores)

    # N-Clusters vs Inertia per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[3], y_axis_label='Inertia', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'inertia']))

    # N-Clusters vs Homogeneity Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[4], y_axis_label='Homogeneity Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'homogeneity']))

    # N-Clusters vs Completeness Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[5], y_axis_label='Completeness Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'completeness']))

    # N-Clusters vs V-Measure Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[6], y_axis_label='V-Measure Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'vmeasure']))

    # N-Clusters vs Adjusted Random Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[7], y_axis_label='Adjusted Random Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'adjustedrand']))

    # N-Clusters vs Adjusted Mutual Information Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[8], y_axis_label='Adjusted Mutual Information Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'adjustedmi']))

    # N-Clusters vs Silhouette Score per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[9], y_axis_label='Silhouette Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'silhouette']))

    # N-Clusters vs Time per Init Centroid
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[2], y_axis_label='Time', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'time']))

    # N-Clusters vs V-Measures per Scoring Method w/o CV
    plot_results(np.array(results), constant_index=[1], constant_value=['k-means++'], x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[4, 5, 6], y_axis_label='Score', trend_labels=['Homogeneity', 'Completeness', 'V-Measure'], title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'kmeans', 'vmeasure']))

    # N-Clusters vs Score per Scoring Method w/o CV
    plot_results(np.array(results), constant_index=[1], constant_value=['k-means++'], x_axis_index=0, x_axis_label='K-Clusters', y_axis_index=[6, 8, 9], y_axis_label='Score', trend_labels=['V-Measure', 'Adjusted Mutual Information', 'Silhouette'], title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'kmeans', 'scores']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'kmeans.csv'])), results, delimiter=",", fmt="%s")


def k_means_original(dataset, n_clusters, init):
    X, y = load_dataset(dataset)
    data = X
    scores = run_k_means(dataset, data, y, n_clusters, init)
    return scores


def k_means_reduced(dataset, n_clusters, init):
    X, y = load_reduced(dataset)
    data = X
    scores = run_k_means(dataset, data, y, n_clusters, init)
    return scores


def run_k_means(dataset, data, labels, n_clusters, init):

    print("Running K-means: ", dataset)

    info = dataset.split('-')
    if len(info) == 3:
        scores = [info[2], int(info[1])]
    else:
        scores = ['none', len(np.unique(labels))]

    estimator = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=random_state)

    t0 = time()
    estimator.fit(data)

    scores.append(n_clusters)
    scores.append(time() - t0)
    scores.append(metrics.v_measure_score(labels, estimator.labels_))
    scores.append(metrics.adjusted_mutual_info_score(labels,  estimator.labels_))
    scores.append(metrics.silhouette_score(data, estimator.labels_))

    predictions = estimator.fit_predict(data)

    np.savetxt('data/' + ('-' . join([dataset, str(n_clusters), 'kmeans.csv'])), np.array(predictions).astype('int'), delimiter=",", fmt='%i')

    plot_cluster_scores(data, predictions, n_clusters, dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'clusters', 'silhoutte', str(n_clusters)]))

    # compare to the features
    feature_scores = []
    total_feature_values = float(len(np.unique(data, axis=None)) / 10.)
    for f in range(0, len(data[0])):
        #print float(len(np.unique(data[:,f])))
        feature_scores.append([metrics.adjusted_mutual_info_score(data[:,f],  estimator.labels_), metrics.adjusted_mutual_info_score(data[:,f],  estimator.labels_) * float(len(np.unique(data[:,f]))) / total_feature_values])

    plot_categorical(np.array(feature_scores), x_axis_label='Feature', y_axis_label='Adjusted Mutal Information Score', title=dataset.title() + ': K-means', filename='-' . join(['km', dataset, 'features', str(n_clusters)]))
    
    return scores


def main():

    results = []

    #--- UNALTERED DATA ---#
    optimize_k_means('abalone', 2, 29 * 2 + 1)
    optimize_k_means('letters', 2, 26 * 2 + 1)

    results.append(k_means_original('abalone', 7, 'k-means++'))
    results.append(k_means_original('letters', 2, 'k-means++'))

    #--- PCA ---#
    results.append(k_means_reduced('abalone-4-pca', 7, 'k-means++'))
    results.append(k_means_reduced('letters-12-pca', 2, 'k-means++'))

    #--- ICA ---#
    results.append(k_means_reduced('abalone-5-ica', 7, 'k-means++'))
    results.append(k_means_reduced('letters-11-ica', 2, 'k-means++'))

    #--- RP ---#
    for i in range(1, 10):
        results.append(k_means_reduced('abalone-4-' + str(i) + 'rp', 7, 'k-means++'))
        results.append(k_means_reduced('letters-6-' + str(i) + 'rp', 2, 'k-means++'))

    #--- FACTOR ---#
    results.append(k_means_reduced('abalone-5-fa', 7, 'k-means++'))
    results.append(k_means_reduced('letters-11-fa', 2, 'k-means++'))

    np.savetxt('output-csv/' + ('-' . join(['kmeans.csv'])), results, delimiter=",", fmt='%s')

    #--- GENERATE DIFFERENT CLUSTER NUMBERS ---#
    optimize_k_means('abalone-4-pca', 2, 29)
    optimize_k_means('letters-12-pca', 2, 26)

    optimize_k_means('abalone-5-ica', 2, 29)
    optimize_k_means('letters-11-ica', 2, 26)

    optimize_k_means('abalone-4-5rp', 2, 29)
    optimize_k_means('letters-6-4rp', 2, 26)

    optimize_k_means('abalone-5-fa', 2, 29)
    optimize_k_means('letters-11-fa', 2, 26)


if __name__ == "__main__":
    main()
