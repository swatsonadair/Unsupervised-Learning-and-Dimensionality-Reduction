from time import time
import numpy as np

from sklearn import metrics
from sklearn.mixture import GaussianMixture

from load_datasets import *
from plot_results import *


random_state = 24
np.random.seed(random_state)


def optimize_em(dataset, min_components, max_components):

    X, y = load_reduced(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_components in range(min_components, max_components):
        print('n_components: ', n_components)
        for init_params in ['kmeans', 'random']:

            for covariance_type in ['full', 'tied', 'diag', 'spherical']:

                scores = []
                estimator = GaussianMixture(n_components=n_components, init_params=init_params, n_init=10, random_state=random_state, covariance_type=covariance_type, reg_covar=1e-2)

                t0 = time()
                estimator.fit(data)

                scores.append(n_components)
                scores.append(init_params)
                scores.append(covariance_type)
                scores.append(time() - t0)
                scores.append(estimator.n_iter_)
                scores.append(estimator.aic(data))
                scores.append(estimator.bic(data))

                predictions = estimator.predict(data)
                scores.append(metrics.homogeneity_score(labels, predictions))
                scores.append(metrics.completeness_score(labels, predictions))
                scores.append(metrics.v_measure_score(labels, predictions))
                scores.append(metrics.adjusted_rand_score(labels, predictions))
                scores.append(metrics.adjusted_mutual_info_score(labels, predictions))
                scores.append(metrics.silhouette_score(data, predictions))

                #print estimator.converged_

                results.append(scores)

    # N-Components vs AIC per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[5], y_axis_label='Akaike Info Criterion', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'akaike']))

    # N-Components vs BIC per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[6], y_axis_label='Bayesian Info Criterion', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'bayesian']))

    # N-Components vs Homogeneity Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[7], y_axis_label='Homogeneity Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'homogeneity']))

    # N-Components vs Completeness Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[8], y_axis_label='Completeness Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'completeness']))

    # N-Components vs V-Measure Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[9], y_axis_label='V-Measure Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'vmeasure']))

    # N-Components vs Adjusted Random Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[10], y_axis_label='Adjusted Random Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'adjustedrand']))

    # N-Components vs Adjusted Mutual Information Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[11], y_axis_label='Adjusted Mutual Information Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'adjustedmi']))

    # N-Components vs Silhouette Score per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[12], y_axis_label='Silhouette Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'silhouette']))

    # N-Components vs Time per Init Centroid
    plot_results(np.array(results), constant_index=[2], constant_value=['full'], trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[3], y_axis_label='Time', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'time']))

    # N-Components vs Criterions per Scoring Method
    plot_results(np.array(results), constant_index=[1, 2], constant_value=['kmeans', 'full'], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[7, 8, 9], y_axis_label='Score', trend_labels=['Homogeneity', 'Completeness', 'V-Measure'], title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'kmeans', 'vmeasure']))

    # N-Components vs V-Measures per Scoring Method
    plot_results(np.array(results), constant_index=[1, 2], constant_value=['kmeans', 'full'], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[5, 6], y_axis_label='Score', trend_labels=['Akaike', 'Bayesian'], title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'kmeans', 'criterion']))

    # N-Components vs Score per Scoring Method
    plot_results(np.array(results), constant_index=[1, 2], constant_value=['kmeans', 'full'], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[9, 11, 12], y_axis_label='Score', trend_labels=['V-Measure', 'Adjusted Mutual Information', 'Silhouette'], title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'kmeans', 'scores']))

    # N-Components vs AIC per Covariance
    plot_results(np.array(results), constant_index=[1], constant_value=['kmeans'], trends_index=2, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[5], y_axis_label='Akaike Info Criterion', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'kmeans', 'aic']))

    # N-Components vs BIC per Covariance
    plot_results(np.array(results), constant_index=[1], constant_value=['kmeans'], trends_index=2, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[6], y_axis_label='Bayesian Info Criterion', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'kmeans', 'bic']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'expmax.csv'])), results, delimiter=",", fmt="%s")



def em_original(dataset, n_clusters, init):
    X, y = load_dataset(dataset)
    data = X
    scores = run_em(dataset, data, y, n_clusters, init)
    return scores


def em_reduced(dataset, n_clusters, init):
    X, y = load_reduced(dataset)
    data = X
    scores = run_em(dataset, data, y, n_clusters, init)
    return scores


def run_em(dataset, data, labels, n_components, init_params):

    print("Running Expectation-Maximization: ", dataset)

    info = dataset.split('-')
    if len(info) == 3:
        scores = [info[2], int(info[1])]
    else:
        scores = ['none', len(np.unique(labels))]

    estimator = GaussianMixture(n_components=n_components, init_params=init_params, n_init=10, random_state=random_state, covariance_type='full', reg_covar=1e-2)

    t0 = time()
    estimator.fit(data)

    scores.append(n_components)
    scores.append(time() - t0)

    scores.append(estimator.aic(data))
    scores.append(estimator.bic(data))

    predictions = estimator.predict(data)
    scores.append(metrics.v_measure_score(labels, predictions))
    scores.append(metrics.adjusted_mutual_info_score(labels, predictions))
    scores.append(metrics.silhouette_score(data, predictions))

    np.savetxt('data/' + ('-' . join([dataset, str(n_components), 'expmax.csv'])), np.array(predictions).astype('int'), delimiter=",", fmt='%i')

    plot_cluster_scores(data, predictions, n_components, dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'components', 'silhoutte', str(n_components)]))

    # compare to the features
    feature_scores = []
    total_feature_values = float(len(np.unique(data, axis=None)) / 10.)
    for f in range(0, len(data[0])):
        #print float(len(np.unique(data[:,f])))
        feature_scores.append([metrics.adjusted_mutual_info_score(data[:,f],  predictions), metrics.adjusted_mutual_info_score(data[:,f], predictions) * float(len(np.unique(data[:,f]))) / total_feature_values])

    plot_categorical(np.array(feature_scores), x_axis_label='Feature', y_axis_label='Adjusted Mutal Information Score', title=dataset.title() + ': Expectation-Maximization', filename='-' . join(['em', dataset, 'features', str(n_components)]))

    return scores


def main():


    results = []

    #--- UNALTERED DATA ---#
    optimize_em('abalone', 2, 29 * 2 + 1)
    optimize_em('letters', 2, 26 * 2 + 1)

    results.append(em_original('abalone', 9, 'kmeans'))
    results.append(em_original('letters', 2, 'kmeans'))

    #--- PCA ---#
    results.append(em_reduced('abalone-4-pca', 9, 'kmeans'))
    results.append(em_reduced('letters-12-pca', 2, 'kmeans'))

    #--- ICA ---#
    results.append(em_reduced('abalone-5-ica', 9, 'kmeans'))
    results.append(em_reduced('letters-11-ica', 2, 'kmeans'))

    #--- RP ---#
    for i in range(1, 10):
        results.append(em_reduced('abalone-4-' + str(i) + 'rp', 9, 'kmeans'))
        results.append(em_reduced('letters-6-' + str(i) + 'rp', 2, 'kmeans'))

    #--- FACTOR ---#
    results.append(em_reduced('abalone-5-fa', 9, 'kmeans'))
    results.append(em_reduced('letters-11-fa', 2, 'kmeans'))

    np.savetxt('output-csv/' + ('-' . join(['expmax.csv'])), results, delimiter=",", fmt='%s')

    #--- GENERATE DIFFERENT CLUSTER NUMBERS ---#
    optimize_em('abalone-4-pca', 2, 29)
    optimize_em('letters-12-pca', 2, 26)

    optimize_em('abalone-5-ica', 2, 29)
    optimize_em('letters-11-ica', 2, 26)

    optimize_em('abalone-4-5rp', 2, 29)
    optimize_em('letters-6-4rp', 2, 26)

    optimize_em('abalone-5-fa', 2, 29)
    optimize_em('letters-11-fa', 2, 26)


if __name__ == "__main__":
    main()
