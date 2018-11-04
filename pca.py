from time import time
import numpy as np

from sklearn import decomposition
from sklearn import metrics

from load_datasets import *
from plot_results import *

random_state = 24
np.random.seed(random_state)


def run_pca(dataset, min_components, max_components):

    X, y = load_dataset(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_components in range(min_components, max_components):
        print('n_components: ', n_components)

        for svd_solver in ['auto', 'full', 'randomized']:

            scores = []
            data = X.copy()
            pca = decomposition.PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)

            t0 = time()
            pca.fit(X)

            scores.append(n_components)
            scores.append(svd_solver)
            scores.append(time() - t0)
            scores.append(pca.score(data))

            scores.append(pca.explained_variance_ratio_[n_components - 1])
            scores.append(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)[n_components - 1])
            
            results.append(scores)


    # N-Components vs Score
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[3], y_axis_label='Score', title=dataset.title() + ': PCA', filename='-' . join(['pca', dataset, 'score']))

    # N-Components vs Variance
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[4], y_axis_label='Variance Ratio', title=dataset.title() + ': PCA', filename='-' . join(['pca', dataset, 'variance']))

    # N-Components vs Variance %
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[5], y_axis_label='% Variance', title=dataset.title() + ': PCA', filename='-' . join(['pca', dataset, 'pvar']))

    # N-Components vs Time
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[2], y_axis_label='Time', title=dataset.title() + ': PCA', filename='-' . join(['pca', dataset, 'time']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'pca.csv'])), results, delimiter=",", fmt="%s")


def save_new_data(dataset, n_components, svd_solver):
    X, y = load_dataset(dataset)
    data = X
    pca = decomposition.PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    pca.fit(data)

    matrix = pca.components_
    new_data = pca.transform(data)

    plot_data('pca', new_data, y, dataset.title() + ': PCA', filename='-' . join(['pca', dataset, 'data', 'trans']))

    results = np.array(new_data)
    np.savetxt('data/' + ('-' . join([dataset, str(n_components), 'pca.csv'])), results, delimiter=",")

    new_data_inv = pca.inverse_transform(new_data)
    loss = metrics.mean_squared_error(data, new_data_inv)
    print loss


def main():
    run_pca('abalone', 1, 11)
    run_pca('letters', 1, 17)

    save_new_data('abalone', 4, 'auto')
    save_new_data('letters', 12, 'auto')
    

if __name__ == "__main__":
    main()
