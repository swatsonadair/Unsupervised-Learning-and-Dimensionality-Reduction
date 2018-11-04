from time import time
import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn import metrics

from load_datasets import *
from plot_results import *

random_state = 24
np.random.seed(random_state)


def run_fa(dataset, min_components, max_components):

    X, y = load_dataset(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_components in range(min_components, max_components):
        print('n_components: ', n_components)

        for svd_method in ['lapack', 'randomized']:

            scores = []
            data = X.copy()
            fa = FactorAnalysis(n_components=n_components, svd_method=svd_method, random_state=random_state)

            t0 = time()
            fa.fit(X)

            scores.append(n_components)
            scores.append(svd_method)
            scores.append(time() - t0)
            scores.append(fa.score(X))
            
            results.append(scores)

    # N-Components vs Log Likelihood
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[3], y_axis_label='Log Liklihood', title=dataset.title() + ': FactorAnalysis', filename='-' . join(['fa', dataset, 'loglike']))

    # N-Components vs Time
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[2], y_axis_label='Time', title=dataset.title() + ': FactorAnalysis', filename='-' . join(['fa', dataset, 'time']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'fa.csv'])), results, delimiter=",", fmt="%s")


def save_new_data(dataset, n_components, svd_method):
    X, y = load_dataset(dataset)
    data = X
    fa = decomposition.FactorAnalysis(n_components=n_components, svd_method=svd_method, random_state=random_state)
    fa.fit(data)

    matrix = fa.components_
    new_data = fa.transform(data)

    plot_data('fa', new_data, y, dataset.title() + ': FactorAnalysis', filename='-' . join(['fa', dataset, 'data', 'trans']))

    results = np.array(new_data)
    np.savetxt('data/' + ('-' . join([dataset, str(n_components), 'fa.csv'])), results, delimiter=",")

    new_data_inv = np.dot(new_data, matrix)
    loss = metrics.mean_squared_error(data, new_data_inv)
    print loss


def main():
    run_fa('abalone', 1, 11)
    run_fa('letters', 1, 17)

    save_new_data('abalone', 5, 'randomized')
    save_new_data('letters', 11, 'randomized')
    

if __name__ == "__main__":
    main()
