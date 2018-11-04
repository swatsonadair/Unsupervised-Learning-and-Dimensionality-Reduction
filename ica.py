from time import time
import numpy as np

from scipy.stats import kurtosis
from sklearn import decomposition
from sklearn import metrics

from load_datasets import *
from plot_results import *

random_state = 24
np.random.seed(random_state)


def run_ica(dataset, min_components, max_components):

    X, y = load_dataset(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_components in range(min_components, max_components):
        print('n_components: ', n_components)

        data = X.copy()
        ica = decomposition.FastICA(n_components=n_components, random_state=random_state, whiten=True)
        
        t0 = time()
        ica.fit(X)

        elapsed = time() - t0
        kurtoses = np.sort(kurtosis(ica.components_))

        kurt_vsgauss_vals = []

        for component_index in range(min_components, max_components):

            scores = []
            scores.append(n_components)
            scores.append(component_index)
            scores.append(time() - t0)
            scores.append(ica.n_iter_)

            sources = ica.transform(X)

            if component_index <= n_components:
                kurt = kurtoses[component_index - 1]
                kurt_vsgauss = abs(3 - kurt)
            else:
                kurt = 0
                kurt_vsgauss = 0

            kurt_vsgauss_vals.append(kurt_vsgauss)
            kurt_norm = sum(kurt_vsgauss_vals) / max_components

            scores.append(kurt)
            scores.append(kurt_vsgauss)
            scores.append(kurt_norm)
            scores.append(kurtosis(ica.components_, axis=None))
            scores.append(kurtosis(sources, axis=None))

            results.append(scores)

    # N-Components vs Kurtosis (Component)
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='Dimensions', y_axis_index=[4], y_axis_label='Kurtosis', title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'kurt']))

    # N-Components vs Kurtosis (vs Gaussian)
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='Dimensions', y_axis_index=[5], y_axis_label='Kurtosis (vs Gaussian)', title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'kurt', 'vsgauss']))

    # N-Components vs Kurtosis (vs Gaussian)
    plot_results(np.array(results), constant_index=[1], constant_value=[max_components - 1], x_axis_index=0, x_axis_label='Dimensions', y_axis_index=[6], y_axis_label='Kurtosis (%)', trend_labels=[''], title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'kurt', 'norm']))

    # N-Components vs Overall Kurtosis (Component)
    plot_results(np.array(results), constant_index=[1], constant_value=[1], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[7], y_axis_label='Overall Kurtosis (Component)', trend_labels=[''], title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'kurt', 'overall', 'comp']))

    # N-Components vs Overall Kurtosis (Source)
    plot_results(np.array(results), constant_index=[1], constant_value=[1], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[8], y_axis_label='Overall Kurtosis (Source)', trend_labels=[''], title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'kurt', 'overall', 'source']))

    # N-Components vs Iter
    plot_results(np.array(results), constant_index=[1], constant_value=[1], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[3], y_axis_label='Iterations', trend_labels=[''], title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'iter']))

    # N-Components vs Time
    plot_results(np.array(results), constant_index=[1], constant_value=[1], x_axis_index=0, x_axis_label='K-Components', y_axis_index=[2], y_axis_label='Time', trend_labels=[''], title=dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'time']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'ica.csv'])), results, delimiter=",", fmt="%s")


def save_new_data(dataset, n_components):
    X, y = load_dataset(dataset)
    data = X
    ica = decomposition.FastICA(n_components=n_components, random_state=random_state, whiten=True)
    ica.fit(data)
    new_data = ica.transform(data)

    plot_data('ica', new_data, y, dataset.title() + ': ICA', filename='-' . join(['ica', dataset, 'data', 'trans']))

    results = np.array(new_data)
    np.savetxt('data/' + ('-' . join([dataset, str(n_components), 'ica.csv'])), results, delimiter=",")

    new_data_inv = ica.inverse_transform(new_data)
    loss = metrics.mean_squared_error(data, new_data_inv)
    print loss


def main():
    run_ica('abalone', 1, 11)
    run_ica('letters', 1, 17)

    save_new_data('abalone', 5)
    save_new_data('letters', 11)
    

if __name__ == "__main__":
    main()
