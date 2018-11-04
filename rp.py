from time import time
import numpy as np

from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn import metrics

from load_datasets import *
from plot_results import *


def run_rp(dataset, min_components, max_components):

    X, y = load_dataset(dataset)
    data = X

    n_samples, n_features = data.shape
    n_labels = len(np.unique(y))
    labels = y

    results = []

    for n_components in range(min_components, max_components):
        print('n_components: ', n_components)

        for max_iters in [10, 40, 100, 500]:

            scores = []
            times = []
            components = []
            kurtoses = []
            losses = []

            for iters in range(0, max_iters):

                data = X.copy()
                rp = GaussianRandomProjection(n_components=n_components)

                t0 = time()
                rp.fit(X)
                times.append(time() - t0)
                components.append(rp.n_components_)
                kurtoses.append(kurtosis(rp.components_, axis=None))

                matrix = rp.components_
                new_data = rp.transform(data)
                new_data_inv = np.dot(new_data, matrix)
                loss = metrics.mean_squared_error(data, new_data_inv)
                losses.append(loss)

            scores.append(n_components)
            scores.append(max_iters)
            scores.append(np.mean(np.array(times)))
            scores.append(np.mean(np.array(components)))
            scores.append(np.mean(np.array(kurtoses)))
            scores.append(np.std(np.array(kurtoses)))
            scores.append(np.mean(np.array(losses)))
            scores.append(np.std(np.array(losses)))

            results.append(scores)


    # N-Components vs Loss
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[6], y_axis_label='Reconstruction Error', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'loss']))

    # N-Components vs Loss (STD)
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[7], y_axis_label='Reconstruction Error (STD)', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'lossstd']))

    # N-Components vs Kurtosis
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[4], y_axis_label='Kurtosis', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'kurtosis']))

    # N-Components vs Kurtosis (STD)
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[5], y_axis_label='Kurtosis (STD)', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'kurtstd']))

    # N-Components vs Components
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[3], y_axis_label='Components', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'comp']))

    # N-Components vs Time
    plot_results(np.array(results), trends_index=1, x_axis_index=0, x_axis_label='K-Components', y_axis_index=[2], y_axis_label='Time', title=dataset.title() + ': RP', filename='-' . join(['rp', dataset, 'time']))

    results = np.array(results)
    np.savetxt('output-csv/' + ('-' . join([dataset, 'rp.csv'])), results, delimiter=",", fmt="%s")


def save_new_data(dataset, n_components, iteration):
    X, y = load_dataset(dataset)
    data = X
    rp = GaussianRandomProjection(n_components=n_components)
    rp.fit(data)

    matrix = rp.components_
    new_data = rp.transform(data)

    plot_data('rp', new_data, y, dataset.title() + ': RP', filename='-' . join(['rp', dataset, str(iteration), 'data', 'trans']))

    results = np.array(new_data)
    np.savetxt('data/' + ('-' . join([dataset, str(n_components), str(iteration) + 'rp.csv'])), results, delimiter=",")

    new_data_inv = np.dot(new_data, matrix)
    loss = metrics.mean_squared_error(data, new_data_inv)
    print loss


def main():
    run_rp('abalone', 1, 11)
    run_rp('letters', 1, 17)

    iteration = 10
    save_new_data('abalone', 4, iteration)
    save_new_data('letters', 6, iteration)
    

if __name__ == "__main__":
    main()
