from time import time
import numpy as np

from load_datasets import *
from plot_results import *

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn import decomposition


random_state = 24
np.random.seed(random_state)


def train_neural_network(X, y, title, dataset):

    if 'abalone' in dataset:
        clf = MLPClassifier(solver='lbfgs', activation='relu', max_iter=140, hidden_layer_sizes=(50,))
    elif 'letters' in dataset:
        clf = MLPClassifier(solver='adam', activation='relu', max_iter=160, hidden_layer_sizes=(70,))

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=10, scoring='f1_macro')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel('Training Size')
    plt.ylabel('Score')

    # Save graph
    plt.grid()
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig('output/' + ('-' . join([dataset, 'lc'])), dpi=100, bbox_inches="tight")
    plt.close("all")

    t0 = time()
    clf.fit(X, y)
    t1 = time() - t0
    print t1
    return clf


def classify_neural_network(dataset, method, n_components, X, X_test, y, y_test, k_means_clusters=0, em_clusters=0):

    filename = ('-' . join([dataset, method, str(n_components)]))

    if method == 'pca':
        dr = decomposition.PCA(n_components=n_components, svd_solver='auto', random_state=random_state)
        title = dataset.title() + ': Neural Network (PCA)'
    elif method == 'ica':
        dr = decomposition.FastICA(n_components=n_components, random_state=random_state, whiten=True)
        title = dataset.title() + ': Neural Network (ICA)' 
    elif method == 'rp':
        dr = GaussianRandomProjection(n_components=n_components)
        title = dataset.title() + ': Neural Network (RP)'
    elif method == 'fa': 
        dr = decomposition.FactorAnalysis(n_components=n_components, svd_method='randomized', random_state=random_state)
        title = dataset.title() + ': Neural Network (FA)'

    X = dr.fit_transform(X)
    X_test_t = dr.transform(X_test)

    if k_means_clusters:
        title += ' (K-Means)'
        filename += '-km'
        estimator = KMeans(n_clusters=k_means_clusters, init='k-means++', n_init=10, random_state=random_state)
        estimator.fit(X)

        new_features = estimator.predict(X)
        X = np.insert(X, 0, new_features, axis=1)

        new_features = estimator.predict(X_test_t)
        X_test_t = np.insert(X_test_t, 0, new_features, axis=1)

    elif k_means_clusters:
        title += ' (Expectation-Maximization)'
        filename += '-em'
        estimator = GaussianMixture(n_components=k_means_clusters, init_params='kmeans', n_init=10, random_state=random_state, covariance_type='full', reg_covar=1e-2)
        estimator.fit(X)

        new_features = estimator.predict(X)
        X = np.insert(X, 0, new_features, axis=1)

        new_features = estimator.predict(X_test_t)
        X_test_t = np.insert(X_test_t, 0, new_features, axis=1)
    
    clf = train_neural_network(X, y.astype('int'), title, filename)

    y_pred = clf.predict(X_test_t)
    print f1_score(y_test.astype('int'), y_pred.astype('int'), average='macro')

    if not k_means_clusters and not em_clusters:
        X_test_t = dr.fit_transform(X_test)
        y_pred = clf.predict(X_test_t)
        print f1_score(y_test.astype('int'), y_pred.astype('int'), average='macro')


def main():

    #--- Abalone Dimension Reduction ---#
    X, y = load_dataset('abalone')
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    classify_neural_network('abalone', 'pca', 4, X, X_test, y, y_test)
    classify_neural_network('abalone', 'ica', 5, X, X_test, y, y_test)
    classify_neural_network('abalone', 'rp', 4, X, X_test, y, y_test)
    classify_neural_network('abalone', 'fa', 5, X, X_test, y, y_test)

    #--- Letters Dimension Reduction ---#
    X, y = load_dataset('letters')
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)
    classify_neural_network('letters', 'pca', 12, X, X_test, y, y_test)
    classify_neural_network('letters', 'ica', 11, X, X_test, y, y_test)
    classify_neural_network('letters', 'rp', 6, X, X_test, y, y_test)
    classify_neural_network('letters', 'fa', 11, X, X_test, y, y_test)

    #--- Abalone Clustering ---#
    X, y = load_dataset('abalone')
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    classify_neural_network('abalone', 'pca', 4, X, X_test, y, y_test, 7)
    classify_neural_network('abalone', 'ica', 5, X, X_test, y, y_test, 7)
    classify_neural_network('abalone', 'rp', 4, X, X_test, y, y_test, 7)
    classify_neural_network('abalone', 'fa', 5, X, X_test, y, y_test, 7)

    X, y = load_dataset('abalone')
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    classify_neural_network('abalone', 'pca', 4, X, X_test, y, y_test, 0, 9)
    classify_neural_network('abalone', 'ica', 5, X, X_test, y, y_test, 0, 9)
    classify_neural_network('abalone', 'rp', 4, X, X_test, y, y_test, 0, 9)
    classify_neural_network('abalone', 'fa', 5, X, X_test, y, y_test, 0, 9)

    
if __name__ == "__main__":
    main()
