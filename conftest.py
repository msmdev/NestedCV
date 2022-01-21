import pytest
import numpy as np
from numpy.testing import assert_allclose as assert_allclose_np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import make_scorer, matthews_corrcoef
from pandas import DataFrame


def assert_allclose(actual, desired, rtol=1.0e-5, atol=1.0e-8, err_msg="", verbose=True):
    return assert_allclose_np(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg, verbose=True)


class dummy_classifier(ClassifierMixin, BaseEstimator):

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
            An array of float.
        y : array-like, shape (n_samples)
            The target values. An array of int {0,1}.
        Returns
        -------
        self : object
            Returns self fitting the estimator.
        """
        if isinstance(X, DataFrame):
            X = X.to_numpy()

        X, y = check_X_y(X, y)

        self.is_fitted_ = True
        self.n_classes_ = 2
        self.classes_ = np.array([0, 1])

        return self

    def predict_proba(self, X):
        """
        Predicts y from X using the previously fitted model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        p : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check, if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        X = np.array(X, dtype=np.float64)

        m, n = X.shape

        # result = self.predict(X)
        # result = np.sum(X * self.alpha, axis=1) / n
        result = np.sum(X, axis=1) / n

        idx = np.nonzero(result < 0.5)[0]
        fp_idx = np.random.choice(
            idx, size=int((1.0 - self.alpha) / 2.0 * idx.size), replace=False, p=None
        )
        idx = np.nonzero(result >= 0.5)[0]
        fn_idx = np.random.choice(
            idx, size=int((1.0 - self.alpha) / 2.0 * idx.size), replace=False, p=None
        )
        result[fp_idx] = np.random.uniform(low=0.5, high=1.0, size=fp_idx.size)
        result[fn_idx] = np.random.uniform(low=0.0, high=0.5, size=fn_idx.size)

        p = np.vstack((1 - result, result)).T

        if not p.shape[0] == m:
            raise ValueError("Input and output must match in the first dimension, "
                             f"but X.shape={X.shape} while p.shape={p.shape}.")

        return p

    def predict(self, X):
        """
        Predicts y from X using the previously fitted model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        # Check, if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        X = np.array(X, dtype=np.float64)

        # _, n = X.shape

        # result = np.sum(X * self.alpha, axis=1) / n
        result = self.predict_proba(X)[:, 1]

        if not result.size == X.shape[0]:
            raise ValueError("Input and output must match in the first dimension, "
                             f"but X.shape={X.shape} while y.shape={y.shape}.")

        return (result >= 0.5).astype(np.float64)

    def score(self, X, y, metric='MCC'):
        """
        Return the score version of the  Matthews correlation coefficient (higher is better)
        on the given test data and labels. Other metrics can be used.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        metric : optional. Metric function, if left empty the MCC score will be used.
        Returns
        -------
        score : float
            score of``self.predict(X)`` wrt. `y` using the given scoring method.
            Or score of MCC.
        """
        if isinstance(X, DataFrame):
            X = X.to_numpy()

        if metric == 'MCC':
            mcc = make_scorer(matthews_corrcoef)
            score_value = mcc(self, X, y)
        else:
            score_value = metric(y, self.predict(X))

        return score_value


@pytest.fixture(scope="session")
def X():
    n = 1000
    return np.array(
        [[1.0] * i + [0.0] * (n - i) for i in range(n + 1)],
        dtype=np.float64,
    )


@pytest.fixture(scope="session")
def y():
    n = 1000
    return (np.array(
        [(1.0 * i + 0.0 * (n - i)) / n for i in range(n + 1)]
    ) >= 0.5).astype(np.float64)


@pytest.fixture(scope="session")
def brier_loss():
    return 0.08325


@pytest.fixture(scope="session")
def log_loss():
    return 0.30655


@pytest.fixture(scope="session")
def score():
    return 1.0


@pytest.fixture(scope="session")
def threshold():
    return 0.5


@pytest.fixture(scope="session")
def RGSCV_results():
    results = DataFrame(data={
        'params': [
            {'alpha': 0.0},
            {'alpha': 0.1},
            {'alpha': 0.2},
            {'alpha': 0.3},
            {'alpha': 0.4},
            {'alpha': 0.5},
            {'alpha': 0.6},
            {'alpha': 0.7},
            {'alpha': 0.8},
            {'alpha': 0.9},
            {'alpha': 1.0}
        ],
        'alpha': np.ma.MaskedArray(
            data=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            mask=[False, False, False, False, False, False, False, False, False, False, False],
            fill_value='?',
            dtype=object
        ),
        'iteration0_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'iteration1_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'iteration2_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'iteration3_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'iteration4_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'mean_mcc': np.array(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.825, 0.925, 1.]
        ),
        'std_mcc': np.array(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        )
    })
    return results
