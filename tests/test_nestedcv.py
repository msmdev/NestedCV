__author__ = __maintainer__ = "Bernhard Reuter"
__email__ = "bernhard-reuter@gmx.de"
__copyright__ = "Copyright 2022, Bernhard Reuter"


import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pandas.util.testing import assert_frame_equal
from pandas import DataFrame
from sklearn.svm import SVC
from tests.conftest import dummy_classifier, assert_allclose
import nestedcv as ncv
import re
from typing import Dict, Any


def test_RGSCV(
    X: np.ndarray,
    y: np.ndarray,
    RGSCV_results: Dict[str, Any],
) -> None:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        train_size=0.8,
        random_state=0,
        shuffle=True,
        stratify=y,
    )

    gs = ncv.RepeatedGridSearchCV(
        estimator=dummy_classifier(),
        param_grid={'alpha': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]},
        scoring='mcc',
        cv=5,
        n_jobs=None,
        Nexp=5,
        save_to=None,
        reproducible=True,
    )
    gs.fit(X_train, y_train)
    assert gs.opt_scores_ == {'mcc': 1.0}
    assert gs.opt_scores_idcs_ == {'mcc': 10}
    assert gs.opt_params_ == {'mcc': {'alpha': 1.0}}
    assert_frame_equal(DataFrame(data=gs.cv_results_), RGSCV_results)


def test_RGSCV_parallel(
    X: np.ndarray,
    y: np.ndarray,
    RGSCV_results: Dict[str, Any],
) -> None:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        train_size=0.8,
        random_state=0,
        shuffle=True,
        stratify=y
    )

    gs = ncv.RepeatedGridSearchCV(
        estimator=dummy_classifier(),
        param_grid={'alpha': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]},
        scoring='mcc',
        cv=5,
        n_jobs=5,
        Nexp=5,
        save_to=None,
        reproducible=True
    )
    gs.fit(X_train, y_train)
    assert gs.opt_scores_ == {'mcc': 1.0}
    assert gs.opt_scores_idcs_ == {'mcc': 10}
    assert gs.opt_params_ == {'mcc': {'alpha': 1.0}}
    assert_frame_equal(DataFrame(data=gs.cv_results_), RGSCV_results)


def test_RNCV_single(
    X: np.ndarray,
    y: np.ndarray,
    brier_loss: float,
    log_loss: float,
    score: float,
    threshold: float,
) -> None:
    estimator = dummy_classifier()

    param_grid = {'alpha': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}

    cv_options = {
        'scoring': ['average_precision',
                    'balanced_accuracy',
                    'brier_loss',
                    'f1',
                    'f2',
                    'log_loss',
                    'mcc',
                    'precision_recall_auc',
                    'pseudo_f1',
                    'pseudo_f2',
                    'roc_auc',
                    'sensitivity'],
        'refit': 'roc_auc',
        'tune_threshold': True,
        'threshold_tuning_scoring': ['f1',
                                     'balanced_accuracy',
                                     'f1',
                                     'f1',
                                     'f2',
                                     'f1',
                                     None,
                                     'f1',
                                     'pseudo_f1',
                                     'pseudo_f2',
                                     'J',
                                     'pseudo_f2'],
        'Nexp2': 1,
        'Nexp1': 1,
        'inner_cv': 5,
        'outer_cv': 5,
        'n_jobs': None,
        'reproducible': True
    }

    clf_grid = ncv.RepeatedStratifiedNestedCV(
        estimator=estimator,
        param_grid=param_grid,
        cv_options=cv_options
    )
    clf_grid.fit(X, y)

    all_results = clf_grid.repeated_nested_cv_results_

    length = cv_options['Nexp2'] * cv_options['outer_cv']
    assert list(all_results.keys()) == cv_options['scoring']
    for scoring, threshold_tuning_scoring in zip(
            cv_options['scoring'], cv_options['threshold_tuning_scoring']):
        results = all_results[scoring]
        for key in results.keys():
            if key == f'best_{threshold_tuning_scoring}':
                assert threshold_tuning_scoring is not None
                assert len(results[key]) == length
                assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_inner_indices':
                assert len(results[key]) == length
                assert np.all(results[key] == 10)
            elif key == 'best_inner_params':
                assert len(results[key]) == length
                assert all(x == {'alpha': 1.0} for x in results[key])
            elif key == 'best_inner_scores':
                assert len(results[key]) == length
                if bool(re.search(r"log_loss", scoring)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.search(r"brier_loss", scoring)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_params':
                assert results[key] == {'alpha': [1.0]}
            elif key == 'best_threshold':
                assert threshold_tuning_scoring is not None
                assert f'best_{threshold_tuning_scoring}' in results.keys()
                assert len(results[key]) == length
                assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"max_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"max_(test|train|ncv)_log_loss", key)):
                    assert 0.98 * log_loss <= results[key] < 1.2 * log_loss
                elif bool(re.match(r"max_(test|train|ncv)_brier_loss", key)):
                    assert 0.98 * brier_loss <= results[key] < 1.2 * brier_loss
                elif key == 'max_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif bool(re.fullmatch(r"mean_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"mean_(test|train|ncv)_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"mean_(test|train|ncv)_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                elif key == 'mean_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], 1.0, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"min_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"min_(test|train|ncv)_log_loss", key)):
                    assert 1.02 * log_loss >= results[key] > 0.8 * log_loss
                elif bool(re.match(r"min_(test|train|ncv)_brier_loss", key)):
                    assert 1.02 * brier_loss >= results[key] > 0.8 * brier_loss
                elif key == 'min_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"ncv_[a-zA-Z0-9_]+", key)):
                assert results[key].size == cv_options['Nexp2']
                if bool(re.match(r"ncv_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"ncv_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"std_[a-zA-Z0-9_]+", key)):
                assert results[key] < 0.05
            elif key == 'ranked_best_inner_params':
                assert results[key] == [{'frequency': length, 'parameters': {'alpha': 1.0}}]
            elif bool(re.fullmatch(r"test_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"test_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.2, atol=0.0)
                elif bool(re.match(r"test_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.2, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"train_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"train_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"train_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            else:
                raise ValueError("Unexpected key: %s" % key)


def test_RNCV_repeated(
    X: np.ndarray,
    y: np.ndarray,
    brier_loss: float,
    log_loss: float,
    score: float,
    threshold: float,
) -> None:
    estimator = dummy_classifier()

    param_grid = {'alpha': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}

    cv_options = {
        'scoring': ['average_precision',
                    'balanced_accuracy',
                    'brier_loss',
                    'f1',
                    'f2',
                    'log_loss',
                    'mcc',
                    'precision_recall_auc',
                    'pseudo_f1',
                    'pseudo_f2',
                    'roc_auc',
                    'sensitivity'],
        'refit': 'roc_auc',
        'tune_threshold': True,
        'threshold_tuning_scoring': ['f1',
                                     'balanced_accuracy',
                                     'f1',
                                     'f1',
                                     'f2',
                                     'f1',
                                     None,
                                     'f1',
                                     'pseudo_f1',
                                     'pseudo_f2',
                                     'J',
                                     'pseudo_f2'],
        'Nexp2': 5,
        'Nexp1': 5,
        'inner_cv': 5,
        'outer_cv': 5,
        'n_jobs': None,
        'reproducible': True
    }

    clf_grid = ncv.RepeatedStratifiedNestedCV(
        estimator=estimator,
        param_grid=param_grid,
        cv_options=cv_options
    )
    clf_grid.fit(X, y)

    all_results = clf_grid.repeated_nested_cv_results_

    length = cv_options['Nexp2'] * cv_options['outer_cv']
    assert list(all_results.keys()) == cv_options['scoring']
    for scoring, threshold_tuning_scoring in zip(
            cv_options['scoring'], cv_options['threshold_tuning_scoring']):
        results = all_results[scoring]
        for key in results.keys():
            if key == f'best_{threshold_tuning_scoring}':
                assert threshold_tuning_scoring is not None
                assert len(results[key]) == length
                assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_inner_indices':
                assert len(results[key]) == length
                assert np.all(results[key] == 10)
            elif key == 'best_inner_params':
                assert len(results[key]) == length
                assert all(x == {'alpha': 1.0} for x in results[key])
            elif key == 'best_inner_scores':
                assert len(results[key]) == length
                if bool(re.search(r"log_loss", scoring)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.search(r"brier_loss", scoring)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_params':
                assert results[key] == {'alpha': [1.0]}
            elif key == 'best_threshold':
                assert threshold_tuning_scoring is not None
                assert f'best_{threshold_tuning_scoring}' in results.keys()
                assert len(results[key]) == length
                assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"max_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"max_(test|train|ncv)_log_loss", key)):
                    assert 0.98 * log_loss <= results[key] < 1.2 * log_loss
                elif bool(re.match(r"max_(test|train|ncv)_brier_loss", key)):
                    assert 0.98 * brier_loss <= results[key] < 1.2 * brier_loss
                elif key == 'max_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif bool(re.fullmatch(r"mean_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"mean_(test|train|ncv)_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"mean_(test|train|ncv)_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                elif key == 'mean_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], 1.0, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"min_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"min_(test|train|ncv)_log_loss", key)):
                    assert 1.02 * log_loss >= results[key] > 0.8 * log_loss
                elif bool(re.match(r"min_(test|train|ncv)_brier_loss", key)):
                    assert 1.02 * brier_loss >= results[key] > 0.8 * brier_loss
                elif key == 'min_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"ncv_[a-zA-Z0-9_]+", key)):
                assert results[key].size == cv_options['Nexp2']
                if bool(re.match(r"ncv_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"ncv_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"std_[a-zA-Z0-9_]+", key)):
                assert results[key] < 0.05
            elif key == 'ranked_best_inner_params':
                assert results[key] == [{'frequency': length, 'parameters': {'alpha': 1.0}}]
            elif bool(re.fullmatch(r"test_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"test_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.2, atol=0.0)
                elif bool(re.match(r"test_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.2, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"train_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"train_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"train_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            else:
                raise ValueError("Unexpected key: %s" % key)


def test_RNCV_repeated_parallel(
    X: np.ndarray,
    y: np.ndarray,
    brier_loss: float,
    log_loss: float,
    score: float,
    threshold: float,
) -> None:
    estimator = dummy_classifier()

    param_grid = {'alpha': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}

    cv_options = {
        'scoring': ['average_precision',
                    'balanced_accuracy',
                    'brier_loss',
                    'f1',
                    'f2',
                    'log_loss',
                    'mcc',
                    'precision_recall_auc',
                    'pseudo_f1',
                    'pseudo_f2',
                    'roc_auc',
                    'sensitivity'],
        'refit': 'roc_auc',
        'tune_threshold': True,
        'threshold_tuning_scoring': ['f1',
                                     'balanced_accuracy',
                                     'f1',
                                     'f1',
                                     'f2',
                                     'f1',
                                     None,
                                     'f1',
                                     'pseudo_f1',
                                     'pseudo_f2',
                                     'J',
                                     'pseudo_f2'],
        'Nexp2': 2,
        'Nexp1': 2,
        'inner_cv': 5,
        'outer_cv': 5,
        'n_jobs': 2,
        'reproducible': True
    }

    clf_grid = ncv.RepeatedStratifiedNestedCV(
        estimator=estimator,
        param_grid=param_grid,
        cv_options=cv_options
    )
    clf_grid.fit(X, y)

    all_results = clf_grid.repeated_nested_cv_results_

    length = cv_options['Nexp2'] * cv_options['outer_cv']
    assert list(all_results.keys()) == cv_options['scoring']
    for scoring, threshold_tuning_scoring in zip(
            cv_options['scoring'], cv_options['threshold_tuning_scoring']):
        results = all_results[scoring]
        for key in results.keys():
            if key == f'best_{threshold_tuning_scoring}':
                assert threshold_tuning_scoring is not None
                assert len(results[key]) == length
                assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_inner_indices':
                assert len(results[key]) == length
                assert np.all(results[key] == 10)
            elif key == 'best_inner_params':
                assert len(results[key]) == length
                assert all(x == {'alpha': 1.0} for x in results[key])
            elif key == 'best_inner_scores':
                assert len(results[key]) == length
                if bool(re.search(r"log_loss", scoring)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.search(r"brier_loss", scoring)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif key == 'best_params':
                assert results[key] == {'alpha': [1.0]}
            elif key == 'best_threshold':
                assert threshold_tuning_scoring is not None
                assert f'best_{threshold_tuning_scoring}' in results.keys()
                assert len(results[key]) == length
                assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"max_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"max_(test|train|ncv)_log_loss", key)):
                    assert 0.98 * log_loss <= results[key] < 1.2 * log_loss
                elif bool(re.match(r"max_(test|train|ncv)_brier_loss", key)):
                    assert 0.98 * brier_loss <= results[key] < 1.2 * brier_loss
                elif key == 'max_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.01, atol=0.0)
            elif bool(re.fullmatch(r"mean_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"mean_(test|train|ncv)_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"mean_(test|train|ncv)_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                elif key == 'mean_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], 1.0, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"min_[a-zA-Z0-9_]+", key)):
                if bool(re.match(r"min_(test|train|ncv)_log_loss", key)):
                    assert 1.02 * log_loss >= results[key] > 0.8 * log_loss
                elif bool(re.match(r"min_(test|train|ncv)_brier_loss", key)):
                    assert 1.02 * brier_loss >= results[key] > 0.8 * brier_loss
                elif key == 'min_best_threshold':
                    assert_allclose(results[key], threshold, rtol=0.02, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"ncv_[a-zA-Z0-9_]+", key)):
                assert results[key].size == cv_options['Nexp2']
                if bool(re.match(r"ncv_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"ncv_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            elif bool(re.fullmatch(r"std_[a-zA-Z0-9_]+", key)):
                assert results[key] < 0.05
            elif key == 'ranked_best_inner_params':
                assert results[key] == [{'frequency': length, 'parameters': {'alpha': 1.0}}]
            elif bool(re.fullmatch(r"test_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"test_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.2, atol=0.0)
                elif bool(re.match(r"test_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.2, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.03, atol=0.0)
            elif bool(re.fullmatch(r"train_[a-zA-Z0-9_]+", key)):
                assert results[key].size == length
                if bool(re.match(r"train_log_loss", key)):
                    assert_allclose(results[key], log_loss, rtol=0.1, atol=0.0)
                elif bool(re.match(r"train_brier_loss", key)):
                    assert_allclose(results[key], brier_loss, rtol=0.1, atol=0.0)
                else:
                    assert_allclose(results[key], score, rtol=0.02, atol=0.0)
            else:
                raise ValueError("Unexpected key: %s" % key)


def test_RNCV_precomputed_kernel() -> None:
    # Test that grid search works when the input features are given in the
    # form of a precomputed kernel matrix
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # compute the training kernel matrix corresponding to the linear kernel
    K_train = np.dot(X_[:180], X_[:180].T)
    y_train = y_[:180]

    clf = SVC(kernel='precomputed', probability=True)
    cv = ncv.RepeatedStratifiedNestedCV(
        estimator=clf,
        param_grid={'C': [0.1, 1.0]},
        cv_options={'refit': True},
    )
    cv.fit(K_train, y_train)

    assert cv.repeated_nested_cv_results_[
        'precision_recall_auc']['max_ncv_precision_recall_auc'] >= 0
    print(cv.repeated_nested_cv_results_['precision_recall_auc']['max_ncv_precision_recall_auc'])

    # compute the test kernel matrix
    K_test = np.dot(X_[180:], X_[:180].T)
    y_test = y_[180:]

    y_pred = cv.predict(K_test)

    assert np.mean(y_pred == y_test) >= 0
    print(np.mean(y_pred == y_test))


def test_grid_search_precomputed_kernel_error_nonsquare() -> None:
    # Test that grid search returns an error with a non-square precomputed
    # training kernel matrix
    K_train = np.zeros((10, 20))
    y_train = np.ones((10, ))
    clf = SVC(kernel='precomputed', probability=True)
    cv = ncv.RepeatedStratifiedNestedCV(
        estimator=clf,
        param_grid={'C': [0.1, 1.0]},
    )
    with pytest.raises(ValueError):
        cv.fit(K_train, y_train)
