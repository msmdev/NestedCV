__author__ = __maintainer__ = "Bernhard Reuter"
__email__ = "bernhard-reuter@gmx.de"
__copyright__ = "Copyright 2022, Bernhard Reuter"


from datetime import datetime
import joblib
from joblib import Parallel, delayed
import json
import math
from matplotlib import pyplot as plt
import numbers
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import re
import sklearn
from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.metrics import auc, average_precision_score, brier_score_loss
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, fbeta_score, log_loss, precision_score
from sklearn.metrics import precision_recall_curve, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid
from sklearn.utils import check_array, check_consistent_length, check_X_y
from sklearn.metrics._classification import _check_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import type_of_target
import sklearn.pipeline
from sklearn.model_selection._split import BaseCrossValidator
import warnings
from typing import Dict, List, Tuple, Union, Optional, ClassVar, Any

__all__ = [
    'generate_timestamp',
    'save_model',
    'save_json',
    'filename_generator',
    'load_model',
    'RepeatedStratifiedNestedCV',
    'RepeatedGridSearchCV'
]


def generate_timestamp():
    dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    return dt_string


def load_model(filename, method="pickle"):
    if method == "joblib":
        model_loaded = joblib.load(filename)
        return model_loaded
    elif method == "pickle":
        with open(filename, "rb") as file:
            model_loaded = pickle.load(file)
        return model_loaded
    else:
        print("There is no such method as", method)


def load_json(path, file):
    file = os.path.join(path, file)
    # Opening JSON file
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def filename_generator(
    filename: str,
    extension: str,
    directory: Optional[str] = None,
    timestamp: bool = True,
) -> str:
    """Generate a filename (including the absolute path to a given directory,
    if requested) that is ready to use.

    Parameters
    ----------
    filename : string
        Single string denoting the filename.

    extension : string
        A single string (e.g., '.png') that will be appended to the
        end of the filename. Do not forget the dot!

    directory : string, default=None
        If None, the returned filename will contain no path.
        Otherwise, a single string must be given that will be merged
        with the filename (using os.path.join()) to get a valid filepath
        denoting the absolute path were the file shall be located,
        if the filename is later used by some other method to store a file.
        Additionally the directory will be created (via pathlib.Path().mkdir()),
        if it does not exist.

    timestamp : string or boolean, default=True
        If True, a string indicating the current date and time (formated like this:
        26.03.2020_17-52-20) will be appended to the filename (before the extension).
        If a string is given, it should indicate a date and time and will be appended
        to the filename (before the extension).
        """
    if timestamp is True:
        dt_string = generate_timestamp()
        fn = filename + "_" + dt_string + extension
    elif isinstance(timestamp, str):
        dt_string = timestamp
        fn = filename + "_" + dt_string + extension
    else:
        fn = filename + extension

    if isinstance(directory, str):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        fn = os.path.join(directory, fn)

    return fn


def save_model(
    my_model,
    directory: str,
    filename: str,
    timestamp: bool = True,
    compress: bool = False,
    method: str = 'joblib',
) -> None:
    if method == 'joblib':
        if compress:
            fn = filename_generator(filename, extension=".joblib.z",
                                    directory=directory, timestamp=timestamp)
        else:
            fn = filename_generator(filename, extension=".joblib",
                                    directory=directory, timestamp=timestamp)
        if not os.path.exists(fn):
            joblib.dump(my_model, fn, compress=compress)
        else:
            print("The file", fn, "already exists!", sep=" ")
    elif method == 'pickle':
        fn = filename_generator(filename, extension=".pkl", directory=directory,
                                timestamp=timestamp)
        if not os.path.exists(fn):
            with open(fn, "wb") as file:
                pickle.dump(my_model, file)
        else:
            print("The file", fn, "already exists!", sep=" ")
    else:
        raise ValueError(f"Unknown method {method} given.")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(
    dictionary: Dict[Any, Any],
    directory: str,
    filename: str,
    timestamp: bool = True,
) -> None:
    fn = filename_generator(filename, extension=".json", directory=directory,
                            timestamp=timestamp)
    if not os.path.exists(fn):
        with open(fn, 'w') as fp:
            json.dump(dictionary, fp, cls=NpEncoder)
    else:
        print("The file", fn, "already exists!", sep=" ")


def save_dataframe_to_excel(
    dataframe: pd.DataFrame,
    directory: str,
    filename: str,
    timestamp: bool = True
) -> None:
    fn = filename_generator(filename, extension=".xlsx", directory=directory,
                            timestamp=timestamp)
    if not os.path.exists(fn):
        dataframe.to_excel(fn)
    else:
        print("The file", fn, "already exists!", sep=" ")


def save_dataframes_to_excel(
    dataframes: List[pd.DataFrame],
    directory: str,
    filename: str,
    sheetid: str,
    timestamp: bool = True
) -> None:
    '''Save several pandas dataframes as several sheets in one excel workbook.

    Parameters
    ----------
    dataframes : list of dataframes

    directory : string
        A single string denoting the absolute path were the workbook shall be saved.

    filename : string
        Single string denoting the filename of the workbook.
        A string indicating the current date and time (formated like this:
        26.03.2020_17-52-20) plus the file type (.xlsx) will be appended.

    sheetid : string
        A single string denoting the sheet name.
        Every dataframe will be saved in a own sheet that will be named
        as sheetid+str(i) with dataframe = dataframes[i].
        '''
    fn = filename_generator(filename, extension=".xlsx", directory=directory,
                            timestamp=timestamp)
    if not os.path.exists(fn):
        with pd.ExcelWriter(fn) as writer:
            counter = 0
            for dataframe in dataframes:
                sheetname = sheetid+str(counter)
                dataframe.to_excel(writer, sheet_name=sheetname)
                counter += 1
    else:
        print("The file", fn, "already exists!", sep=" ")


def specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """sklearn-compartible function to calculate the specificity = (TN / (TN + FP))."""

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    if not y_type == "binary":
        raise ValueError("%s is not supported" % y_type)

    TN, FP, _, _ = confusion_matrix(
        y_true, y_pred, labels=None, sample_weight=None, normalize=None
    ).ravel(order='C')

    numerator = TN
    denominator = TN + FP
    if denominator == 0.0:
        warnings.warn("Divide through zero encountered while trying to calculate "
                      "the specificity. Specificity is set to zero.")
        result = 0.0
    else:
        result = numerator / denominator
    return result


def Fbeta(
    precision: Union[float, List[float], np.ndarray],
    recall: Union[float, List[float], np.ndarray],
    beta: float
) -> Union[float, np.ndarray]:
    """Function to calculate the 'F-beta' score,
    weighting the two input scores differently depending on beta (beta>0):
    For beta>1 the second input argument 'recall' is stronger weighted
    than the first input argument 'precision',
    while for beta<1 the opposite is true.
    For beta=1, they are equally weighted."""

    if isinstance(precision, list) and isinstance(recall, list):
        precision = np.array(precision, dtype=np.float64)
        recall = np.array(recall, dtype=np.float64)
    if isinstance(precision, np.ndarray) and isinstance(recall, np.ndarray):
        precision = precision.astype(np.float64)
        recall = recall.astype(np.float64)
        if not precision.ndim == recall.ndim == 1:
            raise ValueError("Input arrays must be vectors.")
        if not precision.size == recall.size:
            raise ValueError("Input vectors must be of equal size.")
        numerator = (1 + beta**2) * (precision * recall)
        denominator = ((beta**2 * precision) + recall)
        # avoid zero division:
        mask = denominator == 0.0
        denominator[mask] = 1.0
        fbeta = numerator / denominator
        if np.any(mask):
            warnings.warn("Divide through zero encountered while trying to calculate "
                          "the F%d score. F%d is set to zero accordingly." % (beta, beta))
            fbeta[mask] = 0.0
    elif isinstance(precision, float) and isinstance(recall, float):
        numerator = (1 + beta**2) * (precision * recall)
        denominator = ((beta**2 * precision) + recall)
        if denominator == 0.0:
            warnings.warn("Divide through zero encountered while trying to calculate "
                          "the F%d score. F%d is set to zero accordingly." % (beta, beta))
            fbeta = 0.0
        else:
            fbeta = numerator / denominator
    else:
        raise ValueError("Precision and recall must bei either both lists "
                         "or np.ndarrays of equal lengths or floats.")
    return fbeta


def matthews_corrcoef(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute the Matthews correlation coefficient (MCC).
    The Matthews correlation coefficient takes
    into account true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even if the classes are of
    very different sizes. The MCC is in essence a correlation coefficient value
    between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
    an average random prediction and -1 an inverse prediction.
    Parameters
    ----------
    y_true : array of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples,)
        Estimated targets as returned by a classifier.
    sample_weight : array of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    mcc : float
        The Matthews correlation coefficient (+1 represents a perfect
        prediction, 0 an average random prediction and -1 and inverse
        prediction).
    References
    ----------
    Implementation based on sklearn.metrics.matthews_corrcoef (version 1.0.2).
    """

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if not y_type == "binary":
        raise ValueError("%s is not supported" % y_type)

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        warnings.warn("Divide through zero encountered while trying to calculate "
                      "the MCC. MCC is set to zero.")
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


def negative_predictive_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    if not y_type == "binary":
        raise ValueError("%s is not supported" % y_type)

    TN, _, FN, _ = confusion_matrix(
        y_true, y_pred, labels=None, sample_weight=None, normalize=None
    ).ravel(order='C')

    numerator = TN
    denominator = TN + FN
    if denominator == 0.0:
        warnings.warn("Divide through zero encountered while trying to calculate "
                      "the negative predictive value (NPV). NPV is set to zero.")
        result = 0.0
    else:
        result = numerator / denominator
    return result


def informedness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    if not y_type == "binary":
        raise ValueError("%s is not supported" % y_type)

    tpr = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary',
                       sample_weight=None, zero_division='warn')

    tnr = specificity(y_true, y_pred)

    return tpr + tnr - 1


def markedness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    if not y_type == "binary":
        raise ValueError("%s is not supported" % y_type)

    ppv = precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary',
                          sample_weight=None, zero_division='warn')

    npv = negative_predictive_value(y_true, y_pred)

    return ppv + npv - 1


class BinaryEncoderPandas(BaseEstimator, TransformerMixin):

    def __init__(self, zero_category="wt"):
        self.zero_category = zero_category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """BinaryEncoder transforms a given Pandas DataFrame with categorical data
        such that all (categorical) attribute values are transformed to binary values:
        values equals zero_category (hyperparameter to be set) are set to zero,
        while all other values are set to 1. Returns a transformed copy of the input"""
        X_bin = X.copy()
        if isinstance(X_bin, pd.DataFrame) or isinstance(X_bin, pd.Series):
            X_bin = X_bin.mask(X_bin != self.zero_category, 1.0)
            X_bin = X_bin.mask(X_bin == self.zero_category, 0.0)
            return X_bin.astype(dtype=np.float64)
        else:
            print("Input is neither a pandas DataFrame nor a pandas Series!")


class BinaryEncoderPandasNaN(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """BinaryEncoder transforms a given Pandas DataFrame with categorical data
        such that all (categorical) attribute values are transformed to binary values:
        values equals np.nan are set to zero,
        while all other values are set to 1. Returns a transformed copy of the input"""
        X_bin = X.copy()
        if isinstance(X_bin, pd.DataFrame) or isinstance(X_bin, pd.Series):
            X_bin = X_bin.mask(~X_bin.isna(), 1.0)
            X_bin = X_bin.mask(X_bin.isna(), 0.0)
            return X_bin.astype(dtype=np.float64)
        else:
            print("Input is neither a pandas DataFrame nor a pandas Series!")


class BinaryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, zero_category="wt"):
        self.zero_category = zero_category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """BinaryEncoder transforms a given array with categorical data
        such that all (categorical) attribute values are transformed to binary values:
        values equals zero_category (hyperparameter to be set) are set to zero,
        while all other values are set to 1. Returns a transformed copy of the input"""
        X_bin = X.copy()
        if X_bin.ndim == 1:
            n = X.size
            for i in range(0, n):
                if X[i] == self.zero_category:
                    X_bin[i] = 0.0
                else:
                    X_bin[i] = 1.0
            return X_bin
        elif X_bin.ndim == 2:
            rows, cols = X.shape
            for i in range(0, rows):
                for j in range(0, cols):
                    if X[i, j] == self.zero_category:
                        X_bin[i, j] = 0.0
                    else:
                        X_bin[i, j] = 1.0
            return X_bin
        else:
            print("Input neither 1- nor 2-dimensional!")


class dropRare(BaseEstimator, TransformerMixin):

    def __init__(self, drop_occurrences=1.0):
        self.drop_occurrences = drop_occurrences

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """dropRare transforms a given numpy ndarray with binary numeric data
        (presence/absence encoding: `x_ij = 1.0` if variant `j` is present in sample `i`,
        `x_ij = 0.0` otherwise; format: n_samples x n_features) such that all
        columns (features) with `drop_occurrences` occurrences or less are droped.
        Returns a transformed copy of the input"""
        if isinstance(X, np.ndarray):
            Xc = X.copy()
            Nsam, Ncol = Xc.shape
            keep = np.ones(Ncol, dtype=bool)
            # beware: don't drop the last columns, since this feature is not a variant,
            # but the rule-based genotype!
            for i in range(0, Ncol - 1):
                keep[i] = (np.sum(Xc[:, i]) > self.drop_occurrences)
            Xc = Xc[:, keep]
            # some sanity checks:
            nsam, ncol = Xc.shape
            if nsam != Nsam:
                raise ValueError("The number of samples (rows) changed during dropping "
                                 "of rare variants (columns).")
            often = np.sum(np.sum(X[:, :-1], axis=0) > self.drop_occurrences)
            rare = np.sum(np.sum(X[:, :-1], axis=0) <= self.drop_occurrences)
            if often + rare != Ncol - 1:
                raise ValueError("Something is wrong: The number of rare variants %d "
                                 "and the number of frequent variants %d don't sum up "
                                 "to the total number of variants (number of columns "
                                 "minus one, since the last column isn't a variant but "
                                 "the genotype) %d." % (rare, often, Ncol-1))
            if ncol - 1 != often:
                raise ValueError("%d variants (columns minus one, since the last column "
                                 "isn't a variant but the genotype) were kept, but it "
                                 "should be %d." % (ncol-1, often))
            if ncol + rare != Ncol:
                raise ValueError("The kept columns and the rare variants don't sum "
                                 "up to the original number of columns.")
            return Xc


class dropRarePandas(BaseEstimator, TransformerMixin):

    def __init__(self, drop_occurrences=1.0):
        self.drop_occurrences = drop_occurrences

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """dropRare transforms a given Pandas dataframe with binary numeric data
        (presence/absence encoding: `x_ij = 1.0` if variant `j` is present in sample `i`,
        `x_ij = 0.0` otherwise; format: n_samples x n_features) such that all
        columns (features) with `drop_occurrences` occurrences or less are droped.
        Returns a transformed copy of the input"""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            Xc = X.copy()
            Nsam, Ncol = Xc.shape
            keep = np.ones(Ncol, dtype=bool)
            # beware: don't drop the last column, since this feature is not a variant,
            # but the rule-based genotype!
            for i in range(0, Ncol - 1):
                # keep[i] = (Xc.iloc[:,i].sum() > self.drop_occurrences)
                keep[i] = (np.sum(Xc.iloc[:, i].to_numpy(dtype=np.float64)) >
                           self.drop_occurrences)
            Xc = Xc.loc[:, keep]
            # some sanity checks:
            nsam, ncol = Xc.shape
            if nsam != Nsam:
                raise ValueError("The number of samples (rows) changed during dropping "
                                 "of rare variants (columns).")
            often = np.sum(
                np.sum(X.iloc[:, :-1].to_numpy(dtype=np.float64), axis=0) > self.drop_occurrences
            )
            rare = np.sum(
                np.sum(X.iloc[:, :-1].to_numpy(dtype=np.float64), axis=0) <= self.drop_occurrences
            )
            if often + rare != Ncol - 1:
                raise ValueError("Something is wrong: The number of rare variants %d "
                                 "and the number of frequent variants %d don't sum up "
                                 "to the total number of variants (number of columns "
                                 "minus one, since the last column isn't a variant but "
                                 "the genotype) %d." % (rare, often, Ncol-1))
            if ncol - 1 != often:
                raise ValueError("%d variants (columns minus one, since the last column "
                                 "isn't a variant but the genotype) were kept, but it "
                                 "should be %d." % (ncol-1, often))
            if ncol + rare != Ncol:
                raise ValueError("The kept columns and the rare variants don't sum "
                                 "up to the original number of columns.")
            return Xc


class DataFrameSelector(BaseEstimator, TransformerMixin):
    # Taken from p.68 of A. Gerón (2017): Hands-On Machine Learning
    # with Scikit-Learn & Tensorflow.

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].to_numpy(copy=True)


def split_train_test(data, labels=None, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, labels):
        data_train = data.loc[train_index]
        data_test = data.loc[test_index]
        labels_train = labels.loc[train_index]
        labels_test = labels.loc[test_index]
    return (data_train, data_test, labels_train, labels_test)


def plot_roc_curve(
    fpr: Union[float, List[float], np.ndarray],
    tpr: Union[float, List[float], np.ndarray],
    marker_x_position: Optional[float] = None,
    marker_y_position: Optional[float] = None,
    marker_label: Optional[str] = None,
    label: Optional[str] = None,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    timestamp: bool = True,
    fmt: str = 'pdf'
) -> None:
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8, 8))
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(fpr, tpr, 'r-', marker='.', linewidth=2, label=label)
    if marker_x_position and marker_y_position:
        plt.scatter(marker_x_position, marker_y_position, marker='x',
                    color='black', label=marker_label, zorder=2.5)
    plt.axis([-0.025, 1.025, 0.025, 1.025])
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc='best')
    plt.grid(True)
    # Save figure if requested
    if isinstance(filename, str):
        if isinstance(directory, str):
            filename = filename_generator(filename, '.' + fmt, directory, timestamp)
        plt.savefig(filename, dpi=300)
    # plt.show() mb. needed, if you want to see the figures and not only save them?
    plt.close()


def plot_precision_recall_curve(
    precisions: Union[float, List[float], np.ndarray],
    recalls: Union[float, List[float], np.ndarray],
    no_skill_level: Optional[float] = None,
    marker_x_position: Optional[float] = None,
    marker_y_position: Optional[float] = None,
    marker_label: Optional[str] = None,
    label: Optional[str] = None,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    timestamp: bool = True,
    fmt: str = 'pdf'
) -> None:
    plt.figure(figsize=(8, 8))
    plt.title("Precision-Recall-Curve")
    if no_skill_level:
        plt.plot([0, 1], [no_skill_level, no_skill_level], 'b--', label='No Skill')
    plt.plot(recalls, precisions, 'r-', marker='.', label=label)
    if marker_x_position and marker_y_position:
        plt.scatter(marker_x_position, marker_y_position, marker='x',
                    color='black', label=marker_label, zorder=2.5)
    plt.axis([-0.025, 1.025, 0.025, 1.025])
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    # Save figure if requested
    if isinstance(filename, str):
        if isinstance(directory, str):
            filename = filename_generator(filename, '.' + fmt, directory, timestamp)
        plt.savefig(filename, dpi=300)
    # plt.show() mb. needed, if you want to see the figures and not only save them?
    plt.close()


def plot_precision_recall_vs_threshold(
    precisions: Union[List[float], np.ndarray],
    recalls: Union[List[float], np.ndarray],
    thresholds: Union[List[float], np.ndarray],
    precision_marker: Optional[float] = None,
    recall_marker: Optional[float] = None,
    threshold: Optional[float] = None,
    precision_marker_label: Optional[str] = None,
    recall_marker_label: Optional[str] = None,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    timestamp: bool = True,
    fmt: str = 'pdf'
) -> None:
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall over Decision Threshold")
    plt.plot(thresholds, precisions[:-1], "b-", marker='.', label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", marker='.', label="Recall (Sensitivity)")
    if precision_marker:
        plt.scatter(threshold, precision_marker, marker='o', color='red',
                    label=precision_marker_label, zorder=2.5)
    if recall_marker:
        plt.scatter(threshold, recall_marker, marker='x', color='black',
                    label=recall_marker_label, zorder=3)
    plt.axis([-0.025, 1.025, 0.025, 1.025])
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.grid(True)
    # Save figure if requested
    if isinstance(filename, str):
        if isinstance(directory, str):
            filename = filename_generator(filename, '.' + fmt, directory, timestamp)
        plt.savefig(filename, dpi=300)
    # plt.show() mb. needed, if you want to see the figures and not only save them?
    plt.close()


def plot_specificity_recall_vs_threshold(
    specificities: Union[List[float], np.ndarray],
    recalls: Union[List[float], np.ndarray],
    thresholds: Union[List[float], np.ndarray],
    specificity_marker: Optional[float] = None,
    recall_marker: Optional[float] = None,
    threshold: Optional[float] = None,
    specificity_marker_label: Optional[str] = None,
    recall_marker_label: Optional[str] = None,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    timestamp: bool = True,
    fmt: str = 'pdf'
) -> None:
    plt.figure(figsize=(8, 8))
    plt.title("Specificity and Sensitivity over Decision Threshold")
    plt.plot(thresholds[::-1][1:-1], specificities[::-1][1:-1],
             "b-", marker='.', label="Specificity")
    plt.plot(thresholds[::-1][1:-1], recalls[::-1][1:-1],
             "g-", marker='.', label="Sensitivity (Recall)")
    if specificity_marker:
        plt.scatter(threshold, specificity_marker, marker='o', color='red',
                    label=specificity_marker_label, zorder=2.5)
    if recall_marker:
        plt.scatter(threshold, recall_marker, marker='x', color='black',
                    label=recall_marker_label, zorder=3)
    plt.axis([-0.025, 1.025, 0.025, 1.025])
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.grid(True)
    # Save figure if requested
    if isinstance(filename, str):
        if isinstance(directory, str):
            filename = filename_generator(filename, '.' + fmt, directory, timestamp)
        plt.savefig(filename, dpi=300)
    # plt.show() mb. needed, if you want to see the figures and not only save them?
    plt.close()


def adjusted_classes(
    y_scores: Union[np.ndarray, List[float]],
    t: float,
) -> np.ndarray:
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return np.array([1.0 if y >= t else 0.0 for y in y_scores])


def process_simple_data(drug, mutations, genotype, phenotype):
    """Simple preprocessing "pipeline" for mutation, genotype and phenotype data """

    mutations_raw = mutations.copy(deep=True)
    geno = genotype.copy(deep=True)
    pheno = phenotype.copy(deep=True)

    mutations_bin = BinaryEncoderPandas(zero_category="wt").transform(mutations_raw)
    # print("binary mutations:", mutations_bin.head())
    selector = np.all([pheno[drug].isin(["R", "S"]),
                       geno[drug].isin(["R", "S"])], axis=0)
    # print("selector:", selector)
    predictor = geno[drug].loc[selector]
    # print("predictor:", predictor)
    predictor = BinaryEncoderPandas(zero_category="S").transform(predictor)
    # print("binary predictor:", predictor)
    mutations = mutations_bin.loc[selector, :]
    # print("selected binary mutations:", mutations)
    mutations = pd.concat([mutations, predictor], axis=1)
    # print("mutations with predictor:", mutations)
    response = pheno[drug].loc[selector]
    # print("response:", response)
    response = BinaryEncoderPandas(zero_category="S").transform(response)
    # print("binary response:", response)
    Nout, Ncol = mutations.shape
    keep = np.ones(Ncol, dtype=bool)
    for i in range(0, Ncol):
        keep[i] = (mutations.iloc[:, i].unique().size > 1)
    # print("keep:", keep)
    mutations = mutations.loc[:, keep]
    # print("kept mutations", mutations)
    X = mutations.to_numpy(dtype=np.float64)
    # print("X:", X)
    y = response.to_numpy(dtype=np.float64)
    # print("y:", y)
    return X, y


def _tackle_imbalance(mutations, response, drug, target_pos_frac=0.3,
                      target_neg_frac=0.7, random_state=42):
    # def helper(pos_frac, neg_frac):
    #     target_pos_frac = (pos_frac + 0.5) / 2.0
    #     target_neg_frac = (neg_frac + 0.5) / 2.0
    #     return target_pos_frac, target_neg_frac
    n_p = response.sum(axis=0)
    if math.ceil(n_p) != math.floor(n_p):
        raise ValueError('Number of positives n_p is not integer.')
    n_p = int(n_p)
    # print('n_p:', n_p)
    n_all = response.size
    # print('n_all:', n_all)
    pos_frac = np.true_divide(n_p, n_all)
    # neg_frac = 1.0 - pos_frac

    column = drug + '_pheno'
    response_df = pd.DataFrame(data=response.to_numpy(dtype=np.float64),
                               index=response.index, columns=[column])
    combi = pd.concat([mutations, response_df], axis=1, sort=False)
    # print('combi:', combi)

    if pos_frac < 0.3:
        # indicate to calling routine that condition was met
        indicator = True
        # target_pos_frac, target_neg_frac = helper(pos_frac, neg_frac)
        pos_selector = response.isin([1.0])
        # print('pos_selector:', pos_selector)
        combi_pos = combi.loc[pos_selector, :]
        # print('combi_pos:', combi_pos)
        if combi_pos.loc[:, column].sum(axis=0) != n_p:
            raise ValueError("The number of positives m_p doesn't match the number of "
                             "positives in the positives selection.")
        if combi_pos.shape[0] != n_p:
            raise ValueError("Not all samples in the positives selection are positive.")
        combi_neg = combi.loc[~pos_selector, :]
        if combi_neg.loc[:, column].sum(axis=0) != 0.0:
            raise ValueError("Not all samples in the negatives selection are negative.")
        # print('combi_neg:', combi_neg)
        n_res = int(n_p / target_pos_frac)
        # print('n_res:', n_res)
        n_n = n_res - n_p
        # print('n_n:', n_n)
        if random_state:
            combi_neg_res = combi_neg.sample(n=n_n, replace=False, random_state=random_state)
        else:
            combi_neg_res = combi_neg.sample(n=n_n, replace=False)
        # print('combi_neg_res:', combi_neg_res)
        if combi_neg_res.shape[0] != n_n:
            raise ValueError("The number of samples in the restricted selection "
                             "of negatives is not equal n_n.")
        combi_res = combi_pos.append(combi_neg_res, sort=False)
        # print('combi_res:', combi_res)
        if combi_res.shape[0] != n_res:
            raise ValueError("The number of samples in the restricted selection "
                             "of samples is not equal n_res.")
        if combi_res.shape[1] != mutations.shape[1] + 1:
            raise ValueError("The number of features in the restricted selection "
                             "of samples is not equal to the original number of "
                             "features plus 1.")
        combi_res.sort_index(axis=0, inplace=True, kind='mergesort')
        mutations_res = combi_res.drop(columns=column, inplace=False)
        # print('mutations_res:', mutations_res)
        if mutations_res.shape[0] != n_res:
            raise ValueError("The number of samples in the restricted selection "
                             "of sample mutations is not equal n_res.")
        if mutations_res.shape[1] != mutations.shape[1]:
            raise ValueError("The number of mutations in the restricted selection "
                             "of sample mutations is not equal to the original "
                             "number of mutations.")
        response_res = combi_res.loc[:, column]
        if response_res.size != n_res:
            raise ValueError("The number of samples in the restricted selection "
                             "of responses is not equal n_res.")
        return mutations_res, response_res, indicator
    else:
        indicator = False
        return mutations, response, indicator


def process_variant_collection_data(
        drug, mutations, genotype, phenotype, geno_drug=False, drop_rare=False,
        drop_occurrences=1, target_pos_frac=0.3, target_neg_frac=0.7, random_state=42
):
    """Simple preprocessing "pipeline" for variant-wise mutation, genotype and
    phenotype data. BEWARE: The returned mutations are actually the mutations of
    te selected samples plus the associated genotype!
    """
    if geno_drug is False:
        geno_drug = drug

    geno = genotype.copy(deep=True)
    pheno = phenotype.copy(deep=True)

    selector = np.all([pheno[drug].isin(["R", "S"]),
                       geno[geno_drug].isin(["R", "S"])], axis=0)
    # print("selector:", selector)
    mutations_sel = mutations.loc[selector, :]
    # mutations_sel = mutations_sel.sparse.to_dense()
    mutations_bin = BinaryEncoderPandasNaN().transform(mutations_sel)
    del mutations_sel
    predictor = geno[geno_drug].loc[selector]
    # print("predictor:", predictor)
    predictor = BinaryEncoderPandas(zero_category="S").transform(predictor)
    # print("binary predictor:", predictor)
    # mutations_bin = mutations_bin.loc[selector,:]
    # print("selected binary mutations:", mutations_bin)
    mutations_bin = pd.concat([mutations_bin, predictor], axis=1)
    # print("mutations with predictor:", mutations_bin)
    response = pheno[drug].loc[selector]
    # print("response:", response)
    response = BinaryEncoderPandas(zero_category="S").transform(response)
    # print("binary response:", response)
    # tackle imbalance, if necessary
    mutations_bin, response, indicator = _tackle_imbalance(
        mutations_bin, response, drug, target_pos_frac, target_neg_frac, random_state
    )
    Nout, Ncol = mutations_bin.shape
    keep = np.ones(Ncol, dtype=bool)
    # drop columns with only one unique value, since they are uninformative
    for i in range(0, Ncol - 1):
        keep[i] = (mutations_bin.iloc[:, i].nunique() > 1)
    # print("keep:", keep)
    mutations_bin = mutations_bin.loc[:, keep]
    # print("kept mutations with predictor", mutations_bin)
    # drop columns belonging to rre variants, if requested
    if drop_rare:
        mutations_bin = dropRarePandas(drop_occurrences).transform(mutations_bin)

    X = mutations_bin.to_numpy(dtype=np.float64)
    # print("X:", X)
    y = response.to_numpy(dtype=np.float64)
    # print("y:", y)

    return X, y, mutations_bin, response, indicator


class RepeatedGridSearchCV:
    """A general class to handle nested cross-validated grid-search for any
    estimator that implements the scikit-learn estimator interface.
    Based on Algorithm 1 from Krstajic et al.: Cross-validation pitfalls when
    selecting and assessing regression and classification models. Journal of
    Cheminformatics 2014, 6:10

    Parameters
    ----------

    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_grid : dict or list of dicts
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries.

    scoring : str or list, default='precision_recall_auc'
            Can be a string or a list with elements out of ``'balanced_accuracy'``,
            ``'brier_loss'``, ``'f1'``, ``'f2'``, ``'log_loss'``, ``'mcc'``,
            ``'pseudo_f1'``, ``'pseudo_f2'``, ``'sensitivity'``, ``'average_precision'``,
            ``'precision_recall_auc'`` or ``'roc_auc'``.
            CAUTION: If a list is given, all elements must be unique.
            If ``'balanced_accuracy'``, ``'f1'``, ``'f2'``, ``'mcc'``, ``'pseudo_f1'``,
            ``'pseudo_f2'`` or ``'sensitivity'`` the estimator must support
            the ``predict`` method for predicting binary classes.
            If ``'average_precision'``, ``'brier_loss'``, ``'log_loss'``,
            ``'precision_recall_auc'`` or ``'roc_auc'`` the estimator must support the
            ``predict_proba`` method for predicting binary class probabilities in [0,1].

    cv : int or cross-validatior (a.k.a. CV splitter),
    default=StratifiedKFold(n_splits=5, shuffle=True)
        Determines the cross-validation splitting strategy.
        Possible inputs for ``cv`` are:
        - ``None``, to use the default 5-fold stratified cross validation,
        - int, to specify the number of folds in a ``StratifiedKFold``,
        - CV splitter.

    n_jobs : int or None, default=None
        Number of jobs of GridSearchCV to run in parallel.
        ``None`` means ``1`` unless in a joblib.parallel_backend context.
        ``-1`` means using all processors.

    Nexp : int, default=10
        Number of CV repetitions.

    save_to : dict or None, default=None
        If not ``None``, the train and test indices for every split
        and ``y_proba`` and ``y_test`` for every point of the parameter
        grid will be saved.
        Pass a dict ``{'directory': str, 'ID': str}`` with two keys
        denoting the location and name of output files. The value of the
        key ``'directory'`` is a single string indicating the location were the
        file will be saved. The value of the key ``'ID'`` is a single
        string used to name the output files. A string indicating the current
        date and time (formated like this: 26.03.2020_17-52-20)
        will be appended before the file extension (.npy).

    reproducible : bool, default=False
        If ``True``, the CV splits will become reproducible by setting
        ``cv=StratifiedKFold(n_splits, shuffle=True, random_state=nexp)``
        with ``nexp`` being the iteration of the repetition loop
        and n_splits as given in via the ``cv`` key.


        Attributes
        ----------

        opt_scores_
            A dict of optimal score(s)/loss(es).

        opt_scores_idcs_
            A dict of parameter grid index/indices with optimal score(s)/loss(es).

        opt_params_
            A dict of optimal parameters with maximal score(s)/loss(es).

        cv_results_
            A dict with keys as column headers and values as columns,
            that can be imported into a pandas DataFrame.

        Methods
        -------

        fit(self, X, y)
            Run fit.
    """
    metrics_proba: ClassVar[List[str]] = [
        'average_precision',
        'brier_loss',
        'log_loss',
        'precision_recall_auc',
        'roc_auc',
    ]

    metrics_noproba: ClassVar[List[str]] = [
        'balanced_accuracy',
        'f1',
        'f2',
        'mcc',
        'pseudo_f1',
        'pseudo_f2',
        'sensitivity',
    ]

    metrics: ClassVar[List[str]] = sorted(set(metrics_proba).union(metrics_noproba))

    def __init__(
            self,
            estimator,
            param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
            *,
            scoring: str = 'precision_recall_auc',
            cv: BaseCrossValidator = StratifiedKFold(n_splits=5, shuffle=True),
            n_jobs: Optional[int] = None,
            Nexp: int = 10,
            save_to: Optional[Dict[str, str]] = None,
            reproducible: bool = False,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        if isinstance(scoring, str) or isinstance(scoring, list):
            self.scoring = scoring
        else:
            raise ValueError(
                "'scoring' must be a single str out of %s or a list thereof."
                % ', '.join(self.metrics)
            )
        if isinstance(scoring, list):
            if not len(self.scoring) == len(set(self.scoring)):
                raise ValueError("All elements of 'scoring' must be unique.")
        if isinstance(cv, numbers.Number):
            self.cv = StratifiedKFold(n_splits=cv, shuffle=True)
        elif issubclass(type(cv), BaseCrossValidator):
            self.cv = cv
        else:
            raise ValueError(
                "'cv' must be either an integer, to specify the number of folds, or a CV splitter."
            )
        self.n_jobs = n_jobs
        self.Nexp = Nexp
        self.save_to = save_to
        self.reproducible = reproducible

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:

        # Generate a timestamp used for file naming
        timestamp = generate_timestamp()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X, y = check_X_y(X, y, estimator='RepeatedGridSearchCV')

        y_type = type_of_target(y)

        if not y_type == "binary":
            raise ValueError("Currently only binary targets are supported.")

        grid = list(ParameterGrid(param_grid=self.param_grid))

        cv_results = {}
        cv_results['params'] = grid
        for i, param_dict in enumerate(grid):
            for key in param_dict.keys():
                if key not in cv_results.keys():
                    cv_results[key] = np.ma.MaskedArray(
                        np.full(len(grid), np.nan, dtype=np.object), mask=True
                    )
                    cv_results[key][i] = param_dict[key]
                else:
                    cv_results[key][i] = param_dict[key]

        collect_predictions = False
        collect_probabilities = False
        scores: Dict[str, List[np.ndarray]] = {}
        if isinstance(self.scoring, str):
            if (self.scoring in self.metrics):
                scorings = [self.scoring]
                scores[self.scoring] = []
            else:
                raise ValueError(
                    "'scoring' must be a single str out of %s or a list thereof."
                    % ', '.join(self.metrics)
                )
            if self.scoring in self.metrics_noproba:
                if hasattr(self.estimator, 'predict'):
                    collect_predictions = True
                else:
                    raise TypeError("'estimator' should be an estimator implementing "
                                    "'predict' method, %r was passed" % self.estimator)
            elif self.scoring in self.metrics_proba:
                if hasattr(self.estimator, 'predict_proba'):
                    collect_probabilities = True
                else:
                    raise TypeError("'estimator' should be an estimator implementing "
                                    "'predict_proba' method, %r was passed" % self.estimator)
        elif isinstance(self.scoring, list):
            scorings = []
            for scoring in self.scoring:
                if (scoring in self.metrics):
                    scorings.append(scoring)
                    scores[scoring] = []
                else:
                    raise ValueError(
                        "'scoring' must be a single string out of %s or a list thereof."
                        % ', '.join(self.metrics)
                    )
                if scoring in self.metrics_noproba:
                    if hasattr(self.estimator, 'predict'):
                        collect_predictions = True
                    else:
                        raise TypeError("'estimator' should be an estimator implementing "
                                        "'predict' method, %r was passed" % self.estimator)
                elif scoring in self.metrics_proba:
                    if hasattr(self.estimator, 'predict_proba'):
                        collect_probabilities = True
                    else:
                        raise TypeError("'estimator' should be an estimator implementing "
                                        "'predict_proba' method, %r was passed" % self.estimator)

        # 1. Repeat the following process Nexp times
        for nexp in range(self.Nexp):

            # CAUTION: THIS IS A HACK TO MAKE THE RESULTS REPRODUCIBLE!!!
            if self.reproducible:
                n_splits = self.cv.get_n_splits()
                self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=nexp)

            # a. Divide the dataset X (labelset y) pseudo-randomly into V folds
            # b. For I from 1 to V
            parameters_list = []
            y_preds_list = []
            y_probas_list = []
            y_tests_list = []

            for (j, (train, test)) in enumerate(self.cv.split(X, y)):
                # i. Define set X_train (y_train) as the dataset (labelset) without the I-th fold
                # ii. Define set X_test (y_test) as the I-th fold of the dataset X (labelset y)
                if isinstance(self.estimator, sklearn.pipeline.Pipeline):
                    X_train, y_train = _safe_split(self.estimator.steps[-1][1], X, y, train)
                else:
                    X_train, y_train = _safe_split(self.estimator, X, y, train)
                if isinstance(self.estimator, sklearn.pipeline.Pipeline):
                    X_test, y_test = _safe_split(self.estimator.steps[-1][1], X, y, test, train)
                else:
                    X_test, y_test = _safe_split(self.estimator, X, y, test, train)

                if isinstance(self.save_to, dict):
                    fn = f"{self.save_to['ID']}_repeat{nexp}_inner_split{j}_train_index"
                    fn = filename_generator(fn, '.npy', directory=self.save_to['directory'],
                                            timestamp=timestamp)
                    np.save(fn, train, allow_pickle=False)
                    fn = f"{self.save_to['ID']}_repeat{nexp}_inner_split{j}_test_index"
                    fn = filename_generator(fn, '.npy', directory=self.save_to['directory'],
                                            timestamp=timestamp)
                    np.save(fn, test, allow_pickle=False)

                def _parallel_fitting(X_train, X_test, y_train, y_test, param_dict):
                    # Set hyperparameters, train model on inner split, predict results.

                    estimator = clone(self.estimator)
                    # Set hyperparameters
                    estimator.set_params(**param_dict)

                    # 1. Build a statistical model f^k = f(X_train; alpha^k)
                    # (Fit model with current hyperparameters)
                    estimator.fit(X_train, y_train)

                    # 2. Apply f^k on X_train and store the predictions/probabilities
                    if collect_predictions:
                        y_pred = np.ravel(estimator.predict(X_test))
                    else:
                        y_pred = None

                    if collect_probabilities:
                        y_proba = np.ravel(estimator.predict_proba(X_test)[:, 1])
                    else:
                        y_proba = None

                    return y_pred, y_proba, param_dict, y_test

                # iii. For k from 1 to K
                #     1. Build a statistical model f^k = f(X_train; alpha^k)
                #     2. Apply f^k on X_test and store the predictions
                results = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_fitting)(
                        X_train, X_test, y_train, y_test, param_dict=param_dict
                    ) for param_dict in grid)

                y_preds, y_probas, parameters, y_tests = zip(*results)
                for y_pred, y_proba, param_dict, y_test in zip(y_preds, y_probas,
                                                               parameters, y_tests):
                    if isinstance(self.save_to, dict):
                        if y_pred is not None:
                            fn = f"{self.save_to['ID']}_repeat{nexp}_inner_split{j}_y_pred"
                            for key in param_dict.keys():
                                fn = f"{fn}_{param_dict[key]}_"
                            fn = filename_generator(
                                fn, '.npy', directory=self.save_to['directory'],
                                timestamp=timestamp
                            )
                            np.save(fn, y_pred, allow_pickle=False)
                        if y_proba is not None:
                            fn = f"{self.save_to['ID']}_repeat{nexp}_inner_split{j}_y_proba"
                            for key in param_dict.keys():
                                fn = f"{fn}_{param_dict[key]}_"
                            fn = filename_generator(
                                fn, '.npy', directory=self.save_to['directory'],
                                timestamp=timestamp
                            )
                            np.save(fn, y_proba, allow_pickle=False)
                        fn = f"{self.save_to['ID']}_repeat{nexp}_inner_split{j}_y_test"
                        for key in param_dict.keys():
                            fn = f"{fn}_{param_dict[key]}_"
                        fn = filename_generator(fn, '.npy', directory=self.save_to['directory'],
                                                timestamp=timestamp)
                        np.save(fn, y_test, allow_pickle=False)
                parameters_list.append(parameters)
                y_preds_list.append(y_preds)
                y_probas_list.append(y_probas)
                y_tests_list.append(y_tests)

            # collect probabilties and labels belonging to each point of the grid
            if collect_predictions and collect_probabilities:
                Predictions: Dict[str, List[float]] = {
                    str(tuple(sorted(x.items()))): [] for x in grid
                }
                Probabilities: Dict[str, List[float]] = {
                    str(tuple(sorted(x.items()))): [] for x in grid
                }
            elif collect_predictions and not collect_probabilities:
                Predictions: Dict[str, List[float]] = {
                    str(tuple(sorted(x.items()))): [] for x in grid
                }
                Probabilities = None
            elif not collect_predictions and collect_probabilities:
                Predictions = None
                Probabilities: Dict[str, List[float]] = {
                    str(tuple(sorted(x.items()))): [] for x in grid
                }
            else:
                raise ValueError("Neither Predictions nor Probabilities are collected.")
            y_D: Dict[str, List[float]] = {
                    str(tuple(sorted(x.items()))): [] for x in grid
                }

            for parameters, y_preds, y_probas, y_tests in zip(parameters_list, y_preds_list,
                                                              y_probas_list, y_tests_list):
                for params, y_pred, y_proba, y_test in zip(parameters, y_preds,
                                                           y_probas, y_tests):
                    if collect_predictions:
                        Predictions[str(tuple(sorted(params.items())))].extend(list(y_pred))
                    if collect_probabilities:
                        Probabilities[str(tuple(sorted(params.items())))].extend(list(y_proba))
                    y_D[str(tuple(sorted(params.items())))].extend(list(y_test))

            # c. For each point in the grid calculate scores/losses for all elements in y.
            param_scores: Dict[str, List[float]] = dict()
            for scoring in scorings:
                param_scores[scoring] = []
                for param_dict in grid:
                    key = str(tuple(sorted(param_dict.items())))
                    if scoring == 'balanced_accuracy':
                        param_scores[scoring].append(balanced_accuracy_score(
                            y_D[key], Predictions[key],
                            sample_weight=None, adjusted=False)
                        )
                    elif scoring == 'brier_loss':
                        param_scores[scoring].append(brier_score_loss(
                            y_D[key], Probabilities[key],
                            sample_weight=None, pos_label=1)
                        )
                    elif scoring == 'f1':
                        param_scores[scoring].append(f1_score(
                            y_D[key], Predictions[key], labels=None, pos_label=1,
                            average='binary', sample_weight=None, zero_division='warn')
                        )
                    elif scoring == 'f2':
                        param_scores[scoring].append(fbeta_score(
                            y_D[key], Predictions[key], beta=2.0, labels=None,
                            pos_label=1, average='binary', sample_weight=None,
                            zero_division='warn')
                        )
                    elif scoring == 'log_loss':
                        param_scores[scoring].append(log_loss(
                            y_D[key], Probabilities[key],
                            eps=1e-15, normalize=True,
                            sample_weight=None, labels=None)
                        )
                    elif scoring == 'mcc':
                        param_scores[scoring].append(
                            matthews_corrcoef(
                                y_D[key], Predictions[key], sample_weight=None
                            )
                        )
                    elif scoring == 'pseudo_f1':
                        param_scores[scoring].append(Fbeta(
                            specificity(y_D[key], Predictions[key]),
                            recall_score(y_D[key], Predictions[key], labels=None, pos_label=1,
                                         average='binary', sample_weight=None,
                                         zero_division='warn'),
                            beta=1.0)
                        )
                    elif scoring == 'pseudo_f2':
                        param_scores[scoring].append(Fbeta(
                            specificity(y_D[key], Predictions[key]),
                            recall_score(y_D[key], Predictions[key], labels=None, pos_label=1,
                                         average='binary', sample_weight=None,
                                         zero_division='warn'),
                            beta=2.0)
                        )
                    elif scoring == 'sensitivity':
                        param_scores[scoring].append(recall_score(
                            y_D[key], Predictions[key], labels=None, pos_label=1,
                            average='binary', sample_weight=None, zero_division='warn')
                        )
                    elif scoring == 'average_precision':
                        param_scores[scoring].append(average_precision_score(
                            y_D[key], Probabilities[key], average=None,
                            pos_label=1, sample_weight=None)
                        )
                    elif scoring == 'precision_recall_auc':
                        precision, recall, _ = precision_recall_curve(y_D[key], Probabilities[key],
                                                                      pos_label=1,
                                                                      sample_weight=None)
                        param_scores[scoring].append(auc(recall, precision))
                    elif scoring == 'roc_auc':
                        param_scores[scoring].append(roc_auc_score(
                            y_D[key], Probabilities[key], average=None,
                            sample_weight=None, max_fpr=None, labels=None)
                        )
                cv_results['iteration'+str(nexp)+'_'+scoring] = np.array(param_scores[scoring])
                scores[scoring].append(np.array(param_scores[scoring]))

        opt_scores = dict()
        opt_scores_idcs = dict()
        opt_params = dict()
        for scoring in scorings:
            score_array = np.array(scores[scoring])

            # 2. For each point in the grid calculate the mean of the Nexp scores/losses
            mean_score = np.mean(score_array, axis=0)
            cv_results['mean_'+scoring] = mean_score
            # ...and std
            if self.Nexp > 1:
                std_score = np.std(score_array, ddof=1, axis=0)
            else:
                std_score = np.std(score_array, ddof=0, axis=0)
            cv_results['std_'+scoring] = std_score

            # 3. Let α’ be the α value for which either
            # the average score is maximal or
            # the average loss is minimal.
            if bool(re.search(r'loss', scoring)):
                opt_score = np.amin(mean_score)
                opt_score_idx = np.argmin(mean_score)
            else:
                opt_score = np.amax(mean_score)
                opt_score_idx = np.argmax(mean_score)
            # If there are multiple α values for which the average score/loss is optimal,
            # then α’ is the one with the lowest standard deviation of the score/loss.
            if np.any(mean_score == opt_score):
                if len(np.nonzero(mean_score == opt_score)) != 1:
                    raise ValueError("Unexpected output of np.nonzero(mean_score == opt_score): %s"
                                     % str(np.nonzero(mean_score == opt_score)))
                for i in np.nonzero(mean_score == opt_score)[0]:
                    if std_score[i] < std_score[opt_score_idx]:
                        opt_score_idx = i

            # 4. Select α’ as the optimal cross-validatory choice for tuning parameter and
            # select statistical model f’ = f(X; α’) as the optimal cross-validatory chosen model.
            opt_scores[scoring] = opt_score
            opt_scores_idcs[scoring] = opt_score_idx
            opt_params[scoring] = grid[opt_score_idx]
        self.opt_scores_ = opt_scores
        self.opt_scores_idcs_ = opt_scores_idcs
        self.opt_params_ = opt_params
        self.cv_results_ = cv_results


class RepeatedStratifiedNestedCV:
    """A class to handle repeated stratified nested cross-validated grid-search
    for any estimator that implements the scikit-learn estimator interface.
    Based on Algorithm 2 from Krstajic et al.: Cross-validation pitfalls when
    selecting and assessing regression and classification models. Journal of
    Cheminformatics 2014, 6:10

    Parameters
    ----------

    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values.

    cv_options : dict, default={}
        Nested cross-validation options.

        'collect_rules' : bool, default=False
            Only available, if the estimator is ``RuleBasedClassifier`` (RBC)
            with ``r_solution_`` attribute.
            If set to ``True``, the rules (disjunctions) learned by RBC during
            the outer CV are collected and ranked.
            CAUTION: Please be aware that if you pass a pipeline as estimator
            the RBC must be the last step in the pipeline.
            Otherwise, the rules can't be collected.

        'inner_cv' : int or cross-validatior (a.k.a. CV splitter),
        default=StratifiedKFold(n_splits=5, shuffle=True)
            Determines the inner cross-validation splitting strategy.
            Possible inputs for ``'inner_cv'`` are:
            - int, to specify the number of folds in a ``StratifiedKFold``,
            - CV splitter.

        'n_jobs' : int or None, default=None
            Number of jobs of ``RepeatedGridSearchCV`` to run in parallel.
            ``None`` means ``1`` while ``-1`` means using all processors.

        'Nexp1' : int, default=10
            Number of inner CV repetitions (for hyper-parameter search).

        'Nexp2' : int, default=10
            Number of nested CV repetitions.

        'outer_cv' : int or cross-validatior (a.k.a. CV splitter),
        default=StratifiedKFold(n_splits=5, shuffle=True)
            Determines the outer cross-validation splitting strategy.
            Possible inputs for ``'outer_cv'`` are:
            - integer, to specify the number of folds in a ``StratifiedKFold``,
            - CV splitter.

        'refit' : bool or str, default=False
            Refit an estimator using the best found parameters (rank 1) on the whole dataset.
            For multiple metric evaluation, this needs to be a str denoting the scorer that
            would be used to find the best parameters for refitting the estimator at the end.
            The refitted estimator is made available at the ``best_estimator_`` attribute.

        'reproducible' : bool, default=False
            If True, the CV splits will become reproducible by setting
            ``cv_options['outer_cv']=StratifiedKFold(n_splits, shuffle=True, random_state=nexp2)``,
            ``cv_options['inner_cv']=StratifiedKFold(n_splits, shuffle=True, random_state=nexp1)``
            with ``nexp2``, ``nexp1`` being the current iteration of the outer/inner repetition
            loop and ``n_splits`` as given in via the ``'outer_cv'`` and ``'inner_cv'`` key.

        'save_best_estimator' : dict or None, default=None
            If not ``None``, the best estimator (using the best found parameters)
            refitted on the whole dataset will be saved.
            This requires ``cv_options['refit']`` set to ``True`` or a str denoting
            the scorer used to find the best parameters for refitting,
            in case of multiple metric evaluation.
            Pass a dict ``{'directory': str, 'ID': str}`` with two keys
            denoting the location and name of output files. The value of the
            key ``'directory'`` is a single string indicating the location were
            the file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.joblib).

        'save_inner_to' : dict or None, default=None
            If not ``None``, the train and test indices for every inner split
            and y_proba and y_test for every point of the parameter
            grid of the inner cross-validated grid search will be saved.
            Pass a dict ``{'directory': str, 'ID': str}`` with two keys
            denoting the location and name of output files. The value of the
            key ``'directory'`` is a single string indicating the location were
            the file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.npy).

        'save_pr_plots' : dict or None, default=None
            Only used, if ``cv_options['tune_threshold']=True``.
            Determines, if the Precision-Recall-Curve and a plot of Precision and Recall
            over the decision threshold (if ``cv_options['threshold_tuning_scoring']``
            is set to ``'f1'`` or ``'f2'``) or the ROC-Curve and a plot of Sensitivity
            and Specificity over the decision threshold
            (if ``cv_options['threshold_tuning_scoring']`` is set to ``'balanced_accurracy'``,
            ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``) shall be saved for every outer CV split.
            If None, no plots will be saved.
            To save every plot as individual PDF, pass a dict ``{'directory': str, 'ID': str}``
            with two keys denoting the location and name of output files. The value of the
            key ``'directory'`` is a single string indicating the location were the
            file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.pdf).

        'save_pred' : dict or None, default=None
            If not ``None``, the train and test indices for every outer split
            and ``y_proba``, ``y_pred`` and ``y_test`` for every repetition and split
            of the outer repeated cross-validation will be saved.
            Pass a dict ``{'directory': str, 'ID': str}`` with two keys
            denoting the location and name of output files. The value of the
            key ``'directory'`` is a single string indicating the location were the
            file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.npy).

        'save_to' : dict or None, default=None
            If not ``None``, the results of all inner cross-validated Grid-Search
            iterations per outer iteration will be compiled all together in
            a single Excel workbook with one sheet per outer split and one
            row per inner iteration. Per outer (nested) iteration, one Excel
            workbook will be stored.
            Additionally, if not ``None``, a dict containing all threshold tuning curves,
            i.e.
            - precision and recall over varying threshold,
              if ``cv_options['threshold_tuning_scoring']`` is set to ``'f1'`` or ``'f2'`` or
            - fpr and tpr over varying threshold,
              if ``cv_options['threshold_tuning_scoring']`` is set to ``'balanced_accuracy'``,
              ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``),
            is saved in a .json file.
            Pass a dict ``{'directory': str, 'ID': str}`` with two keys
            denoting the location and identifier of output files. The value of the
            key ``'directory'`` is a single string indicating the location were the
            file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.xlsx).

        'save_tt_plots' : dict or None, default=None
            Only used, if ``cv_options['tune_threshold']=True``.
            Determines, if the Precision-Recall-Curve and a plot of Precision and Recall
            over the decision threshold (if ``cv_options['threshold_tuning_scoring']``
            is set to ``'f1'`` or ``'f2'``) or the ROC-Curve and a plot of Sensitivity and
            Specificity over the decision threshold (if ``cv_options['threshold_tuning_scoring']``
            is set to ``'balanced_accurracy'``, ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``)
            for every outer CV split, compiled together into one PDF per nested CV repetition,
            shall be saved. If ``None``, no plots will be saved.
            To save them, pass a dict ``{'directory': str, 'ID': str}`` with two keys
            denoting the location and name of output files. The value of the
            key ``'directory'`` is a single string indicating the location were the
            file will be saved. The value of the key ``'ID'`` is a single
            string used as part of the name of the output files.
            A string indicating the current date and time
            (formated like this: 26.03.2020_17-52-20)
            will be appended before the file extension (.pdf).

        'scoring' : string or list, default='precision_recall_auc'
            Can be a string or a list with elements out of ``'balanced_accuracy'``,
            ``'brier_loss'``, ``'f1'``, ``'f2'``, ``'log_loss'``, ``'mcc'``, ``'pseudo_f1'``,
            ``'pseudo_f2'``, ``'sensitivity'``, ``'average_precision'``,
            ``'precision_recall_auc'`` or ``'roc_auc'``.
            CAUTION: If a list is given, all elements must be unique.
            If ``'balanced_accuracy'``, ``'f1'``, ``'f2'``, ``'mcc'``, ``'pseudo_f1'``,
            ``'pseudo_f2'`` or ``'sensitivity'`` the estimator must support the ``predict``
            method for predicting binary classes.
            If ``'average_precision'``, ``'brier_loss'``, ``'log_loss'``,
            ``'precision_recall_auc'`` or ``'roc_auc'`` the estimator must support the
            ``predict_proba`` method for predicting binary class probabilities in [0,1].

        'threshold_tuning_scoring' : string or list, default='f2'
            Only used, if ``cv_options['tune_threshold']=True``.
            If a single metric is chosen as scoring (e.g. ``cv_options['scoring']='mcc'``),
            it can be one out of ``'balanced_accurracy'``, ``'f1'``, ``'f2'``,
            ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``.
            If multiple metrics are chosen (e.g. ``cv_options['scoring']=['f1', 'mcc']``),
            it can be one of ``'balanced_accurracy'``, ``'f1'``, ``'f2'``, ``'J'``, ``'pseudo_f1'``
            or ``'pseudo_f2'`` (to perform the same threshold tuning method for all metrics)
            or a list of according length with elements out of ``['balanced_accurracy', 'f1',
            'f2', 'J', 'pseudo_f1', 'pseudo_f2', None]`` to perform different threshold
            tuning methods for every metric:
            E.g., specifying ``cv_options['threshold_tuning_scoring']=['J', None]``,
            while specifying ``cv_options['scoring']=['roc_auc', 'mcc']``, implies tuning the
            decision threshold by selecting the optimal threshold from the ROC curve by
            choosing the threshold with the maximum value of Youden's J statistic after
            performing hyperparameter optimization using ``'roc_auc'`` as scoring, while
            performing no threshold tuning after hyperparameter optimization
            using ``'mcc'`` as scoring.
            For backward compartibility only, ``'precision_recall_auc'`` or ``'roc_auc'``
            are supported options, but shouldn't be used otherwise.
            If choosing ``'precision_recall_auc'``, the optimal threshold will be selected
            from the Precision-Recall curve by choosing the threshold with the maximum
            value of the F-beta score with ``beta=2``. If ``'roc_auc'`` is chosen,
            the optimal threshold will be selected from the Precision-Recall curve
            by choosing the threshold with the maximum value of Youden's J statistic.
            CAUTION:
            Please keep in mind that the scoring metric used during threshold tuning
            should harmonize with the scoring metric used to select the optimal
            hyperparameters during grid search. E.g., choosing ``cv_options['scoring']='roc_auc'``,
            it would be a sensible choice to set ``cv_options['threshold_tuning_scoring']`` to one
            out of ``'balanced_accurracy'``, ``'pseudo_f1'``, ``'pseudo_f2'`` or ``'J'``.
            Choosing ``cv_options['scoring']='precision_recall_auc'``, it would be sensible
            to set ``cv_options['threshold_tuning_scoring']`` to either ``'f1'`` or ``'f2'``.

        'tune_threshold' : boolean, default=True
            If ``True``, perform threshold tuning on each outer training fold
            for the estimator with the best hyperparameters (found by tuning of these
            in the inner cross validation) retrained on the outer training fold.
            A list of thresholds is returned that maximize the scoring metric
            selected via ``cv_options['threshold_tuning_scoring']`` for the best estimators
            (with optimal hyperparameters) on each training fold.
            CAUTION: The estimator must support the ``predict_proba`` method for
            predicting binary class probabilities in [0,1]
            to tune the decision threshold.

        Attributes
        ----------

        best_inner_scores_
            Best inner scores for each outer loop cumulated in a dict.

        best_inner_indices_
            Best inner indices cumulated in a dict.

        best_params_
            All best params from the inner loop cumulated in a dict.

        ranked_best_inner_params_
            Ranked (most frequent first) best inner params as a list of
            dictionaries (every dict, i.e. parameter combination,
            occurs only once).

        best_inner_params_
            Best inner params for each outer loop as a list of dictionaries.

        best_thresholds_
            If cv_option['tune_threshold']=True, this is a dict containing the best thresholds
            and threshold-tuning-scorings for each scoring-threshold_tuning_scoring-pair.
            Given as a dict with scorings as keys and dicts as values. Each of these dicts
            given as ``{'best_<threshold_tuning_scoring>': list, 'best_theshold': list}``
            contains a list of thresholds, which maximize the threshold tuning scoring
            for the best estimators on the outer training folds, and a list of the
            respective maximum values of the threshold-tuning-scoring as values.
            Else the dict is empty.

        best_estimator_
            If ``cv_options['refit']`` is set to ``True`` or a str denoting a valid scoring metric,
            this gives the best estimator refitted on the whole dataset using
            the best parameters (rank 1) found during grid search.
            For multiple metric evaluation, the scoring given via ``cv_options['refit']``
            is used to find the best parameters for refitting the estimator at the end.

        mean_best_threshold_
            If ``cv_options['refit']`` is set to True or a str denoting a valid scoring metric,
            this gives the mean of the best thresholds, which maximize the threshold
            tuning scoring on the outer training folds for the best parameters (rank 1)
            found during grid search.
            For multiple metric evaluation, the scoring given via ``cv_options['refit']``
            is used to find the best parameters.

        repeated_cv_results_lists_
            List of ``Nexp2`` lists of ``n_split_outer`` dicts. Each dict contains the
            compiled results of ``Nexp1`` iterations of ``n_split_inner``-fold cross-
            validated grid searches with keys as column headers and values as columns.
            Each of those dicts can be imported into a pandas DataFrame.

        repeated_cv_results_as_dataframes_list_
            List of ``Nexp2`` lists of ``n_split_outer`` pandas dataframes containing
            the compiled results of of ``Nexp1`` iterations of ``n_split_inner``-fold
            cross-validated grid searches.

        repeated_nested_cv_results_
            A list of ``Nexp2`` dicts of compiled nested CV results.

        Methods
        -------

        fit(self, X, y[, baseline_prediction])
            Method to run repeated stratified nested cross-validated grid-search.

        predict(self, X)
            Call predict on the estimator with the best found parameters.
            Only available if ``cv_options['refit']=True`` and the underlying
            estimator supports ``predict``.
            If threshold tuning was applied, the mean of the best thresholds,
            which maximize the threshold tuning scoring on the outer training
            folds for the best parameters (rank 1) found during grid search,
            is used as decision threshold for class prediction.

        predict_proba(self, X)
            Call predict_proba on the estimator with the best found parameters.
            Only available if ``cv_options['refit']=True`` and the underlying
            estimator supports ``predict_proba``.
    """
    metrics_proba: ClassVar[List[str]] = [
        'average_precision',
        'brier_loss',
        'log_loss',
        'precision_recall_auc',
        'roc_auc',
    ]

    metrics_noproba: ClassVar[List[str]] = [
        'balanced_accuracy',
        'f1',
        'f2',
        'mcc',
        'pseudo_f1',
        'pseudo_f2',
        'sensitivity',
    ]

    metrics: ClassVar[List[str]] = sorted(set(metrics_proba).union(metrics_noproba))

    threshold_tuning_metrics: ClassVar[List[str]] = [
        'balanced_accuracy',
        'f1',
        'f2',
        'J',
        'pseudo_f1',
        'pseudo_f2',
    ]

    standard_metrics_proba: ClassVar[List[str]] = [
        'average_precision',
        'balanced_accuracy',
        'brier_loss',
        'f1',
        'informedness',
        'log_loss',
        'markedness',
        'mcc',
        'negative_predictive_value',
        'precision',
        'precision_recall_auc',
        'roc_auc',
        'sensitivity',
        'specificity',
    ]

    standard_metrics_noproba: ClassVar[List[str]] = [
        'balanced_accuracy',
        'f1',
        'informedness',
        'markedness',
        'mcc',
        'negative_predictive_value',
        'precision',
        'sensitivity',
        'specificity',
    ]

    def __init__(
        self,
        estimator,
        param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
        *,
        cv_options: Dict[str, Any] = {},
    ) -> None:

        self.estimator = estimator

        # check, if estimator has .predict method
        if not hasattr(self.estimator, 'predict'):
            raise TypeError("estimator should be an estimator implementing "
                            "'predict' method, %r was passed" % self.estimator)

        # check, if estimator has .predict_proba method
        if hasattr(self.estimator, 'predict_proba'):
            self.has_proba = True
        else:
            self.has_proba = False

        if self.has_proba:
            self.standard_metrics = self.standard_metrics_proba
        else:
            self.standard_metrics = self.standard_metrics_noproba

        self.param_grid = param_grid

        cv_options_keys = [
            'collect_rules',
            'inner_cv',
            'n_jobs',
            'Nexp1',
            'Nexp2',
            'outer_cv',
            'refit',
            'reproducible',
            'save_best_estimator',
            'save_inner_to',
            'save_pr_plots',
            'save_pred',
            'save_to',
            'save_tt_plots',
            'scoring',
            'threshold_tuning_scoring',
            'tune_threshold',
        ]
        for key in cv_options.keys():
            if key not in cv_options_keys:
                raise ValueError(f"Unknown cv_options key '{key}'.")

        self.collect_rules = cv_options.get('collect_rules', False)
        inner_cv = cv_options.get('inner_cv', StratifiedKFold(n_splits=5, shuffle=True))
        if isinstance(inner_cv, numbers.Number):
            self.inner_cv = StratifiedKFold(n_splits=inner_cv, shuffle=True)
        elif issubclass(type(inner_cv), BaseCrossValidator):
            self.inner_cv = inner_cv
        else:
            raise ValueError("The value of the 'inner_cv' key must be either an integer, "
                             "to specify the number of folds, or a CV splitter.")
        self.n_jobs = cv_options.get('n_jobs', None)
        self.Nexp1 = cv_options.get('Nexp1', 10)
        self.Nexp2 = cv_options.get('Nexp2', 10)
        outer_cv = cv_options.get('outer_cv', StratifiedKFold(n_splits=5, shuffle=True))
        if isinstance(outer_cv, numbers.Number):
            self.outer_cv = StratifiedKFold(n_splits=outer_cv, shuffle=True)
        elif issubclass(type(outer_cv), BaseCrossValidator):
            self.outer_cv = outer_cv
        else:
            raise ValueError("The value of the 'outer_cv' key must be either an integer, "
                             "to specify the number of folds, or a CV splitter.")
        self.refit = cv_options.get('refit', False)
        if not isinstance(self.refit, bool) and not isinstance(self.refit, str):
            raise ValueError("The value of the 'refit' key must bei either boolean or str.")
        self.reproducible = cv_options.get('reproducible', False)
        self.save_best_estimator = cv_options.get('save_best_estimator', None)
        if self.save_best_estimator is not None and not isinstance(self.save_best_estimator, dict):
            raise ValueError(
                "The value of the 'save_best_estimator' key must bei either a dict or None."
            )
        if not self.refit and self.save_best_estimator is not None:
            raise ValueError("Can't save best estimator, if 'refit' is set to False.")
        self.save_inner_to = cv_options.get('save_inner_to', None)
        self.save_pr_plots = cv_options.get('save_pr_plots', None)
        self.save_pred = cv_options.get('save_pred', None)
        self.save_to = cv_options.get('save_to', None)
        self.save_tt_plots = cv_options.get('save_tt_plots', None)
        self.scoring = cv_options.get('scoring', 'precision_recall_auc')
        if isinstance(self.scoring, str):
            if not isinstance(self.refit, bool):
                raise ValueError("The value of the 'refit' key must be boolean "
                                 "in case of single-metric evaluation.")
        elif isinstance(self.scoring, list):
            if not len(self.scoring) == len(set(self.scoring)):
                raise ValueError("You supplied a list as value of the 'scoring' key, "
                                 "but its elements aren't unique.")
            if self.refit:
                if isinstance(self.refit, str):
                    if self.refit not in self.scoring:
                        raise ValueError("The value of the 'refit' key must be a single "
                                         "str out of %s." % ', '.join(self.scoring))
                else:
                    raise ValueError("You must specify the metric to use for refitting, "
                                     "if you are performing multimetric evaluation.")
        else:
            raise ValueError("The value of the 'scoring' key must be a single str out of %s or "
                             "a list thereof." % ', '.join(self.metrics))
        self.threshold_tuning_scoring = cv_options.get('threshold_tuning_scoring', 'f2')
        if (not (isinstance(self.threshold_tuning_scoring, str) or
                 isinstance(self.threshold_tuning_scoring, list))):
            raise ValueError(
                "The value of the 'threshold_tuning_scoring' key must be a single "
                "str out of %s or a list with elements out of [%s, None]."
                % (', '.join(self.threshold_tuning_metrics),
                   ','.join(self.threshold_tuning_metrics))
            )
        self.tune_threshold = cv_options.get('tune_threshold', True)
        if not isinstance(self.tune_threshold, bool):
            raise ValueError("The value of the 'tune_threshold' key must be boolean.")
        # set threshold_tuning_scoring to None (just a safety precaution),
        # if no threshold tuning is requested
        if not self.tune_threshold:
            self.threshold_tuning_scoring = None

    # to convert array of dict to dict with array values,
    # so it can be used as params for parameter tuning
    def _score_to_best_params(
        self,
        best_inner_params_list: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        params_dict: Dict[str, List[Any]] = {}
        for best_inner_params in best_inner_params_list:
            for key, value in best_inner_params.items():
                if key in params_dict:
                    if value not in params_dict[key]:
                        params_dict[key].append(value)
                else:
                    params_dict[key] = [value]
        return params_dict

    # sort the best inner params based on their frequency:
    # return a list of params dicts with the most frequent
    # params dict on the first place (index 0), the second
    # most frequent on the second place (index 1) and so on.
    # Every combination of parameters from best_inner_params_list
    # will only occur once.
    def _rank_params(
        self,
        best_inner_params_list: Union[List[Dict[str, Any]], List[Any]],
    ) -> List[Dict[str, Union[int, Dict[str, Any], List[Any]]]]:
        ranked_params_list = []
        for params in best_inner_params_list:
            if params not in ranked_params_list:
                ranked_params_list.append(params)
        frequency = np.zeros(len(ranked_params_list), dtype=int)
        for params in best_inner_params_list:
            counter = 0
            for p in ranked_params_list:
                if params == p:
                    frequency[counter] += 1
                counter += 1

        def _take_first(elem):
            return elem[0]
        zipped_sorted = sorted(zip(frequency, ranked_params_list), reverse=True, key=_take_first)
        # frequency, ranked_params_list = zip(*zipped_sorted)
        ranked_params_list = []
        for f, params in zipped_sorted:
            ranked_params_list.append({'frequency': f, 'parameters': params})
        return ranked_params_list

    def _tune_threshold_by_prc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ID: Optional[str],
        score: str = 'f2',
        timestamp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        # predict train probabilities (for refited best model)
        y_proba_train = self.estimator.predict_proba(X_train)[:, 1]
        # calculate precision-recall-curve
        precision, recall, threshold = precision_recall_curve(y_train, y_proba_train,
                                                              pos_label=1, sample_weight=None)
        # convert to f-beta score
        if score == 'f1':
            fbeta: np.ndarray = Fbeta(precision, recall, beta=1.0)
        elif score == 'f2':
            fbeta = Fbeta(precision, recall, beta=2.0)
        else:
            raise ValueError("Score for tuning the threshold on the Precision-Recall curve "
                             "must be either 'f1' or 'f2'")
        # convert to f-beta (beta=2) score
        idx = np.argmax(fbeta)
        best_threshold = threshold[idx]
        best_fbeta = fbeta[idx]
        if isinstance(ID, str):
            # plot the precision recall curve on for the model (on X_train)
            no_skill_level = float(len(y_train[y_train == 1])) / float(len(y_train))
            fn = ID + '_Precision-Recall'
            plot_precision_recall_curve(
                precision, recall, no_skill_level, marker_x_position=recall[idx],
                marker_y_position=precision[idx],
                marker_label='F_beta=%.6f\nThreshold=%.6f' % (fbeta[idx], best_threshold),
                label='Model', directory=self.save_pr_plots['directory'],
                filename=fn, timestamp=timestamp
            )
            # plot Precision and Recall over the threshold
            fn = ID + '_Precision_Recall_vs_Threshold'
            plot_precision_recall_vs_threshold(
                precision, recall, threshold, precision_marker=precision[idx],
                recall_marker=recall[idx], threshold=best_threshold,
                precision_marker_label='Precision=%.6f\nThreshold=%.6f' % (precision[idx],
                                                                           best_threshold),
                recall_marker_label='Recall=%.6f\nThreshold=%.6f' % (recall[idx], best_threshold),
                directory=self.save_pr_plots['directory'], filename=fn, timestamp=timestamp
            )
        return precision, recall, threshold, best_threshold, best_fbeta

    def _tune_threshold_by_roc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ID: Optional[str],
        score: str = 'J',
        timestamp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        # predict train probabilities (for refited best model)
        y_proba_train = self.estimator.predict_proba(X_train)[:, 1]
        # calculate precision-recall-curve
        fpr, tpr, threshold = roc_curve(y_train, y_proba_train, pos_label=1,
                                        sample_weight=None, drop_intermediate=False)
        spec = 1 - fpr
        name = score
        if score == 'J':
            name = "Youden's J"
            # calculate Youden's J statistic
            scores = tpr - fpr
            if np.any(scores < 0.0):
                # warn, if any elements of J are negative
                warnings.warn(f"At least one element of Youden's-J={scores} is negative.")
            elif np.all(scores < 0.0):
                # warn, if all elements of J are negative
                warnings.warn(f"All elements of Youden's-J={scores} are negative indicating that "
                              "positive and negative labels have been switched.")
        elif score == 'pseudo_f1':
            scores = Fbeta(spec, tpr, beta=1.0)
        elif score == 'pseudo_f2':
            scores = Fbeta(spec, tpr, beta=2.0)
        elif score == 'balanced_accuracy':
            scores = (tpr + spec) / 2.0
        else:
            raise ValueError("Score for tuning the threshold on the ROC curve must be one out of "
                             "'balanced_accuracy', 'pseudo_f1', 'pseudo_f2', or 'J'")
        # locate the index of the largest score
        idx = np.argmax(scores)
        best_threshold = threshold[idx]
        best_score = scores[idx]
        if isinstance(ID, str):
            # plot the roc for the model (on X_train)
            fn = ID + '_ROC'
            plot_roc_curve(
                fpr, tpr, marker_x_position=fpr[idx], marker_y_position=tpr[idx],
                marker_label="%s=%.6f\nThreshold=%.6f" % (name, best_score, best_threshold),
                directory=self.save_pr_plots['directory'], filename=fn, timestamp=timestamp
            )
            # plot Specificity and Recall over the threshold
            fn = ID + '_Recall_Specificity_vs_Threshold'
            plot_specificity_recall_vs_threshold(
                spec, tpr, threshold, specificity_marker=spec[idx],
                recall_marker=tpr[idx], threshold=best_threshold,
                specificity_marker_label='Specificity=%.6f\nThreshold=%.6f' % (spec[idx],
                                                                               best_threshold),
                recall_marker_label='Recall=%.6f\nThreshold=%.6f' % (tpr[idx], best_threshold),
                directory=self.save_pr_plots['directory'], filename=fn, timestamp=timestamp
            )
        return fpr, tpr, threshold, best_threshold, best_score

    def _predict(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert X_train.shape[0] == y_train.shape[0], \
            "X_train.shape[0] != y_train.shape[0] in _predict."
        assert X_test.shape[0] == y_test.shape[0], \
            "X_test.shape[0] != y_test.shape[0] in _predict."
        if threshold and self.has_proba:
            # predict train probabilities (for refitted best model)
            y_proba_train = self.estimator.predict_proba(X_train)[:, 1]
            # predict train classes with respect to threshold t
            y_pred_train = adjusted_classes(y_proba_train, threshold)
            # predict test probabilities
            y_proba_test = self.estimator.predict_proba(X_test)[:, 1]
            # predict test classes with respect to threshold t
            y_pred_test = adjusted_classes(y_proba_test, threshold)
        elif self.has_proba:
            # predict train probabilities (for refitted best model)
            y_proba_train = self.estimator.predict_proba(X_train)[:, 1]
            # predict train classes
            y_pred_train = self.estimator.predict(X_train)
            # predict test probabilities
            y_proba_test = self.estimator.predict_proba(X_test)[:, 1]
            # predict test classes
            y_pred_test = self.estimator.predict(X_test)
        else:
            y_proba_train = np.full((y_train.shape[0],), fill_value=np.nan)
            # predict train classes
            y_pred_train = self.estimator.predict(X_train)

            y_proba_test = np.full((y_test.shape[0],), fill_value=np.nan)
            # predict test classes
            y_pred_test = self.estimator.predict(X_test)
        return y_proba_train, y_pred_train, y_proba_test, y_pred_test

    def _score(
        self,
        y: Union[np.ndarray, float],
        y_pred: Union[np.ndarray, float],
        y_proba: Union[np.ndarray, float] = np.nan,
        scoring: str = 'mcc',
    ) -> float:
        if scoring == 'average_precision':
            if not np.isnan(y_proba).any():
                score = average_precision_score(
                    y, y_proba, average=None, pos_label=1, sample_weight=None
                )
            else:
                score = np.nan
        elif scoring == 'balanced_accuracy':
            score = balanced_accuracy_score(
                y, y_pred, sample_weight=None, adjusted=False
            )
        elif scoring == 'brier_loss':
            if not np.isnan(y_proba).any():
                score = brier_score_loss(y, y_proba, sample_weight=None, pos_label=1)
            else:
                score = np.nan
        elif scoring == 'f1':
            score = f1_score(
                y, y_pred, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn'
            )
        elif scoring == 'f2':
            score = fbeta_score(
                y, y_pred, beta=2.0, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn'
            )
        elif scoring == 'informedness':
            score = informedness(y, y_pred)
        elif scoring == 'markedness':
            score = markedness(y, y_pred)
        elif scoring == 'mcc':
            score = matthews_corrcoef(
               y, y_pred, sample_weight=None
            )
        elif scoring == 'log_loss':
            if not np.isnan(y_proba).any():
                score = log_loss(
                    y, y_proba, eps=1e-15, normalize=True, sample_weight=None, labels=None
                )
            else:
                score = np.nan
        elif scoring == 'negative_predictive_value':
            score = negative_predictive_value(y, y_pred)
        elif scoring == 'precision':
            score = precision_score(
                y, y_pred, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn'
            )
        elif scoring == 'precision_recall_auc':
            if not np.isnan(y_proba).any():
                precision, recall, _ = precision_recall_curve(y, y_proba,
                                                              pos_label=1,
                                                              sample_weight=None)
                score = auc(recall, precision)
            else:
                score = np.nan
        elif scoring == 'pseudo_f1':
            score = Fbeta(
                specificity(y, y_pred),
                recall_score(y, y_pred, labels=None, pos_label=1,
                             average='binary', sample_weight=None,
                             zero_division='warn'),
                beta=1.0
            )
        elif scoring == 'pseudo_f2':
            score = Fbeta(
                specificity(y, y_pred),
                recall_score(y, y_pred, labels=None, pos_label=1,
                             average='binary', sample_weight=None,
                             zero_division='warn'),
                beta=2.0
            )
        elif scoring == 'roc_auc':
            if not np.isnan(y_proba).any():
                score = roc_auc_score(
                    y, y_proba, average=None, sample_weight=None, max_fpr=None, labels=None
                )
            else:
                score = np.nan
        elif scoring == 'sensitivity':
            score = recall_score(
                y, y_pred, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn'
            )
        elif scoring == 'specificity':
            score = specificity(y, y_pred)
        else:
            raise ValueError(f"Scoring needs to be one of {self.performance_evaluation_metrics}.")
        return score

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Call predict on the estimator with the best found parameters.
        Only available if ``cv_options['refit']=True`` and the underlying
        estimator supports ``predict``.
        If threshold tuning was applied, the mean of the best thresholds,
        which maximize the threshold tuning scoring on the outer training
        folds for the best parameters (rank 1) found during grid search,
        is used as decision threshold for class prediction.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        # Check, if fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Return predict from estimator method, if refit=True
        if self.refit:
            if self.mean_best_threshold_:
                return adjusted_classes(
                    self.estimator.predict_proba(X)[:, 1], self.mean_best_threshold_
                )
            else:
                return self.best_estimator_.predict(X)
        else:
            raise AttributeError("RepeatedStratifiedNestedC was initialized "
                                 "with cv_options['refit']=False. The 'predict' method is "
                                 "available only after refitting on the best "
                                 "parameters.")

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Call predict_proba on the estimator with the best found parameters.
        Only available if ``cv_options['refit']=True`` and the underlying
        estimator supports ``predict_proba``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        # Check, if fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Return proba from estimator method, if refit=True
        if self.refit:
            if self.has_proba:
                return self.best_estimator_.predict_proba(X)
            else:
                raise AttributeError("Estimator %s doesn't support 'predict_proba' method."
                                     % self.estimator)
        else:
            raise AttributeError("RepeatedStratifiedNestedC was initialized "
                                 "with cv_options['refit']=False. The 'predict_proba' method is "
                                 "available only after refitting on the best "
                                 "parameters.")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        baseline_prediction: Optional[np.ndarray] = None
    ) -> None:
        """A method to run repeated stratified nested cross-validated grid-search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification.

        baseline_prediction : None or array-like of shape (n_samples,), default=None
            Given baseline prediction of the target.
            Used to calculate the baseline performance scores.

        Returns
        -------
        It will not return the values directly,
        but they are accessible from the class object itself.
        You should be able to access the following class attributes
        of RepeatedStratifiedNestedCV:

        Attributes
        ----------

        best_inner_scores_
            Best inner scores for each outer loop.

        best_inner_indices_
            Best inner indices.

        best_params_
            All best params from the inner loop cumulated in a dict.

        ranked_best_inner_params_
            Ranked (most frequent first) best inner params as a list of
            dictionaries (every dict, i.e. parameter combination,
            occurs only once).

        best_inner_params_
            Best inner params for each outer loop as a list of dictionaries.

        best_thresholds_
            If cv_options['tune_threshold']=True, this is a dict containing the best thresholds
            and threshold-tuning-scorings for each scoring-threshold_tuning_scoring-pair.
            Given as a dict with scorings as keys and dicts as values. Each of these dicts
            given as {'best_<threshold_tuning_scoring>': value, 'best_theshold': value}
            contains a list of thresholds, which maximize the threshold tuning scoring
            for the best estimators on the outer training folds, and a list of the
            respective maximum values of the threshold-tuning-scoring as values.
            Else the dict is empty.

        best_estimator_
            If cv_options['refit'] is set to True or a str denoting a valid scoring metric,
            this gives the best estimator refitted on the whole dataset using
            the best parameters (rank 1) found during grid search.
            For multiple metric evaluation, the scoring given via the refit key
            is used to find the best parameters for refitting the estimator at the end.

        mean_best_threshold_
            If cv_options['refit'] is set to True or a str denoting a valid scoring metric,
            this gives the mean of the best thresholds, which maximize the threshold
            tuning scoring on the outer training folds for the best parameters (rank 1)
            found during grid search.
            For multiple metric evaluation, the scoring given via the refit key
            is used to find the best parameters.

        repeated_cv_results_lists_
            List of Nexp2 lists of n_split_outer dicts. Each dict contains the
            compiled results of Nexp1 iterations of n_split_inner-fold cross-
            validated grid searches with keys as column headers and values as columns.
            Each of those dicts can be imported into a pandas DataFrame.

        repeated_cv_results_as_dataframes_list_
            List of Nexp2 lists of n_split_outer pandas dataframes containing
            the compiled results of of Nexp1 iterations of n_split_inner-fold
            cross-validated grid searches.

        repeated_nested_cv_results_
            A list of Nexp2 dicts of compiled nested CV results.
        """

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X, y = check_X_y(X, y, estimator='RepeatedStratifiedNestedCV')

        y_type = type_of_target(y)

        if not y_type == "binary":
            raise ValueError("Currently only binary targets are supported.")

        self.X = X
        self.y = y

        # Generate a timestamp used for file naming
        timestamp = generate_timestamp()

        # set up hyperparameter tuning scorings
        scores: Dict[str, Dict[str, List[float]]] = {}
        best_inner_index_dict: Dict[str, List[int]] = {}
        best_inner_params_dict: Dict[str, List[Dict[str, Any]]] = {}
        best_inner_params_json_dict: Dict[str, List[Dict[str, Any]]] = {}
        best_inner_score_dict: Dict[str, List[float]] = {}
        params_dummy: Dict[str, List[Dict[str, Any]]] = {}
        rules: Dict[str, List[List[int]]] = {}
        if isinstance(self.scoring, str):
            if self.scoring in self.metrics:
                scorings = [self.scoring]
                scores[self.scoring] = {}
                best_inner_index_dict[self.scoring] = []
                best_inner_params_dict[self.scoring] = []
                best_inner_params_json_dict[self.scoring] = []
                best_inner_score_dict[self.scoring] = []
                # dummy dict just for assertion use:
                params_dummy[self.scoring] = []
                rules[self.scoring] = []
            else:
                raise ValueError("The value of the 'scoring' key must be a single str out of %s "
                                 "or a list thereof." % ', '.join(self.metrics))
            if self.scoring in self.metrics_noproba:
                if not hasattr(self.estimator, 'predict'):
                    raise TypeError("estimator should be an estimator implementing "
                                    "'predict' method, %r was passed" % self.estimator)
            elif self.scoring in self.metrics_proba:
                if not hasattr(self.estimator, 'predict_proba'):
                    raise TypeError("estimator should be an estimator implementing "
                                    "'predict_proba' method, %r was passed" % self.estimator)
        elif isinstance(self.scoring, list):
            scorings = []
            for scoring in self.scoring:
                if scoring in self.metrics:
                    scorings.append(scoring)
                    scores[scoring] = {}
                    best_inner_index_dict[scoring] = []
                    best_inner_params_dict[scoring] = []
                    best_inner_params_json_dict[scoring] = []
                    best_inner_score_dict[scoring] = []
                    # dummy dict just for assertion use:
                    params_dummy[scoring] = []
                    rules[scoring] = []
                else:
                    raise ValueError("The value of the 'scoring' key must be a single str out "
                                     "of %s or a list thereof." % ', '.join(self.metrics))
                if scoring in self.metrics_noproba:
                    if not hasattr(self.estimator, 'predict'):
                        raise TypeError("The estimator must be an estimator implementing "
                                        "'predict' method, but %r was passed" % self.estimator)
                elif scoring in self.metrics_proba:
                    if not hasattr(self.estimator, 'predict_proba'):
                        raise TypeError("The estimator must be an estimator implementing the "
                                        "'predict_proba' method, but %r was passed"
                                        % self.estimator)

        # set up performance evaluation metrics
        self.performance_evaluation_metrics = set(self.standard_metrics).union(set(scorings))

        # set up threshold tuning scorings
        if isinstance(self.threshold_tuning_scoring, str):
            if self.threshold_tuning_scoring in self.threshold_tuning_metrics:
                threshold_scorings: List[Union[str, None]] = [
                    self.threshold_tuning_scoring for x in range(len(scorings))
                ]
                if not hasattr(self.estimator, 'predict_proba') and self.tune_threshold:
                    raise TypeError("To tune the desicion threshold, the estimator must be an "
                                    "estimator implementing the 'predict_proba' method, "
                                    "but %r was passed" % self.estimator)
            else:
                raise ValueError(
                    "The value of the 'threshold_tuning_scoring' key must be a single "
                    "str out of %s or a list with elements out of [%s, None]."
                    % (', '.join(self.threshold_tuning_metrics),
                       ','.join(self.threshold_tuning_metrics))
                )
        elif isinstance(self.threshold_tuning_scoring, list):
            assert isinstance(self.scoring, list), \
                "If 'threshold_tuning_scoring' is a list, 'scoring' must be a list as well."
            assert len(self.threshold_tuning_scoring) == len(self.scoring), \
                "len(threshold_tuning_scoring) != len(scoring)."
            threshold_scorings = []
            for threshold_scoring in self.threshold_tuning_scoring:
                if threshold_scoring in self.threshold_tuning_metrics:
                    threshold_scorings.append(threshold_scoring)
                    if not hasattr(self.estimator, 'predict_proba') and self.tune_threshold:
                        raise TypeError("To tune the decision threshold, the estimator must be an "
                                        "estimator implementing the 'predict_proba' method, "
                                        "but %r was passed" % self.estimator)
                elif threshold_scoring is None:
                    threshold_scorings.append(None)
                else:
                    raise ValueError(
                        "The value of the 'threshold_tuning_scoring' key must be a single "
                        "str out of %s or a list with elements out of [%s, None]."
                        % (', '.join(self.threshold_tuning_metrics),
                           ','.join(self.threshold_tuning_metrics))
                    )
            if not any(threshold_scorings):
                raise ValueError("There is no point in requesting threshold tuning while "
                                 "setting all threshold tuning scorings to None")
        else:
            threshold_scorings = [None for x in range(len(scorings))]

        thresholds: Dict[str, Dict[str, List[float]]] = {}
        tuning_curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for scoring, threshold_scoring in zip(scorings, threshold_scorings):
            if self.tune_threshold is True:
                thresholds[scoring] = {}
                tuning_curves[scoring] = {}
                if threshold_scoring in self.threshold_tuning_metrics:
                    thresholds[scoring][f'best_{threshold_scoring}'] = []
                    thresholds[scoring]['best_thresholds'] = []
                    if threshold_scoring in ['f1', 'f2']:
                        tuning_curves[scoring]['precision'] = []
                        tuning_curves[scoring]['recall'] = []
                        tuning_curves[scoring]['threshold'] = []
                    elif threshold_scoring in ['balanced_accuracy', 'pseudo_f1',
                                               'pseudo_f2', 'J']:
                        tuning_curves[scoring]['fpr'] = []
                        tuning_curves[scoring]['tpr'] = []
                        tuning_curves[scoring]['threshold'] = []

        repeated_cv_results_lists = []
        repeated_cv_results_as_dataframes_list = []

        gs = RepeatedGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.inner_cv,
            n_jobs=self.n_jobs,
            Nexp=self.Nexp1,
            save_to=self.save_inner_to,
            reproducible=self.reproducible
        )

        # Repeated Nested CV with parameter optimization.
        for nexp2 in range(self.Nexp2):

            # CAUTION: THIS IS A HACK TO MAKE THE RESULTS REPRODUCIBLE!!!
            if self.reproducible:
                n_splits = self.outer_cv.get_n_splits()
                self.outer_cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=nexp2
                )

            train_indices = []
            test_indices = []
            # for train, test in outer_cv.split(X, y):
            for i, (train, test) in enumerate(self.outer_cv.split(X, y)):
                train_indices.append(train)
                test_indices.append(test)

                # save train/test split indices
                if isinstance(self.save_pred, dict):
                    fn = f"{self.save_pred['ID']}_repeat{nexp2}_outer_split{i}_train_index"
                    fn = filename_generator(fn, '.npy', directory=self.save_pred['directory'],
                                            timestamp=timestamp)
                    np.save(fn, train, allow_pickle=False)
                    fn = f"{self.save_pred['ID']}_repeat{nexp2}_outer_split{i}_test_index"
                    fn = filename_generator(fn, '.npy', directory=self.save_pred['directory'],
                                            timestamp=timestamp)
                    np.save(fn, test, allow_pickle=False)

            repeated_cv_results_list = []
            repeated_cv_results_as_dataframes = []
            for i, (train, test) in enumerate(zip(train_indices, test_indices)):
                if isinstance(self.estimator, sklearn.pipeline.Pipeline):
                    X_train, y_train = _safe_split(self.estimator.steps[-1][1], X, y, train)
                else:
                    X_train, y_train = _safe_split(self.estimator, X, y, train)
                print('Repetition %s, outer split %s:' % (str(nexp2), str(i)))
                print('Beginning of grid search at %s.' % generate_timestamp())
                gs.fit(X_train, y_train)
                print('End of grid search at %s.' % generate_timestamp())
                print('')
                repeated_cv_results_list.append(gs.cv_results_)
                repeated_cv_results_as_dataframes.append(pd.DataFrame(data=gs.cv_results_))

                # for key in gs.opt_scores_.keys():
                for scoring in scorings:
                    best_inner_score_dict[scoring].append(gs.opt_scores_[scoring])
                    best_inner_index_dict[scoring].append(gs.opt_scores_idcs_[scoring])
                    best_inner_params_dict[scoring].append(gs.opt_params_[scoring])
                    dict_ = dict()
                    for key, item in gs.opt_params_[scoring].items():
                        if (isinstance(item, (float, int, str, list, dict, tuple, bool,
                                              np.integer, np.floating, np.ndarray)) or
                                item is None):
                            dict_[key] = item
                        else:
                            try:
                                dict_[key] = str(item)
                            except UnicodeDecodeError:
                                dict_[key] = item
                                warnings.warn(
                                    f"{scoring}: The value {item} of the parameter {key} is not "
                                    "JSON encodeable. Thus writing out the results "
                                    "as JSON will fail."
                                )
                    best_inner_params_json_dict[scoring].append(dict_)

            for scoring, threshold_scoring in zip(scorings, threshold_scorings):
                performances: Dict[str, Dict[str, List[float]]] = {
                    'train': dict(),
                    'test': dict(),
                }
                for pem in self.performance_evaluation_metrics:
                    performances['train'][pem] = []
                    performances['test'][pem] = []
                params_list = []
                Probabilities = []
                Predictions = []
                y_D = []
                rules_list = []

                # if threshold is tuned, plot all specificity-recall-plots together
                if self.tune_threshold is True and isinstance(self.save_tt_plots, dict):
                    # markers = ['.', 'o', 'v', '^', '>', '<', 's', '*', 'x', 'D']
                    plt.figure(figsize=(8, 8))
                for i, (train, test) in enumerate(zip(train_indices, test_indices)):
                    y_dict = dict()
                    X_dict = dict()
                    if isinstance(self.estimator, sklearn.pipeline.Pipeline):
                        X_dict['train'], y_dict['train'] = _safe_split(
                            self.estimator.steps[-1][1], X, y, train
                        )
                        X_dict['test'], y_dict['test'] = _safe_split(
                            self.estimator.steps[-1][1], X, y, test, train
                        )
                    else:
                        X_dict['train'], y_dict['train'] = _safe_split(
                            self.estimator, X, y, train
                        )
                        X_dict['test'], y_dict['test'] = _safe_split(
                            self.estimator, X, y, test, train
                        )

                    index = (len(train_indices) * nexp2) + i
                    # collect hyperparameters for params_dummy dict:
                    params_list.append(best_inner_params_dict[scoring][index])

                    # Set hyperparameters
                    self.estimator.set_params(**best_inner_params_dict[scoring][index])

                    # Fit model with current optimal hyperparameters and score it
                    self.estimator.fit(X_dict['train'], y_dict['train'])

                    # if estimator is RuleBasedClassifier obtain and collect the learned rule,
                    # if requested:
                    if self.collect_rules:
                        if isinstance(self.estimator, sklearn.pipeline.Pipeline):
                            if hasattr(self.estimator.steps[-1][1], 'r_solution_'):
                                try:
                                    rules_list.append(self.estimator.steps[-1][1].r_solution_)
                                except Exception as e:
                                    print(e, e.args)
                        else:
                            if hasattr(self.estimator, 'r_solution_'):
                                try:
                                    rules_list.append(self.estimator.r_solution_)
                                except Exception as e:
                                    print(e, e.args)

                    if self.tune_threshold is True:
                        if isinstance(self.save_pr_plots, dict):
                            if threshold_scoring in self.threshold_tuning_metrics:
                                ID = (f"{self.save_pr_plots['ID']}_repeat{nexp2}"
                                      f"_outer_split{i}_{scoring}_{threshold_scoring}")
                            else:
                                ID = None
                        else:
                            ID = None
                        if threshold_scoring in ['f1', 'f2']:
                            (precision,
                             recall,
                             threshold,
                             best_threshold,
                             best_score) = self._tune_threshold_by_prc(X_dict['train'],
                                                                       y_dict['train'], ID,
                                                                       score=threshold_scoring,
                                                                       timestamp=timestamp)
                            thresholds[scoring][f'best_{threshold_scoring}'].append(best_score)
                            thresholds[scoring]['best_thresholds'].append(best_threshold)
                            tuning_curves[scoring]['precision'].append(precision)
                            tuning_curves[scoring]['recall'].append(recall)
                            tuning_curves[scoring]['threshold'].append(threshold)
                            # plot all precision-recall-plots together, if asked
                            if isinstance(self.save_tt_plots, dict):
                                if i == 0:
                                    plt.plot(threshold, precision[:-1], "b-",
                                             label="Precision", linewidth=1)
                                    plt.plot(threshold, recall[:-1], "r-",
                                             label="Sensitivity", linewidth=1)
                                else:
                                    plt.plot(threshold, precision[:-1], "b-", linewidth=1)
                                    plt.plot(threshold, recall[:-1], "r-", linewidth=1)
                        elif threshold_scoring in ['balanced_accuracy', 'pseudo_f1',
                                                   'pseudo_f2', 'J']:
                            (fpr,
                             tpr,
                             threshold,
                             best_threshold,
                             best_score) = self._tune_threshold_by_roc(X_dict['train'],
                                                                       y_dict['train'], ID,
                                                                       score=threshold_scoring,
                                                                       timestamp=timestamp)
                            thresholds[scoring][f'best_{threshold_scoring}'].append(best_score)
                            thresholds[scoring]['best_thresholds'].append(best_threshold)
                            tuning_curves[scoring]['fpr'].append(fpr)
                            tuning_curves[scoring]['tpr'].append(tpr)
                            tuning_curves[scoring]['threshold'].append(threshold)
                            # plot all recall-specificity-plots together, if asked
                            if isinstance(self.save_tt_plots, dict):
                                spec = 1 - fpr
                                if i == 0:
                                    plt.plot(threshold[::-1][1:-1], spec[::-1][1:-1], "b-",
                                             label="Specificity", linewidth=1)
                                    plt.plot(threshold[::-1][1:-1], tpr[::-1][1:-1], "r-",
                                             label="Sensitivity", linewidth=1)
                                else:
                                    plt.plot(threshold[::-1][1:-1], spec[::-1][1:-1],
                                             'b-', linewidth=1)
                                    plt.plot(threshold[::-1][1:-1], tpr[::-1][1:-1],
                                             'r-', linewidth=1)
                        else:
                            best_threshold = None
                    else:
                        best_threshold = None
                    # generate train/test predictions and probabilities
                    y_p: Dict[str, Dict[str, np.ndarray]] = {'pred': dict(), 'proba': dict()}
                    (y_p['proba']['train'],
                     y_p['pred']['train'],
                     y_p['proba']['test'],
                     y_p['pred']['test']) = self._predict(
                        X_dict['test'], y_dict['test'], X_dict['train'], y_dict['train'],
                        threshold=best_threshold
                    )
                    # collect test predictions and probabilities
                    Probabilities.extend(list(y_p['proba']['test']))
                    Predictions.extend(list(y_p['pred']['test']))
                    y_D.extend(list(y_dict['test']))
                    # save test labels, predictions and probabilities
                    if isinstance(self.save_pred, dict):
                        fn = (f"{self.save_pred['ID']}_repeat{nexp2}_"
                              f"{scoring}_outer_split{i}_y_proba")
                        for key in params_list[i].keys():
                            fn = f"{fn}_{params_list[i][key]}_"
                        fn = filename_generator(fn, '.npy', directory=self.save_pred['directory'],
                                                timestamp=timestamp)
                        np.save(fn, y_p['proba']['test'], allow_pickle=False)
                        fn = (f"{self.save_pred['ID']}_repeat{nexp2}_"
                              f"{scoring}_outer_split{i}_y_pred")
                        for key in params_list[i].keys():
                            fn = f"{fn}_{params_list[i][key]}_"
                        fn = filename_generator(fn, '.npy', directory=self.save_pred['directory'],
                                                timestamp=timestamp)
                        np.save(fn, y_p['pred']['test'], allow_pickle=False)
                        fn = (f"{self.save_pred['ID']}_repeat{nexp2}_"
                              f"{scoring}_outer_split{i}_y_test")
                        for key in params_list[i].keys():
                            fn = f"{fn}_{params_list[i][key]}_"
                        fn = filename_generator(fn, '.npy', directory=self.save_pred['directory'],
                                                timestamp=timestamp)
                        np.save(fn, y_dict['test'], allow_pickle=False)
                    # score
                    for key in performances.keys():
                        for inner_key in performances[key].keys():
                            performances[key][inner_key].append(
                                self._score(
                                    y_dict[key], y_p['pred'][key],
                                    y_p['proba'][key], scoring=inner_key
                                )
                            )

                # if threshold is tuned, plot all specificity-recall-plots together
                if (self.tune_threshold is True and threshold_scoring is not None and
                        isinstance(self.save_tt_plots, dict)):
                    plt.axis([-0.005, 1, 0, 1.005])
                    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
                    plt.yticks(np.arange(0, 1, 0.05))
                    plt.ylabel("Score")
                    plt.xlabel("Decision Threshold")
                    plt.legend(loc='best')
                    plt.grid(True)
                    if threshold_scoring in ['f1', 'f2']:
                        filename = (f"{self.save_tt_plots['ID']}_repeat{nexp2}_outer_split0-"
                                    f"{len(train_indices)-1}_{scoring}_{threshold_scoring}_"
                                    "Precision_Recall_vs_Threshold")
                    elif threshold_scoring in ['balanced_accuracy', 'pseudo_f1',
                                               'pseudo_f2', 'J']:
                        filename = (f"{self.save_tt_plots['ID']}_repeat{nexp2}_outer_split0-"
                                    f"{len(train_indices)-1}_{scoring}_{threshold_scoring}_"
                                    "Recall_Specificity_vs_Threshold")
                    filename = filename_generator(
                        filename, '.pdf', self.save_tt_plots['directory'], timestamp
                    )
                    plt.savefig(filename, dpi=300)
                    # plt.show() mb. needed, if you want to see the figures and not only save them?
                    plt.close()

                performances['ncv'] = dict()
                for pem in self.performance_evaluation_metrics:
                    performances['ncv'][pem] = [self._score(
                        y_D, Predictions, Probabilities, scoring=pem
                    )]

                params_dummy[scoring].extend(params_list)
                assert params_dummy[scoring] == best_inner_params_dict[scoring], \
                    "%s : Parameter collections don't match." % scoring

                rules[scoring].extend(rules_list)

                if (isinstance(baseline_prediction, np.ndarray) or
                        isinstance(baseline_prediction, list)):
                    performances['baseline'] = dict()
                    # calculate the baseline scores on the CV splits
                    baseline_prediction = np.ravel(baseline_prediction)
                    for pem in self.performance_evaluation_metrics:
                        performances['baseline'][pem] = []
                    Baseline_Predictions = []
                    for train, test in zip(train_indices, test_indices):
                        # baseline_prediction_train = baseline_prediction[train]
                        baseline_prediction_test = baseline_prediction[test]
                        Baseline_Predictions.extend(baseline_prediction_test)
                        y_train, y_test = y[train], y[test]
                        for key in performances['baseline'].keys():
                            performances['baseline'][key].append(self._score(
                                y_test, baseline_prediction_test, scoring=key)
                            )
                    for pem in self.performance_evaluation_metrics:
                        performances['baseline'][f'ncv_{pem}'] = [self._score(
                            y_D, Baseline_Predictions, scoring=pem)]

                for key in performances.keys():
                    for inner_key in performances[key].keys():
                        value = performances[key][inner_key]
                        if f'{key}_{inner_key}' in scores[scoring].keys():
                            scores[scoring][f'{key}_{inner_key}'].extend(value)
                        else:
                            scores[scoring][f'{key}_{inner_key}'] = value

            repeated_cv_results_lists.append(repeated_cv_results_list)
            repeated_cv_results_as_dataframes_list.append(repeated_cv_results_as_dataframes)
            # Save all cv_results_ dicts to a single Excel workbook
            if isinstance(self.save_to, dict):
                save_dataframes_to_excel(
                    repeated_cv_results_as_dataframes,
                    self.save_to['directory'],
                    f"{self.save_to['ID']}_RepeatedGridSearchCV_results_repeat{nexp2}",
                    'outer_split',
                    timestamp=timestamp
                )

        if isinstance(self.save_to, dict):
            save_json(
                tuning_curves,
                self.save_to['directory'],
                f"{self.save_to['ID']}_threshold_tuning_curves",
                timestamp=timestamp
            )
        del tuning_curves

        repeated_nested_cv_results: Dict[str, Dict[str, Any]] = {}
        self.best_params_ = {}
        self.ranked_best_inner_params_ = {}
        for scoring, threshold_scoring in zip(scorings, threshold_scorings):
            repeated_nested_cv_results[scoring] = {}
            if self.tune_threshold is True:
                if threshold_scoring in self.threshold_tuning_metrics:
                    repeated_nested_cv_results[scoring][
                        f'best_{threshold_scoring}'] = np.array(
                            thresholds[scoring][f'best_{threshold_scoring}']
                        )
                    repeated_nested_cv_results[scoring][
                        f'mean_best_{threshold_scoring}'] = np.mean(
                            np.array(thresholds[scoring][f'best_{threshold_scoring}'])
                        )
                    if self.Nexp2 > 1:
                        repeated_nested_cv_results[scoring][
                            f'std_best_{threshold_scoring}'] = np.std(
                                np.array(thresholds[scoring][f'best_{threshold_scoring}']), ddof=1
                            )
                    else:
                        repeated_nested_cv_results[scoring][
                            f'std_best_{threshold_scoring}'] = np.std(
                                np.array(thresholds[scoring][f'best_{threshold_scoring}']), ddof=0
                            )
                    repeated_nested_cv_results[scoring][f'min_best_{threshold_scoring}'] = np.amin(
                        np.array(thresholds[scoring][f'best_{threshold_scoring}'])
                    )
                    repeated_nested_cv_results[scoring][f'max_best_{threshold_scoring}'] = np.amax(
                        np.array(thresholds[scoring][f'best_{threshold_scoring}'])
                    )
                    repeated_nested_cv_results[scoring]['best_threshold'] = np.array(
                        thresholds[scoring]['best_thresholds']
                    )
                    repeated_nested_cv_results[scoring][
                        'mean_best_threshold'] = np.mean(
                            np.array(thresholds[scoring]['best_thresholds'])
                        )
                    if self.Nexp2 > 1:
                        repeated_nested_cv_results[scoring]['std_best_threshold'] = np.std(
                            np.array(thresholds[scoring]['best_thresholds']), ddof=1
                        )
                    else:
                        repeated_nested_cv_results[scoring]['std_best_threshold'] = np.std(
                            np.array(thresholds[scoring]['best_thresholds']), ddof=0
                        )
                    repeated_nested_cv_results[scoring]['min_best_threshold'] = np.amin(
                        np.array(thresholds[scoring]['best_thresholds'])
                    )
                    repeated_nested_cv_results[scoring]['max_best_threshold'] = np.amax(
                        np.array(thresholds[scoring]['best_thresholds'])
                    )

            for key in scores[scoring].keys():
                repeated_nested_cv_results[scoring][key] = np.array(
                    scores[scoring][key]
                )
                repeated_nested_cv_results[scoring]['mean_' + key] = np.mean(
                    np.array(scores[scoring][key])
                )
                if self.Nexp2 > 1:
                    repeated_nested_cv_results[scoring]['std_' + key] = np.std(
                        np.array(scores[scoring][key]), ddof=1
                    )
                else:
                    repeated_nested_cv_results[scoring]['std_' + key] = np.std(
                        np.array(scores[scoring][key]), ddof=0
                    )
                repeated_nested_cv_results[scoring]['min_' + key] = np.amin(
                    np.array(scores[scoring][key])
                )
                repeated_nested_cv_results[scoring]['max_' + key] = np.amax(
                    np.array(scores[scoring][key])
                )

            self.best_params_[scoring] = self._score_to_best_params(
                best_inner_params_dict[scoring]
            )
            repeated_nested_cv_results[scoring][
                'best_params'] = self._score_to_best_params(
                    best_inner_params_json_dict[scoring]
                )
            self.ranked_best_inner_params_[scoring] = self._rank_params(
                best_inner_params_dict[scoring]
            )
            repeated_nested_cv_results[scoring][
                'ranked_best_inner_params'] = self._rank_params(
                    best_inner_params_json_dict[scoring]
                )
            repeated_nested_cv_results[scoring][
                'best_inner_indices'] = np.array(best_inner_index_dict[scoring])
            repeated_nested_cv_results[scoring][
                'best_inner_params'] = best_inner_params_json_dict[scoring]
            repeated_nested_cv_results[scoring][
                'best_inner_scores'] = np.array(best_inner_score_dict[scoring])
            if self.collect_rules:
                repeated_nested_cv_results[scoring]['outer_rules'] = rules[scoring]
                repeated_nested_cv_results[scoring][
                    'ranked_outer_rules'] = self._rank_params(rules[scoring])
                flat_list = [item for sublist in rules[scoring] for item in sublist]
                repeated_nested_cv_results[scoring][
                    'ranked_outer_rule_elements'] = self._rank_params(flat_list)
                del flat_list
                rule_lengths = []
                for sublist in rules[scoring]:
                    rule_lengths.append(len(sublist))
                repeated_nested_cv_results[scoring]['mean_outer_rule_length'] = np.mean(
                    np.array(rule_lengths)
                )
                if self.Nexp2 > 1:
                    repeated_nested_cv_results[scoring]['std_outer_rule_length'] = np.std(
                        np.array(rule_lengths), ddof=1
                    )
                else:
                    repeated_nested_cv_results[scoring]['std_outer_rule_length'] = np.std(
                        np.array(rule_lengths), ddof=0
                    )
                del rule_lengths

        self.best_inner_indices_ = best_inner_index_dict
        self.best_inner_scores_ = best_inner_score_dict
        self.best_inner_params_ = best_inner_params_dict
        self.best_thresholds_ = thresholds
        self.repeated_cv_results_lists_ = repeated_cv_results_lists
        self.repeated_cv_results_as_dataframes_list_ = repeated_cv_results_as_dataframes_list
        self.repeated_nested_cv_results_ = repeated_nested_cv_results

        if self.refit:
            if isinstance(self.refit, str):
                # Set best hyperparameters
                self.estimator.set_params(**self.ranked_best_inner_params_[
                    self.refit][0]['parameters'])
                # Fit model with best hyperparameters
                self.estimator.fit(X, y)
                self.best_estimator_ = self.estimator
                if self.tune_threshold is True:
                    for scoring, threshold_scoring in zip(scorings, threshold_scorings):
                        if self.refit == scoring:
                            if threshold_scoring in self.threshold_tuning_metrics:
                                self.mean_best_threshold_ = repeated_nested_cv_results[
                                    self.refit]['mean_best_threshold']
                            else:
                                self.mean_best_threshold_ = None
                else:
                    self.mean_best_threshold_ = None
            else:
                # Set best hyperparameters
                self.estimator.set_params(**self.ranked_best_inner_params_[
                    self.scoring][0]['parameters'])
                # Fit model with best hyperparameters
                self.estimator.fit(X, y)
                self.best_estimator_ = self.estimator
                if self.tune_threshold is True:
                    self.mean_best_threshold_ = repeated_nested_cv_results[
                        self.scoring]['mean_best_threshold']
                else:
                    self.mean_best_threshold_ = None
            if isinstance(self.save_best_estimator, dict):
                if isinstance(self.refit, str):
                    filename = f"{self.save_best_estimator['ID']}_{self.refit}_best_estimator"
                else:
                    filename = f"{self.save_best_estimator['ID']}_{self.scoring}_best_estimator"
                save_model(self.estimator, self.save_best_estimator['directory'], filename,
                           timestamp=timestamp, compress=False, method='joblib')
        else:
            self.best_estimator_ = None
            self.mean_best_threshold_ = None
