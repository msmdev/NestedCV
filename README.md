# NestedCV: repeated nested stratified cross-validation

This package implements a method to perform repeated stratified nested cross-validation for any estimator that implements the scikit-learn estimator interface.
The method is based on Algorithm 2 from [[Krstajic et al., 2014]](https://doi.org/10.1186/1758-2946-6-10).
Simple repeated grid-search cross-validation (Algorithm 1, [[Krstajic et al., 2014]](https://doi.org/10.1186/1758-2946-6-10)) is supported as well via the `RepeatedGridSearchCV` class.

## Installation:

You can install the package using pip via (note: you must be inside the package folder):
```bash
pip install .
```

## Developer mode:

Instead you can install in "develop" or "editable" mode using pip
```bash
pip install --editable .
```
This puts a link into the python installation to your code, such, that your package is installed but any changes will immediately take effect.
At the same time, all your client code can import your package the usual way.

## A word of warning:

Installing as described above may replace packages in your environment with package version incompatible to other packages in your environment. Things will break, if this happens. Thus, think, be cautious, and ideally use disposable environments when installing a package.
To be on the save side (better save than sorry...) you should create an new empty conda environment with a matching version of python (python 3.7.9 is recommended)
```bash
conda create -n nestedcv_env python=3.7.9
```
and perform the install as described above inside of this environment.

## Running tests

A set of basic tests can be run by invoking pytest inside of the repository:
```bash
pytest -v
```

## API

The most important class is `RepeatedStratifiedNestedCV`, which provides everything needed to perform repeated stratified nested cross-validation for any estimator that implements the scikit-learn estimator interface.

### RepeatedStratifiedNestedCV Parameters

| Name        | type           | description  |
| :------------- |:-------------| :-----|
| estimator | estimator object | This is assumed to implement the scikit-learn estimator interface. |
| params_grid | dict or list of dicts | Dictionary with parameters names (str) as keys and lists of parameter values to iterate over during grid-search. |
| cv_options | dict, default={} | Nested cross-validation options (see [Next section](#cv_options-value-options)). |

### `cv_options` value options

**'collect_rules' :** bool, default=False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only available, if the estimator is ``RuleBasedClassifier`` (RBC) with ``r_solution_`` attribute. If set to ``True``, the rules (disjunctions) learned by RBC during the outer CV are collected and ranked. CAUTION: Please be aware that if you pass a pipeline as estimator the RBC must be the last step in the pipeline. Otherwise, the rules can't be collected.

**'inner_cv' :** int or cross-validatior (a.k.a. CV splitter), default=``StratifiedKFold(n_splits=5, shuffle=True)``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Determines the inner cross-validation splitting strategy. Possible inputs for ``'inner_cv'`` are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- integer, to specify the number of folds in a ``StratifiedKFold``,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- CV splitter,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- a list of CV splitters of length Nexp1.

**'n_jobs' :** int or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of jobs of ``RepeatedGridSearchCV`` to run in parallel. ``None`` means ``1`` while ``-1`` means using all processors.

**'Nexp1' :** int, default=10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of inner CV repetitions (for hyper-parameter search).

**'Nexp2' :** int, default=10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of nested CV repetitions.

**'outer_cv' :** int or cross-validatior (a.k.a. CV splitter), default=``StratifiedKFold(n_splits=5, shuffle=True)``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Determines the outer cross-validation splitting strategy. Possible inputs for ``'outer_cv'`` are:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- integer, to specify the number of folds in a ``StratifiedKFold``,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- CV splitter,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- a list of CV splitters of length Nexp2.

**'refit' :** bool or str, default=False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Refit an estimator using the best found parameters (rank 1) on the whole dataset. For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters for refitting the estimator at the end. The refitted estimator is made available at the ``best_estimator_`` attribute.

**'reproducible' :** bool, default=False

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If True, the CV splits will become reproducible by setting ``cv_options['outer_cv']=StratifiedKFold(n_splits, shuffle=True, random_state=nexp2)``, ``cv_options['inner_cv']=StratifiedKFold(n_splits, shuffle=True, random_state=nexp1)`` with ``nexp2``, ``nexp1`` being the current iteration of the outer/inner repetition loop and ``n_splits`` as given in via the ``'outer_cv'`` and ``'inner_cv'`` key.

**'save_best_estimator' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If not ``None``, the best estimator (using the best found parameters) refitted on the whole dataset will be saved. This requires ``cv_options['refit']`` set to ``True`` or a str denoting the scorer used to find the best parameters for refitting, in case of multiple metric evaluation. Pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and name of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.joblib).

**'save_inner_to' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If not ``None``, the train and test indices for every inner split and ``y_proba`` and ``y_test`` for every point of the parameter grid of the inner cross-validated grid search will be saved. Pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and name of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.npy).

**'save_pr_plots' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only used, if ``cv_options['tune_threshold']=True``. Determines, if the Precision-Recall-Curve and a plot of Precision and Recall over the decision threshold (if ``cv_options['threshold_tuning_scoring']`` is set to ``'f1'`` or ``'f2'``) or the ROC-Curve and a plot of Sensitivity and Specificity over the decision threshold (if ``cv_options['threshold_tuning_scoring']`` is set to ``'balanced_accurracy'``, ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``) shall be saved for every outer CV split. If None, no plots will be saved. To save every plot as individual PDF, pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and name of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.pdf).

**'save_pred' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If not ``None``, the train and test indices for every outer split and ``y_proba``, ``y_pred`` and ``y_test`` for every repetition and split of the outer repeated cross-validation will be saved. Pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and name of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.npy).

**'save_to' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If not ``None``, the results of all inner cross-validated Grid-Search iterations per outer iteration will be compiled all together in a single Excel workbook with one sheet per outer split and one row per inner iteration. Per outer (nested) iteration, one Excel workbook will be stored. Additionally, if not ``None``, a dict containing all threshold tuning curves, i.e.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- precision and recall over varying threshold, if ``cv_options['threshold_tuning_scoring']`` is set to ``'f1'`` or ``'f2'`` or

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- fpr and tpr over varying threshold, if ``cv_options['threshold_tuning_scoring']`` is set to ``'balanced_accuracy'``, ``'J'``, ``'pseudo_f1'``, or ``'pseudo_f2'``,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is saved in a .json file. Pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and identifier of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.xlsx).

**'save_tt_plots' :** dict or None, default=None

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only used, if ``cv_options['tune_threshold']=True``. Determines, if the Precision-Recall-Curve and a plot of Precision and Recall over the decision threshold (if ``cv_options['threshold_tuning_scoring']`` is set to ``'f1'`` or ``'f2'``) or the ROC-Curve and a plot of Sensitivity and Specificity over the decision threshold (if ``cv_options['threshold_tuning_scoring']`` is set to ``'balanced_accurracy'``, ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``) for every outer CV split, compiled together into one PDF per nested CV repetition, shall be saved. If ``None``, no plots will be saved. To save them, pass a dict ``{'directory': str, 'ID': str}`` with two keys denoting the location and name of output files. The value of the key ``'directory'`` is a single string indicating the location were the file will be saved. The value of the key ``'ID'`` is a single string used as part of the name of the output files. A string indicating the current date and time (formated like this: 26.03.2020_17-52-20) will be appended before the file extension (.pdf).

**'scoring' :** str or list, default='precision_recall_auc'

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Can be a string or a list with elements out of ``'balanced_accuracy'``, ``'brier_loss'``, ``'f1'``, ``'f2'``, ``'log_loss'``, ``'mcc'``, ``'pseudo_f1'``, ``'pseudo_f2'``, ``'sensitivity'``, ``'average_precision'``, ``'precision_recall_auc'`` or ``'roc_auc'``. CAUTION: If a list is given, all elements must be unique. If ``'balanced_accuracy'``, ``'f1'``, ``'f2'``, ``'mcc'``, ``'pseudo_f1'``, ``'pseudo_f2'`` or ``'sensitivity'`` the estimator must support the ``predict`` method for predicting binary classes. If ``'average_precision'``, ``'brier_loss'``, ``'log_loss'``, ``'precision_recall_auc'`` or ``'roc_auc'`` the estimator must support the ``predict_proba`` method for predicting binary class probabilities in [0,1].

**'threshold_tuning_scoring'** : str or list, default='f2'

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only used, if ``cv_options['tune_threshold']=True``. If a single metric is chosen as scoring (e.g. ``cv_options['scoring']='mcc'``), it can be one out of ``'balanced_accurracy'``, ``'f1'``, ``'f2'``, ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'``. If multiple metrics are chosen (e.g. ``cv_options['scoring']=['f1', 'mcc']``), it can be one of ``'balanced_accurracy'``, ``'f1'``, ``'f2'``, ``'J'``, ``'pseudo_f1'`` or ``'pseudo_f2'`` (to perform the same threshold tuning method for all metrics) or a list of according length with elements out of ``['balanced_accurracy', 'f1', 'f2', 'J', 'pseudo_f1', 'pseudo_f2', None]`` to perform different threshold tuning methods for every metric: E.g., specifying ``cv_options['threshold_tuning_scoring']=['J', None]``, while specifying ``cv_options['scoring']=['roc_auc', 'mcc']``, implies tuning the decision threshold by selecting the optimal threshold from the ROC curve by choosing the threshold with the maximum value of Youden's J statistic after performing hyperparameter optimization using ``'roc_auc'`` as scoring, while performing no threshold tuning after hyperparameter optimization using ``'mcc'`` as scoring. For backward compartibility only, ``'precision_recall_auc'`` or ``'roc_auc'`` are supported options, but shouldn't be used otherwise. If choosing ``'precision_recall_auc'``, the optimal threshold will be selected from the Precision-Recall curve by choosing the threshold with the maximum value of the F-beta score with ``beta=2``. If ``'roc_auc'`` is chosen, the optimal threshold will be selected from the Precision-Recall curve by choosing the threshold with the maximum value of Youden's J statistic. CAUTION: Please keep in mind that the scoring metric used during threshold tuning should harmonize with the scoring metric used to select the optimal hyperparameters during grid search. E.g., choosing ``cv_options['scoring']='roc_auc'``, it would be a sensible choice to set ``cv_options['threshold_tuning_scoring']`` to one out of ``'balanced_accurracy'``, ``'pseudo_f1'``, ``'pseudo_f2'`` or ``'J'``. Choosing ``cv_options['scoring']='precision_recall_auc'``, it would be sensible to set ``cv_options['threshold_tuning_scoring']`` to either ``'f1'`` or ``'f2'``.

**'tune_threshold' :** bool, default=True

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If ``True``, perform threshold tuning on each outer training fold for the estimator with the best hyperparameters (found by tuning of these in the inner cross validation) retrained on the outer training fold. A list of thresholds is returned that maximize the scoring metric selected via ``cv_options['threshold_tuning_scoring']`` for the best estimators (with optimal hyperparameters) on each training fold. CAUTION: The estimator must support the ``predict_proba`` method for predicting binary class probabilities in [0,1] to tune the decision threshold.

### Attributes

**`best_inner_scores_` :** Best inner scores for each outer loop cumulated in a dict.

**`best_inner_indices_` :** Best inner indices cumulated in a dict.

**`best_params_` :** All best params from the inner loop cumulated in a dict.

**`ranked_best_inner_params_` :** Ranked (most frequent first) best inner params as a list of dictionaries (every dict, i.e. parameter combination, occurs only once).

**`best_inner_params_` :** Best inner params for each outer loop as a list of dictionaries.

**`best_thresholds_` :** If cv_option['tune_threshold']=True, this is a dict containing the best thresholds and threshold-tuning-scorings for each scoring-threshold_tuning_scoring-pair. Given as a dict with scorings as keys and dicts as values. Each of these dicts given as ``{'best_<threshold_tuning_scoring>': list, 'best_theshold': list}`` contains a list of thresholds, which maximize the threshold tuning scoring for the best estimators on the outer training folds, and a list of the respective maximum values of the threshold-tuning-scoring as values. Else the dict is empty.

**`best_estimator_` :** If ``cv_options['refit']`` is set to ``True`` or a str denoting a valid scoring metric, this gives the best estimator refitted on the whole dataset using the best parameters (rank 1) found during grid search. For multiple metric evaluation, the scoring given via ``cv_options['refit']`` is used to find the best parameters for refitting the estimator at the end.

**`mean_best_threshold_` :** If ``cv_options['refit']`` is set to True or a str denoting a valid scoring metric, this gives the mean of the best thresholds, which maximize the threshold tuning scoring on the outer training folds for the best parameters (rank 1) found during grid search. For multiple metric evaluation, the scoring given via ``cv_options['refit']`` is used to find the best parameters.

**`repeated_cv_results_lists_` :** List of ``Nexp2`` lists of ``n_split_outer`` dicts. Each dict contains the compiled results of ``Nexp1`` iterations of ``n_split_inner``-fold cross-validated grid searches with keys as column headers and values as columns. Each of those dicts can be imported into a pandas DataFrame.

**`repeated_cv_results_as_dataframes_list_` :** List of ``Nexp2`` lists of ``n_split_outer`` pandas dataframes containing the compiled results of of ``Nexp1`` iterations of ``n_split_inner``-fold cross-validated grid searches.

**`repeated_nested_cv_results_` :** A list of ``Nexp2`` dicts of compiled nested CV results.


**`best_inner_scores_` :** Best inner scores for each outer loop cumulated in a dict.

**`best_inner_indices_` :** Best inner indices cumulated in a dict.

**`best_params_` :** All best params from the inner loop cumulated in a dict.

**`ranked_best_inner_params_` :** Ranked (most frequent first) best inner params as a list of dicts (every dict, i.e. parameter combination, occurs only once).

**`best_inner_params_` :** Best inner params for each outer loop as a list of dicts.

**`best_thresholds_` :** If ``cv_option['tune_threshold']=True``, this is a dict containing the best thresholds and threshold-tuning-scorings for each scoring-threshold_tuning_scoring-pair. Given as a dict with scorings as keys and dicts as values. Each of these dicts given as ``{'best_<threshold_tuning_scoring>': list, 'best_theshold': list}`` contains a list of thresholds, which maximize the threshold tuning scoring for the best estimators on the outer training folds, and a list of the respective maximum values of the threshold-tuning-scoring as values. Else the dict is empty.

**`best_estimator_` :** If ``cv_options['refit']`` is set to ``True`` or a str denoting a valid scoring metric, this gives the best estimator refitted on the whole dataset using the best parameters (rank 1) found during grid search. For multiple metric evaluation, the scoring given via ``cv_options['refit']`` is used to find the best parameters for refitting the estimator at the end.

**`mean_best_threshold_` :** If ``cv_options['refit']`` is set to True or a str denoting a valid scoring metric, this gives the mean of the best thresholds, which maximize the threshold tuning scoring on the outer training folds for the best parameters (rank 1) found during grid search. For multiple metric evaluation, the scoring given via ``cv_options['refit']`` is used to find the best parameters.

**`repeated_cv_results_lists_` :** List of ``Nexp2`` lists of ``n_split_outer`` dicts. Each dict contains the compiled results of ``Nexp1`` iterations of ``n_split_inner``-fold cross-validated grid searches with keys as column headers and values as columns. Each of those dicts can be imported into a pandas DataFrame.

**`repeated_cv_results_as_dataframes_list_` :** List of ``Nexp2`` lists of ``n_split_outer`` pandas dataframes containing the compiled results of of ``Nexp1`` iterations of ``n_split_inner``-fold cross-validated grid searches.

**`repeated_nested_cv_results_` :** A list of ``Nexp2`` dicts of compiled nested CV results.


### Methods

**`fit(self, X, y[, baseline_prediction, groups])` :** A method to run repeated nested stratified cross-validated grid-search.

**`predict(self, X)` :** Call predict on the estimator with the best found parameters. Only available if ``cv_options['refit']=True`` and the underlying estimator supports ``predict``. If threshold tuning was applied, the mean of the best thresholds, which maximize the threshold tuning scoring on the outer training folds for the best parameters (rank 1) found during grid search, is used as decision threshold for class prediction.

**`predict_proba(self, X)` :** Call predict_proba on the estimator with the best found parameters. Only available if ``cv_options['refit']=True`` and the underlyingestimator supports ``predict_proba``.


## Usage

Be mindful of the options that are available for `RepeatedStratifiedNestedCV`. Most cross-validation options are defined in a dictionary `cv_options`.
This package is optimized for any `sklearn` estimator. It should also be possible to use it with estimators that implement a scikit-learn wrapper, e.g., `XGBoost`, `LightGBM`, `KerasRegressor`, `KerasClassifier` etc.

### Example

Here is a simple example using a Random Forest classifier.
```python
from nestedcv import RepeatedStratifiedNestedCV
from sklearn.ensemble import RandomForestClassifier

# Define a parameters grid
param_grid = {
     'max_depth': [3, None],
     'n_estimators': [100, 1000]
}

ncv = RepeatedStratifiedNestedCV(
    estimator=RandomForestRegressor(),
    params_grid=param_grid,
)
ncv.fit(X=X,y=y)
ncv.outer_scores
```

## Limitations

Currently only data with binary labels is supported (no multiclass support).

## Why should you use nested cross-validation?

Controlling the bias-variance trade-off is an essential and important task in machine learning, as indicated by [[Cawley and Talbot, 2010]](http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf). Many articles state that this can be archieved by the use of nested cross-validation, i.e., [Varma and Simon, 2006](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/pdf/1471-2105-7-91.pdf). Futher sources worth reading are esp. [[Krstajic et al., 2014]](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10) and [[Varoquaox et al., 2017]](https://arxiv.org/pdf/1606.05201.pdf).
