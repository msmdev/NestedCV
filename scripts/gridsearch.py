'''
Find the best parameters using x-times repeated nested stratified cross-validation.

Will save results to several .tsv files.
'''
import argparse
import numpy as np
import os.path
import pandas as pd
import pathlib
from pprint import pprint
import re
import sklearn
import sys
import traceback
import warnings
from typing import List
from sklearn.svm import SVC
import nestedcv as ncv


def main(
    ANA_PATH: str,
    DATA_PATH: str,
    drug: str,
) -> None:

    # Generate a timestamp used for file naming
    timestamp = ncv.generate_timestamp()

    print('')
    print("drug:", drug)
    print('Time:', timestamp)
    print('')

    # Create directories for the output files
    maindir = os.path.join(ANA_PATH, 'GridSearchCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)
    rfiledir = os.path.join(ANA_PATH, 'GridSearchCV/results/')
    pathlib.Path(rfiledir).mkdir(parents=True, exist_ok=True)
    efiledir = os.path.join(ANA_PATH, 'GridSearchCV/estimators/')
    pathlib.Path(efiledir).mkdir(parents=True, exist_ok=True)
    pfiledir = os.path.join(ANA_PATH, 'GridSearchCV/plots/')
    pathlib.Path(pfiledir).mkdir(parents=True, exist_ok=True)
    # ifiledir = os.path.join(ANA_PATH, 'GridSearchCV/results/RGSCV/')
    # pathlib.Path(ifiledir).mkdir(parents=True, exist_ok=True)
    # ofiledir = os.path.join(ANA_PATH, 'GridSearchCV/results/RNCV/')
    # pathlib.Path(ofiledir).mkdir(parents=True, exist_ok=True)

    # define identifiers used for naming of output files
    rfile = (drug + '_RGSCV')
    efile = (drug + '_RNCV')
    pfile = (drug + '_RNCV')
    # ifile = (drug + '_RGSCV')
    # ofile = (drug + '_RNCV')

    # set options for NestedGridSearchCV
    cv_options = {
        'scoring': ['average_precision',
                    'brier_loss',
                    'log_loss',
                    'roc_auc',
                    'mcc'],
        'refit': 'average_precision',
        'tune_threshold': True,
        'threshold_tuning_scoring': ['f1', 'f1', 'f1', 'J', None],
        'save_to': {'directory': rfiledir, 'ID': rfile},
        'save_best_estimator': {'directory': efiledir, 'ID': efile},
        'save_pr_plots': {'directory': pfiledir, 'ID': pfile},
        'save_tt_plots': {'directory': pfiledir, 'ID': pfile},
        'save_pred': None,
        'save_inner_to': None,
        'reproducible': True,
        'inner_cv': 5,
        'outer_cv': 5,
        'Nexp1': 5,
        'Nexp2': 10,
        'n_jobs': 28,
    }

    # Set up grid of parameters to optimize over
    Cs: List[float] = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
    degrees: List[int] = [1, 2, 3, 4, 5]
    coef0s: List[int] = [0, 1, 10, 100]
    param_grid = {
        'dataselector__degree': degrees,
        'dataselector__coef0': coef0s,
        'svc__C': Cs,
    }

    print('parameter grid:')
    print(param_grid)
    print('')

    # cachedir = mkdtemp()
    estimator = SVC(
        kernel='poly',
        class_weight='balanced',
        probability=True,
        random_state=1337,
    )

    # load the labels
    y = np.load(os.path.join(DATA_PATH, 'y.npy'))

    print('shape of binary response array:', y.size)
    print('number of positives:', np.sum(y))
    print('number of positives divided by total number of samples:', np.sum(y)/y.size)
    print('')

    # load the (samples x features)-matrix
    X = np.load(os.path.join(DATA_PATH, 'X.npy'))

    clf_grid = ncv.RepeatedStratifiedNestedCV(
        estimator=estimator,
        param_grid=param_grid,
        cv_options=cv_options
    )
    clf_grid.fit(X, y)

    # save the fitted RNCV instance
    if cv_options['refit']:
        if isinstance(cv_options['refit'], str):
            filename = f"{efile}_{cv_options['refit']}_RepeatedStratifiedNestedCV"
        else:
            filename = f"{efile}_{cv_options['scoring']}_RepeatedStratifiedNestedCV"
        ncv.save_model(
            clf_grid, efiledir, filename, timestamp=timestamp, compress=False, method='joblib'
        )

    result = clf_grid.repeated_nested_cv_results_

    try:
        ncv.save_json(
            {key: dict(sorted(value.items())) for key, value in result.items()},
            rfiledir,
            drug + '_repeated_nested_cv_results',
            timestamp=timestamp
        )
    except TypeError:
        warnings.warn('Saving the results as JSON file failed due to a TypeError.')

    print('results:')
    pprint(result, width=200)
    print('')

    performances = []
    ncv_performances = []
    scorings = result.keys()
    for scoring in scorings:

        performance = {}
        performance['best_inner_params'] = result[scoring]['best_inner_params']
        scores = []
        for score in result[scoring].keys():
            if (bool(re.match(r'best', score)) and not
                    bool(re.search(r'ncv|best_inner|best_params', score))):
                performance[score] = result[scoring][score]
            elif (bool(re.match(r'test|train|baseline', score)) and not
                    bool(re.search(r'ncv|best_inner|best_params', score))):
                scores.append(re.sub(r'test_|train_|baseline_', '', score, count=1))
        for score in sorted(scores):
            performance[f'baseline_{score}'] = result[scoring][f'baseline_{score}']
            performance[f'test_{score}'] = result[scoring][f'test_{score}']
            performance[f'train_{score}'] = result[scoring][f'train_{score}']

        assert len(performance['best_inner_params']) % cv_options['Nexp2'] == 0, \
            'len(best_inner_params[%s] modulo Nexp2 != 0 for %s.)' % (drug, scoring)
        index = []
        n_cv_splits = len(performance['best_inner_params']) // cv_options['Nexp2']
        for x2 in range(cv_options['Nexp2']):
            for x1 in range(n_cv_splits):
                index.append('outer-repetition' + str(x2) + '_outer-split' + str(x1))
        performances.append(pd.DataFrame(performance, index=index))

        ncv_performance = {}
        scores = []
        for score in result[scoring].keys():
            if bool(re.match(r'ncv|baseline_ncv', score)):
                scores.append(re.sub(r'ncv_|baseline_ncv_', '', score, count=1))
        for score in sorted(scores):
            ncv_performance[f'baseline_ncv_{score}'] = result[scoring][f'baseline_ncv_{score}']
            ncv_performance[f'ncv_{score}'] = result[scoring][f'ncv_{score}']
        index = []
        for x2 in range(cv_options['Nexp2']):
            index.append('outer-repetition' + str(x2))
        ncv_performances.append(pd.DataFrame(ncv_performance, index=index))

    # save train/test performance scores to an excel file with one sheet per scoring
    fn = ncv.filename_generator(
        f'{drug}_train_test_performance_scores',
        extension=".xlsx",
        directory=rfiledir,
        timestamp=timestamp
    )
    if not os.path.exists(fn):
        with pd.ExcelWriter(fn) as writer:
            for dataframe, scoring in zip(performances, scorings):
                dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
    else:
        warnings.warn(f"Overwriting already existing file {fn}.")
        with pd.ExcelWriter(fn) as writer:
            for dataframe, scoring in zip(performances, scorings):
                dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')

    # save ncv performance scores to an excel file with one sheet per scoring
    fn = ncv.filename_generator(
        f'{drug}_ncv_performance_scores',
        extension=".xlsx",
        directory=rfiledir,
        timestamp=timestamp
    )
    if not os.path.exists(fn):
        with pd.ExcelWriter(fn) as writer:
            for dataframe, scoring in zip(ncv_performances, scorings):
                dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
    else:
        warnings.warn(f"Overwriting already existing file {fn}.")
        with pd.ExcelWriter(fn) as writer:
            for dataframe, scoring in zip(ncv_performances, scorings):
                dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
    del performances, ncv_performances

    # collect and save outer cross-validation train- and test-split performance scores
    for i, scoring in enumerate(scorings):

        performance_mean = {}
        performance_min_max = {}
        performance_min_max_nice = {}
        scores = []
        counter = 0
        threshold_tuning_score = None

        for score in result[scoring].keys():
            if (bool(re.match(r'best', score)) and not
                    bool(re.search(r'ncv|best_inner|best_params', score))):
                if bool(re.fullmatch(r'best_threshold', score)):
                    dummy = score
                else:
                    threshold_tuning_score = re.sub(r'best_', '', score)
                    dummy = 'best_threshold_tuning_score'
                    counter += 1
                if counter > 1:
                    warnings.warn(f'Unexpected "best" score {score}')
                mean = f'mean_{score}'
                std = f'std_{score}'
                min_ = f'min_{score}'
                max_ = f'max_{score}'
                dummy_mean = f'mean_{dummy}'
                dummy_min = f'min_{dummy}'
                dummy_max = f'max_{dummy}'
                dummy_min_max = f'min_max_{dummy}'
                performance_mean[dummy_mean] = [
                    "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                    + " " + "%.6f" % result[scoring][std]
                ]
                performance_min_max[dummy_min] = ["%.3f" % result[scoring][min_]]
                performance_min_max[dummy_max] = ["%.3f" % result[scoring][max_]]
                performance_min_max_nice[dummy_min_max] = [
                    "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                ]
            elif (bool(re.match(r'test|train|baseline', score)) and not
                    bool(re.search(r'ncv|best_inner|best_params', score))):
                scores.append(re.sub(r'test_|train_|baseline_', '', score, count=1))

        if not threshold_tuning_score:
            performance_mean['mean_best_threshold_tuning_score'] = [np.nan]
            performance_min_max['min_best_threshold_tuning_score'] = [np.nan]
            performance_min_max['max_best_threshold_tuning_score'] = [np.nan]
            performance_min_max_nice['min_max_best_threshold_tuning_score'] = ['[nan, nan]']
            performance_mean['mean_best_threshold'] = [np.nan]
            performance_min_max['min_best_threshold'] = [np.nan]
            performance_min_max['max_best_threshold'] = [np.nan]
            performance_min_max_nice['min_max_best_threshold'] = ['[nan, nan]']

        for score in sorted(scores):
            for x in ['baseline', 'test', 'train']:
                mean = f'mean_{x}_{score}'
                std = f'std_{x}_{score}'
                min_ = f'min_{x}_{score}'
                max_ = f'max_{x}_{score}'
                min_max = f'min_max_{x}_{score}'
                if bool(re.search(r'brier_loss', score)):
                    performance_mean[mean] = [
                        "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.5f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.5f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                elif bool(re.search(r'log_loss', score)):
                    performance_mean[mean] = [
                        "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.4f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.4f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                else:
                    performance_mean[mean] = [
                        "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.3f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.3f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                    ]

        print('mean outer cross-validation train- and test-split performance '
              'scores (%s optimized hyperparameters):' % scoring)
        pprint(performance_mean)
        print('')
        print('min max outer cross-validation train- and test-split performance '
              'scores (%s optimized hyperparameters):' % scoring)
        pprint(performance_min_max_nice)
        print('')

        if threshold_tuning_score:
            index = [f'{scoring} | {threshold_tuning_score}']
        else:
            index = [scoring]

        if i == 0:
            performance_mean_df = pd.DataFrame(
                performance_mean, index=index
            )
            performance_min_max_df = pd.DataFrame(
                performance_min_max, index=index
            )
            performance_min_max_nice_df = pd.DataFrame(
                performance_min_max_nice, index=index
            )
        else:
            performance_mean_df = performance_mean_df.append(
                pd.DataFrame(performance_mean, index=index),
                verify_integrity=True
            )
            performance_min_max_df = performance_min_max_df.append(
                pd.DataFrame(performance_min_max, index=index),
                verify_integrity=True
            )
            performance_min_max_nice_df = performance_min_max_nice_df.append(
                pd.DataFrame(performance_min_max_nice, index=index),
                verify_integrity=True
            )

    filename = "collected_mean_train_test_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_mean_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_mean_df.to_excel(fn, na_rep='nan')
    filename = "collected_min_max_train_test_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_min_max_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_min_max_df.to_excel(fn, na_rep='nan')
    filename = "collected_formatted_min_max_train_test_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_min_max_nice_df.to_excel(fn, na_rep='nan')
    del performance_mean, performance_mean_df, performance_min_max, performance_min_max_df
    del performance_min_max_nice, performance_min_max_nice_df

    # collect and save nested cross-validation performance scores
    for i, scoring in enumerate(scorings):

        performance_mean = {}
        performance_min_max = {}
        performance_min_max_nice = {}
        scores = []
        for score in result[scoring].keys():
            if bool(re.match(r'ncv|baseline_ncv', score)):
                scores.append(re.sub(r'ncv_|baseline_ncv_', '', score, count=1))
        for score in sorted(scores):
            for x in ['baseline_ncv', 'ncv']:
                mean = f'mean_{x}_' + score
                std = f'std_{x}_' + score
                min_ = f'min_{x}_' + score
                max_ = f'max_{x}_' + score
                min_max = f'min_max_{x}_' + score
                if bool(re.search(r'brier_loss', score)):
                    performance_mean[mean] = [
                        "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.5f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.5f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                elif bool(re.search(r'log_loss', score)):
                    performance_mean[mean] = [
                        "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.4f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.4f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                else:
                    performance_mean[mean] = [
                        "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max[min_] = ["%.3f" % result[scoring][min_]]
                    performance_min_max[max_] = ["%.3f" % result[scoring][max_]]
                    performance_min_max_nice[min_max] = [
                        "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                    ]

        print('mean nested cross-validation performance scores '
              '(%s optimized hyperparameters):' % scoring)
        pprint(performance_mean)
        print('')
        print('min max nested cross-validation performance scores '
              '(%s optimized hyperparameters):' % scoring)
        pprint(performance_min_max_nice)
        print('')

        if i == 0:
            performance_mean_df = pd.DataFrame(
                performance_mean, index=[scoring]
            )
            performance_min_max_df = pd.DataFrame(
                performance_min_max, index=[scoring]
            )
            performance_min_max_nice_df = pd.DataFrame(
                performance_min_max_nice, index=[scoring]
            )
        else:
            performance_mean_df = performance_mean_df.append(
                pd.DataFrame(performance_mean, index=[scoring]),
                verify_integrity=True
            )
            performance_min_max_df = performance_min_max_df.append(
                pd.DataFrame(performance_min_max, index=[scoring]),
                verify_integrity=True
            )
            performance_min_max_nice_df = performance_min_max_nice_df.append(
                pd.DataFrame(performance_min_max_nice, index=[scoring]),
                verify_integrity=True
            )

    filename = "collected_mean_ncv_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_mean_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_mean_df.to_excel(fn, na_rep='nan')
    filename = "collected_min_max_ncv_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_min_max_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_min_max_df.to_excel(fn, na_rep='nan')
    filename = "collected_formatted_min_max_ncv_performance_scores"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    performance_min_max_nice_df.to_excel(fn, na_rep='nan')
    del performance_mean, performance_mean_df, performance_min_max, performance_min_max_df
    del performance_min_max_nice, performance_min_max_nice_df

    # compact collected printout
    for scoring in scorings:
        performance_mean = {}
        for score in result[scoring].keys():
            if bool(re.match(r'ncv|baseline_ncv', score)):
                mean = 'mean_' + score
                std = 'std_' + score
                performance_mean[mean] = [
                    "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                    + " " + "%.6f" % result[scoring][std]
                ]
        print('mean nested cross-validation performance scores '
              '(%s optimized hyperparameters):' % scoring)
        pprint(performance_mean)
        print('')

    print('End:', ncv.generate_timestamp())


if __name__ == "__main__":
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('Start:', ncv.generate_timestamp())

    timestamp = ncv.generate_timestamp()
    warning_file = open(f"warnings_{timestamp}.log", "w")

    def warn_with_traceback(
        message, category, filename, lineno, file=None, line=None
    ):
        try:
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=warning_file)
        except Exception:
            sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=sys.stderr)

    warnings.showwarning = warn_with_traceback
    warnings.simplefilter("default")

    parser = argparse.ArgumentParser(
        description=('Function to run nested cross-validated grid-search for OligoSVM models')
    )
    parser.add_argument(
        '--analysis-dir', dest='analysis_dir', metavar='DIR', required=True,
        help='Path to the directory were the analysis shall be performed and stored.'
    )
    parser.add_argument(
        '--data-dir', dest='data_dir', metavar='DIR', required=True,
        help='Path to the directory were the data is located.'
    )
    parser.add_argument(
        '--drug', dest='drug', required=True,
        help='Drug to train resistance prediction model for. Supply one string.'
    )
    args = parser.parse_args()

    try:
        main(
            args.analysis_dir,
            args.data_dir,
            args.drug,
        )
    finally:

        warning_file.close()
