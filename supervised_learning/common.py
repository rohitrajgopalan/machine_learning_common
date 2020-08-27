import enum
import random
import warnings
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression, ElasticNet, SGDRegressor, \
    HuberRegressor
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import RobustScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings('ignore')
classifiers = {'Logistic Regression': LogisticRegression(n_jobs=-1),
               'Decision Tree': DecisionTreeClassifier(),
               'Support Vector Machine': SVC(gamma='auto', kernel='rbf'),
               'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
               'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
               'Random Forest': RandomForestClassifier(n_jobs=-1),
               'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
               'Bayes': GaussianNB(),
               'Neural Network': MLPClassifier()}
regressors = {'Linear Regression': LinearRegression(n_jobs=-1),
              'Decision Tree': DecisionTreeRegressor(),
              'Random Forest': RandomForestRegressor(n_jobs=-1),
              'K-Nearest Neighbour': KNeighborsRegressor(n_jobs=-1),
              'Lasso': Lasso(),
              'Ridge': Ridge(),
              'Elastic Net': ElasticNet(),
              'Neural Network': MLPRegressor(),
              'Radius Neighbours': RadiusNeighborsRegressor(n_jobs=-1),
              'SGD': SGDRegressor(),
              'Huber': HuberRegressor()}
scoring_classifiers = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc']
scoring_regressors = ['explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                      'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
metrics_classifiers = ['Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1 score',
                       'Area underneath the curve']
metrics_regressors = ['Explained Variance', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error',
                      'Root Mean Squared Error',
                      'Median Absolute Error', 'R2 Score']


class MethodType(enum.Enum):
    Classification = 1
    Regression = 2


scaler = RobustScaler()
normalizer = Normalizer()


def train_test_split_from_data(df_from_each_file, enable_scaling=True, num_test_files=0, test_size=0,
                               enable_normalization=False):
    if num_test_files > 0:
        training_sets = df_from_each_file[:len(df_from_each_file) - num_test_files]
        test_sets = df_from_each_file[len(df_from_each_file) - num_test_files:]
        training = pd.concat(training_sets, ignore_index=True)
        test = pd.concat(test_sets, ignore_index=True)

        cols = [col for col in training.columns]
        features = cols[:len(cols) - 1]
        label = cols[-1]

        train_x = np.array([training[features]]).reshape(-1, len(features))
        train_y = np.array(training[label]).reshape(-1, 1)
        test_x = np.array([test[features]]).reshape(-1, len(features))
        test_y = np.array(test[label]).reshape(-1, 1)

    else:
        df = pd.concat(df_from_each_file, ignore_index=True)
        cols = [col for col in df.columns]
        features = cols[:len(cols) - 1]
        label = cols[-1]
        x = np.array([df[features]]).reshape(-1, len(features))
        y = df[label]
        if test_size > 0:
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
        else:
            train_x = x
            train_y = y
            test_x = None
            test_y = None

    if enable_scaling:
        train_x = scaler.fit_transform(train_x)
        if test_x is not None:
            test_x = scaler.transform(test_x)
    if enable_normalization:
        train_x = normalizer.fit_transform(train_x)
        if test_x is not None:
            test_x = normalizer.transform(test_x)

    return train_x, train_y, test_x, test_y


def load_from_directory(files_dir, cols=[], concat=False, sheet_name='', header_index=0, cols_to_types={}):
    data_files = [join(files_dir, f) for f in listdir(files_dir) if
                  isfile(join(files_dir, f))]
    df_from_each_file = []
    for f in data_files:
        df = None
        if f.endswith(".csv"):
            if not bool(cols_to_types):
                df = pd.read_csv(f, usecols=cols, dtype=cols_to_types)
            else:
                df = pd.read_csv(f, usecols=cols)
        elif f.endswith(".xls") or f.endswith(".xlsx"):
            if not bool(cols_to_types):
                if len(sheet_name) > 0:
                    df = pd.read_excel(f, sheet_name=sheet_name, header=header_index, usecols=cols, dtype=cols_to_types)
                else:
                    df = pd.read_excel(f, usecols=cols, dtype=cols_to_types)
            else:
                if len(sheet_name) > 0:
                    df = pd.read_excel(f, sheet_name=sheet_name, header=header_index, usecols=cols)
                else:
                    df = pd.read_excel(f, usecols=cols)
        df = df.dropna()
        if df is None:
            continue
        df_from_each_file.append(df)

    return pd.concat(df_from_each_file, ignore_index=True) if concat else df_from_each_file


def perform_experiment_on_data(df_from_each_file, method_type, enable_scaling=True, enable_normalization=True):
    results = {}
    test_sizes = []
    max_num_test_files = int(len(df_from_each_file) / 5)
    if max_num_test_files >= 1:
        for num_test_files in range(1, max_num_test_files + 1):
            test_sizes.append(num_test_files)
            results[num_test_files] = run_with_different_methods(method_type, df_from_each_file, enable_scaling,
                                                                 num_test_files=num_test_files,
                                                                 enable_normalization=enable_normalization)
    else:
        for test_size in list(np.arange(0.01, 0.2, 0.01)):
            test_sizes.append(test_size)
            results[test_size] = run_with_different_methods(method_type, df_from_each_file, enable_scaling,
                                                            test_size=test_size,
                                                            enable_normalization=enable_normalization)
    return results, test_sizes


def run_with_different_methods(method_type, df_from_each_file, enable_scaling=True, num_test_files=0, test_size=0,
                               enable_normalization=True):
    x_train, y_train, x_test, y_test = train_test_split_from_data(df_from_each_file, enable_scaling, num_test_files,
                                                                  test_size, enable_normalization)
    methods = classifiers if method_type == MethodType.Classification else regressors
    metrics = metrics_classifiers if method_type == MethodType.Classification else metrics_regressors
    scoring = scoring_classifiers if method_type == MethodType.Classification else scoring_regressors
    cv = 5 if method_type == MethodType.Classification else 10
    models_data = {'Model': list(methods.keys()),
                   'Enable Scaling': ['Yes' if enable_scaling else 'No'] * len(methods.keys()),
                   'Enable Normalization': ['Yes' if enable_normalization else 'No'] * len(methods.keys()),
                   'Fitting Time': [],
                   'Scoring Time': []}
    for metric in metrics:
        models_data.update({metric: []})
    for key in methods:
        scores = cross_validate(methods[key], x_train, y_train, scoring=scoring, cv=cv)
        sorted(scores.keys())
        models_data['Fitting Time'].append(scores['fit_time'].mean())
        models_data['Scoring Time'].append(scores['score_time'].mean())
        for i, metric in enumerate(metrics):
            average_value = scores['test_{0}'.format(scoring[i])].mean()
            if average_value < 0:
                average_value *= -1
            models_data[metric].append(average_value)
    return models_data


def shape_experimental_data_for_plotting(results, test_sizes, methods, metrics):
    metrics_to_data = {}
    method_names = list(methods.keys())
    for metric in metrics:
        data_for_metric = []
        for i in range(len(method_names)):
            data_for_metric.append([])
        for test_size in test_sizes:
            metric_data = results[test_size][metric]
            for i in range(len(data_for_metric)):
                data_for_metric[i].append(metric_data[i])
        metrics_to_data[metric] = data_for_metric
    return metrics_to_data, test_sizes


def select_method(choosing_method, method_type, use_grid_search=True):
    chosen_method = None
    methods = classifiers if method_type == MethodType.Classification else regressors

    if choosing_method == 'random':
        chosen_method = randomly_select_method(methods)
    else:
        for method_name in methods.keys():
            if method_name.lower() == choosing_method.lower():
                chosen_method = methods[method_name]
                break
    if use_grid_search:
        params = get_testable_parameters(chosen_method)
        return set_up_gridsearch(chosen_method, params, method_type)
    else:
        return chosen_method

def randomly_select_method(methods):
    key = random.choice(list(methods.keys()))
    return methods[key]


def set_up_gridsearch(method, params, method_type):
    if not bool(params):
        return GridSearchCV(method, param_grid=params, cv=5, scoring='neg_mean_squared_error' if method_type == MethodType.Regression else 'accuracy', verbose=1, n_jobs=-1)
    else:
        return method


def get_testable_parameters(method_name):
    if method_name == 'Linear Regression':
        return {'normalize': [True, False]}
    elif method_name in ['Decision Tree', 'Random Forest']:
        return {'max_features': ['auto', 'log2', 'sqrt']}
    elif method_name in ['Lasso', 'Ridge']:
        return {'normalize': [True, False],
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4]}
    elif method_name == 'Huber':
        return {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
    elif method_name == 'Elastic Net':
        return {'normalize': [True, False],
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'l1_ratio': list(np.arange(0, 1, 0.05))}
    elif method_name == 'Neural Network':
        return {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate': ['adaptive', 'invscaling', 'constant'],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [(400,), (400, 400), (400, 400, 400), (400, 400, 400, 400)]}
    elif method_name in ['K-Nearest Neighbour', 'Radius Neighbours']:
        return {'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    elif method_name == 'SGD':
        return {'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'l1_ratio': list(np.arange(0, 1, 0.05)),
                'learning_rate': ['adaptive', 'invscaling', 'constant', 'optimal']}
    else:
        return {}
