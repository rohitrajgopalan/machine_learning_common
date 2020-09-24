import enum
import random
import warnings
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor, \
    RidgeClassifier, SGDClassifier, LinearRegression, HuberRegressor, ElasticNet, Lasso
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor, \
    RadiusNeighborsClassifier
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.svm import SVR, LinearSVR, NuSVR, SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

warnings.filterwarnings('ignore')
method_names = ['Decision Tree', 'Random Forest', 'K-Nearest Neighbours', 'Radius Neighbours', 'Ridge', 'SGD', 'Nu-SVM',
                'Linear-SVM', 'SVM']
classifiers = {'Logistic Regression': LogisticRegression(),
               'Decision Tree': DecisionTreeClassifier(),
               'Random Forest': RandomForestClassifier(n_jobs=-1),
               'K-Nearest Neighbours': KNeighborsClassifier(n_jobs=-1),
               'Radius Neighbours': RadiusNeighborsClassifier(n_jobs=-1),
               'Ridge': RidgeClassifier(),
               'SGD': SGDClassifier(),
               'Nu-SVM': NuSVC(),
               'Linear-SVM': LinearSVC(),
               'SVM': SVC()}
regressors = {'Linear Regression': LinearRegression(n_jobs=-1),
              'Decision Tree': DecisionTreeRegressor(),
              'Random Forest': RandomForestRegressor(n_jobs=-1),
              'K-Nearest Neighbours': KNeighborsRegressor(n_jobs=-1),
              'Ridge': Ridge(),
              'Radius Neighbours': RadiusNeighborsRegressor(n_jobs=-1),
              'SGD': SGDRegressor(),
              'Nu-SVM': NuSVR(),
              'Linear-SVM': LinearSVR(),
              'SVM': SVR(),
              'Huber': HuberRegressor(),
              'Elastic Net': ElasticNet(),
              'Lasso': Lasso()}
scoring_classifiers = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc']
scoring_regressors = ['explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                      'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
metrics_classifiers = ['Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1 score',
                       'Area underneath the curve']
metrics_regressors = ['Explained Variance', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error',
                      'Root Mean Squared Error',
                      'Median Absolute Error', 'R2 Score']


class ScalingType(enum.Enum):
    STANDARD = 1,
    ROBUST = 2,
    NONE = 3,

    @staticmethod
    def all():
        return [ScalingType.NONE, ScalingType.STANDARD, ScalingType.ROBUST]

    @staticmethod
    def get_type_by_name(name):
        for scaling_type in ScalingType.all():
            if scaling_type.name.lower() == name.lower():
                return scaling_type

        return None


class MethodType(enum.Enum):
    Classification = 1
    Regression = 2


def get_scaler_by_type(scaling_type):
    if scaling_type == ScalingType.STANDARD:
        return StandardScaler()
    elif scaling_type == ScalingType.ROBUST:
        return RobustScaler()
    else:
        return None


normalizer = Normalizer()


def train_test_split_from_data(df_from_each_file, scaling_type=ScalingType.NONE, enable_normalization=False, **args):
    scaler = get_scaler_by_type(scaling_type)
    if 'num_test_files' in args:
        num_test_files = args['num_test_files']
        training_sets = df_from_each_file[:len(df_from_each_file) - num_test_files]
        test_sets = df_from_each_file[len(df_from_each_file) - num_test_files:]
        training = pd.concat(training_sets, ignore_index=True)
        test = pd.concat(test_sets, ignore_index=True)

        cols = [col for col in training.columns]
        features = cols[:len(cols) - 1]
        label = cols[-1]

        train_x = training[features]
        train_y = training[label]
        test_x = test[features]
        test_y = test[label]

    else:
        test_size = args['test_size']
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

    if scaler is not None:
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


def perform_experiment_on_data(df_from_each_file, method_type, scaling_type=ScalingType.NONE,
                               enable_normalization=True):
    results = {}
    test_sizes = []
    max_num_test_files = int(len(df_from_each_file) / 5)
    if max_num_test_files >= 1:
        for num_test_files in range(1, max_num_test_files + 1):
            test_sizes.append(num_test_files)
            results[num_test_files] = run_with_different_methods(method_type, df_from_each_file, scaling_type,
                                                                 num_test_files=num_test_files,
                                                                 enable_normalization=enable_normalization)
    else:
        for test_size in list(np.arange(0.01, 0.2, 0.01)):
            test_sizes.append(test_size)
            results[test_size] = run_with_different_methods(method_type, df_from_each_file, scaling_type,
                                                            test_size=test_size,
                                                            enable_normalization=enable_normalization)
    return results, test_sizes


def get_average_score(model_name, df_from_each_file, scaling_type=ScalingType.NONE, enable_normalization=True,
                      use_grid_search=False, method_type=MethodType.Regression):
    max_num_test_files = int(len(df_from_each_file) / 5)
    model = select_method(model_name, method_type, use_grid_search)
    scores = np.zeros(max_num_test_files)
    for num_test_files in range(1, max_num_test_files + 1):
        x_train, y_train, x_test, y_test = train_test_split_from_data(df_from_each_file, scaling_type,
                                                                      enable_normalization,
                                                                      num_test_files=num_test_files)
        model.fit(x_train, y_train)
        actual = model.predict(x_test)
        scores[num_test_files - 1] = mean_squared_error(y_test,
                                                        actual) if method_type == MethodType.Regression else accuracy_score(
            y_test, actual)
    return np.mean(scores)


def output_average_scores_for_all_methods(output_dir, out_file_name, files_dir, features, label, method_type):
    df_results = pd.DataFrame(
        columns=['Regressor' if method_type == MethodType.Regression else 'Classifier', 'Scaling Type',
                 'Enable Normalization', 'Use Default Params',
                 'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy'])
    cols = [feature for feature in features]
    cols.append(label)
    df_from_each_file = load_from_directory(files_dir, cols, concat=False)
    methods = classifiers if method_type == MethodType.Classification else regressors
    for method_name in methods:
        for scaling_type in ScalingType.all():
            for enable_normalization in [False, True]:
                for use_grid_search in [False, True]:
                    df_results = df_results.append(
                        {'Regressor' if method_type == MethodType.Regression else 'Classifier': method_name,
                         'Scaling Type': scaling_type.name,
                         'Enable Normalization': 'Yes' if enable_normalization else 'No',
                         'Use Default Params': 'No' if use_grid_search else 'Yes',
                         'Mean Squared Error' if method_type == MethodType.Regression else 'Accuracy':
                             get_average_score(method_name, df_from_each_file, scaling_type, enable_normalization, use_grid_search, method_type)},
                        ignore_index=True)
    df_results.to_csv(join(output_dir, '{0}.csv'.format(out_file_name)))


def run_with_different_methods(method_type, df_from_each_file, enable_scaling=True,
                               enable_normalization=True, **args):
    x_train, y_train, x_test, y_test = train_test_split_from_data(df_from_each_file, enable_scaling,
                                                                  enable_normalization, **args)
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
    method_keys = list(methods.keys())
    for metric in metrics:
        data_for_metric = []
        for i in range(len(method_keys)):
            data_for_metric.append([])
        for test_size in test_sizes:
            metric_data = results[test_size][metric]
            for i in range(len(data_for_metric)):
                data_for_metric[i].append(metric_data[i])
        metrics_to_data[metric] = data_for_metric
    return metrics_to_data, test_sizes


def select_method(choosing_method, method_type, use_grid_search=True, enable_normalization=False):
    chosen_method = None
    methods = classifiers if method_type == MethodType.Classification else regressors
    if choosing_method == 'random':
        chosen_method = randomly_select_method(methods)
    else:
        for method_name in methods.keys():
            if method_name.lower() == choosing_method.lower():
                chosen_method = methods[method_name]
                break
    if choosing_method in ['Linear Regression', 'Lasso', 'Ridge', 'Elastic Net']:
        if choosing_method == 'Linear Regression':
            chosen_method = LinearRegression(normalize=enable_normalization, n_jobs=-1)
        elif choosing_method == 'Lasso':
            chosen_method = Lasso(normalize=enable_normalization)
        elif choosing_method == 'Ridge':
            chosen_method = Ridge(normalize=enable_normalization)
        elif choosing_method == 'Elastic Net':
            chosen_method = ElasticNet(normalize=enable_normalization)
    if use_grid_search:
        params = get_testable_parameters(chosen_method, method_type)
        return set_up_gridsearch(chosen_method, params, method_type)
    else:
        return chosen_method


def randomly_select_method(methods):
    key = random.choice(list(methods.keys()))
    return methods[key]


def set_up_gridsearch(method, params, method_type):
    if not bool(params):
        return GridSearchCV(method, param_grid=params, cv=10,
                            scoring='neg_mean_squared_error' if method_type == MethodType.Regression else 'accuracy',
                            verbose=0, n_jobs=-1)
    else:
        return method


def get_testable_parameters(method_name, method_type):
    if method_name in ['Decision Tree', 'Random Forest']:
        return {'max_features': ['auto', 'log2', 'sqrt']}
    elif method_name in ['Lasso', 'Ridge']:
        return {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4]}
    elif method_name == 'Huber':
        return {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
    elif method_name == 'Elastic Net':
        return {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'l1_ratio': list(np.arange(0, 1, 0.05))}
    elif method_name in ['K-Nearest Neighbour', 'Radius Neighbours']:
        return {'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    elif method_name == 'SGD':
        return {'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'l1_ratio': list(np.arange(0, 1, 0.05)),
                'learning_rate': ['adaptive', 'invscaling', 'constant', 'optimal']}
    elif method_name == 'Nearest Centroid':
        return {'metric': ['euclidean', 'manhattan']}
    elif method_name == 'Linear Discriminant Analysis':
        return {'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': list(np.arange(0, 1, 0.05)),
                'store_covariance': [True, False],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4]}
    elif method_name == 'Quadratic Discriminant Analysis':
        return {'reg_param': list(np.arange(0, 1, 0.05)),
                'store_covariance': [True, False],
                'tol': [1e-1, 1e-2, 1e-3, 1e-4]}
    elif method_name == 'Logistic Regression':
        return {'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'dual': [True, False],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'l1_ratio': list(np.arange(0, 1, 0.05))}
    elif method_name == 'SVM':
        if method_type == MethodType.Classification:
            return {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto'],
                    'shrinking': [True, False],
                    'probability': [True, False],
                    'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                    'cache_size': list(np.arange(100, 1000, 100)),
                    'decision_function': ['ovo', 'ovr'],
                    'break_ties': [True, False]}
        else:
            return {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto'],
                    'shrinking': [True, False],
                    'epsilon': list(np.arange(0, 1, 0.05)),
                    'cache_size': list(np.arange(100, 1000, 100)),
                    'decision_function': ['ovo', 'ovr'],
                    'break_ties': [True, False]}
    elif method_name == 'Nu-SVM':
        if method_type == MethodType.Classification:
            return {'nu': list(np.arange(0, 1, 0.05)),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto'],
                    'shrinking': [True, False],
                    'probability': [True, False],
                    'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                    'cache_size': list(np.arange(100, 1000, 100)),
                    'decision_function': ['ovo', 'ovr'],
                    'break_ties': [True, False]}
        else:
            return {'nu': list(np.arange(0, 1, 0.05)),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto'],
                    'shrinking': [True, False],
                    'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                    'cache_size': list(np.arange(100, 1000, 100)),
                    'break_ties': [True, False]}
    elif method_name == 'Linear-SVM':
        if method_type == MethodType.Classification:
            return {'penalty': ['l1', 'l2'],
                    'loss': ['hinge', 'squared_hinge'],
                    'dual': [True, False],
                    'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                    'multi_class': ['ovr', 'crammer_singer'],
                    'fit_intercept': [True, False]}
        else:
            return {'epsilon': list(np.arange(0, 1, 0.05)),
                    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'dual': [True, False],
                    'tol': [1e-1, 1e-2, 1e-3, 1e-4],
                    'fit_intercept': [True, False]}
    else:
        return {}
