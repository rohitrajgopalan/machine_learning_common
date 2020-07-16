import enum
import random
import warnings
from os import listdir
from os.path import join, isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings('ignore')
classifiers = {'Logistic Regression': LogisticRegression(),
               'Decision Tree': DecisionTreeClassifier(),
               'Support Vector Machine': SVC(gamma='auto', kernel='rbf'),
               'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
               'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
               'Random Forest': RandomForestClassifier(),
               'K-Nearest Neighbors': KNeighborsClassifier(),
               'Bayes': GaussianNB()}
regressors = {'Decision Tree': DecisionTreeRegressor(),
              'Support Vector Machine': SVR(gamma='auto', kernel='rbf'),
              'Random Forest': RandomForestRegressor(),
              'K-Nearest Neighbour': KNeighborsRegressor(),
              'Lasso': Lasso(),
              'Ridge': Ridge()}
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc']
metrics = ['Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'Area underneath the curve']


class MethodType(enum.Enum):
    Classification = 1
    Regression = 2


scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


def train_test_split_from_data(df_from_each_file, num_test_files):
    if num_test_files > 0:
        training_sets = df_from_each_file[:len(df_from_each_file) - num_test_files]
        test_sets = df_from_each_file[len(df_from_each_file) - num_test_files:]
        training = pd.concat(training_sets, ignore_index=True)
        test = pd.concat(test_sets, ignore_index=True)

        cols = [col for col in training.columns]
        features = cols[:len(cols) - 1]
        label = cols[-1]

        train_x = training[features]
        scaler_x.fit(train_x)
        train_x = scaler_x.transform(train_x)
        train_y = np.array(training[label]).reshape(-1, 1)
        scaler_y.fit(train_y)
        train_y = scaler_y.transform(train_y)
        test_x = test[features]
        scaler_x.fit(test_x)
        test_x = scaler_x.transform(test_x)
        test_y = np.array(test[label]).reshape(-1, 1)
        scaler_y.fit(test_y)
        test_y = scaler_y.transform(test_y)

        return train_x, train_y, test_x, test_y


def plot_data(metrics_to_data, test_sizes, methods):
    method_names = list(methods.keys())
    for metric in metrics_to_data:
        for i in range(len(method_names)):
            plt.plot(test_sizes, metrics_to_data[metric][i], label=method_names[i])
        plt.xlabel('Number of Test Files')
        plt.ylabel(metric)
        plt.title('Comparing {0} values between methods'.format(metric))
        plt.legend()
        plt.show()
        print()


def load_from_directory(csv_dir, cols=[], filters={}, concat=False):
    datasets = [join(csv_dir, csv) for csv in listdir(csv_dir) if isfile(join(csv_dir, csv))]
    label = cols[len(cols) - 1]

    df_from_each_file = []

    for csv in datasets:
        df = pd.read_csv(csv)
        if not bool(filters):
            for key in filters:
                df = df.loc[df[key] == filters[key]]
                if df is None:
                    break
        if df is None or len(df.index) == 0:
            continue
        if not is_numeric_dtype(df[label]):
            unique_labels = list(np.unique(df[label]))
            if unique_labels == ['True', 'False']:
                replace_dict = {'True': 1, 'False': 0}
            else:
                replace_dict = {}
                for i, unique_label in enumerate(unique_labels):
                    replace_dict.update({unique_label: i + 1})
            df.replace({label: replace_dict})
        df = df[cols]
        if df is not None and len(df.index) > 0:
            df_from_each_file.append(df)

    return pd.concat(df_from_each_file, ignore_index=True) if concat else df_from_each_file


def perform_and_plot_experiment_on_data(csv_dir, methods):
    df_from_each_file = load_from_directory(csv_dir)
    results, test_sizes = perform_experiment_on_data(df_from_each_file, methods, type)
    metrics_to_data, test_sizes = shape_experimental_data_for_plotting(results, test_sizes, methods)
    plot_data(metrics_to_data, test_sizes, methods)


def perform_and_plot_experiment_on_data_with_classifiers(csv_dir):
    perform_and_plot_experiment_on_data(csv_dir, classifiers)


def perform_and_plot_experiment_on_data_with_regressors(csv_dir):
    perform_and_plot_experiment_on_data(csv_dir, regressors)


def perform_experiment_on_data(df_from_each_file, methods, method_type):
    results = {}
    test_sizes = []
    for num_test_files in range(1, int(len(df_from_each_file) / 5) + 1):
        test_sizes.append(num_test_files)
        if method_type == MethodType.Regression:
            results[num_test_files] = run_with_different_methods(df_from_each_file, num_test_files, methods)
        else:
            results[num_test_files] = run_with_different_classifiers(df_from_each_file, num_test_files)
    return results, test_sizes


def run_with_different_methods(df_from_each_file, num_test_files, methods):
    x_train, y_train, x_test, y_test = train_test_split_from_data(df_from_each_file, num_test_files)
    models_data = {'Model': list(methods.keys()),
                   'Accuracy': []}
    for key in methods:
        methods[key].fit(x_train, y_train)
        accuracy = methods[key].score(x_test, y_test)
        models_data['Accuracy'].append(accuracy)

    return models_data


def run_with_different_classifiers(df_from_each_file, num_test_files):
    x_train, y_train, x_test, y_test = train_test_split_from_data(df_from_each_file, num_test_files)
    models_data = {'Model': list(classifiers.keys()),
                   'Fitting time': [],
                   'Scoring time': [],
                   'Accuracy': [],
                   'Precision': [],
                   'Recall': [],
                   'F1 score': [],
                   'Area underneath the curve': []}
    for key in classifiers:
        scores = cross_validate(classifiers[key], x_train, y_train, scoring=scoring)
        sorted(scores.keys())

        models_data['Fitting time'].append(scores['fit_time'].mean())
        models_data['Scoring time'].append(scores['score_time'].mean())
        models_data['Accuracy'].append(scores['test_accuracy'].mean())
        models_data['Precision'].append(scores['test_precision_macro'].mean())
        models_data['Recall'].append(scores['test_recall_macro'].mean())
        models_data['F1 score'].append(scores['test_f1_weighted'].mean())
        models_data['Area underneath the curve'].append(scores['test_roc_auc'].mean())

    return models_data


def shape_experimental_data_for_plotting(results, test_sizes, methods, metrics=['Accuracy']):
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


def select_best_method(csv_dir, methods, best_type='', metric='Accuracy', features=[], label='', filters={},
                       method_type=MethodType.Classification):
    cols = features
    cols.append(label)
    cols = list(np.unique(cols))
    df_from_each_file = load_from_directory(csv_dir, cols, filters)
    method_names = list(methods.keys())
    results, test_sizes = perform_experiment_on_data(df_from_each_file, methods, method_type)
    metrics_to_data, _ = shape_experimental_data_for_plotting(results, test_sizes, methods)
    if metric not in metrics_to_data:
        return None
    else:
        dataset = metrics_to_data[metric]
        metric_values = []
        for i in range(len(method_names)):
            if best_type == 'average':
                metric_values.append(np.mean(dataset[i]))
            else:
                if metric.endswith('time'):
                    metric_values.append(np.min(dataset[i]))
                else:
                    metric_values.append(np.max(dataset[i]))
        sort_index = np.argsort(metric_values)
        best_method = methods[method_names[sort_index[-1]]]
        return best_method


def select_method(csv_dir, choosing_method='best', features=[], label='', filters={},
                  method_type=MethodType.Classification):
    methods = classifiers if method_type == MethodType.Classification else regressors
    if choosing_method == 'best':
        chosen_method = select_best_method(csv_dir, methods, features=features, label=label, filters=filters,
                                           method_type=method_type)
    elif choosing_method == 'random':
        chosen_method = randomly_select_method(methods)
    elif choosing_method in methods.keys():
        chosen_method = methods[choosing_method]
    return chosen_method


def randomly_select_method(methods):
    key = random.choice(list(methods.keys()))
    return methods[key]
