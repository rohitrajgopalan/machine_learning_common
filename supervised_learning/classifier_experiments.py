import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import basename, isfile, join, splitext
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,NuSVC
import warnings
warnings.filterwarnings('ignore')
classifiers = {'Logistic Regression': LogisticRegression(),
               'Decision Tree': DecisionTreeClassifier(),
               'Support Vector Machine': make_pipeline(StandardScaler(),SVC(gamma='auto',kernel='rbf')),
               'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
               'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
               'Random Forest': RandomForestClassifier(),
               'K-Nearest Neighbors': KNeighborsClassifier(),
               'Bayes': GaussianNB()}
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
metrics= ['Fitting time','Scoring time','Accuracy','Precision','Recall','F1 score','Area underneath the curve']

def train_test_split_from_data(df_from_each_file,num_test_files):
    training_sets = df_from_each_file[:len(df_from_each_file)-num_test_files]
    test_sets = df_from_each_file[len(df_from_each_file)-num_test_files:]
    training = pd.concat(training_sets,ignore_index=True)
    test = pd.concat(test_sets,ignore_index=True)

    cols = [col for col in training.columns]
    features = cols[:len(cols)-1]
    label = cols[-1]

    return training[features],test[features],training[label],test[label]

def run_with_different_methods(df_from_each_file,num_test_files):
    X_train, X_test, y_train, y_test = train_test_split_from_data(df_from_each_file,num_test_files)
    best_methods = []
    best_scores = []
    models_data = {'Model': list(classifiers.keys()),
            'Fitting time': [],
            'Scoring time': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 score': [],
            'Area underneath the curve':[]}
    test_size = num_test_files/len(df_from_each_file)
    features = X_train.columns
    n_splits = int(test_size*len(features)*2)
    for key in classifiers:
        scores = cross_validate(classifiers[key],X_train,y_train, scoring=scoring)

        sorted(scores.keys())

        models_data['Fitting time'].append(scores['fit_time'].mean())
        models_data['Scoring time'].append(scores['score_time'].mean())
        models_data['Accuracy'].append(scores['test_accuracy'].mean())
        models_data['Precision'].append(scores['test_precision_macro'].mean())
        models_data['Recall'].append(scores['test_recall_macro'].mean())
        models_data['F1 score'].append(scores['test_f1_weighted'].mean())
        models_data['Area underneath the curve'].append(scores['test_roc_auc'].mean())
        
    return models_data

def plot_data(metrics_to_data,test_sizes):
    method_names = list(classifiers.keys())
    for metric in metrics_to_data:
        for i in range(len(method_names)):
            plt.plot(test_sizes,metrics_to_data[metric][i],label=method_names[i])
        plt.xlabel('Number of Test Files')
        plt.ylabel(metric)
        plt.title('Comparing {0} values between classifiers'.format(metric))
        plt.legend()
        plt.show()
        print()
    
def return_top_four_classifiers(df_from_each_file,metric_name,best_type):
    metrics_to_data,_ = perform_experiment_on_data(df_from_each_file)
    method_names = list(classifiers.keys())
    if not metric_name in metrics_to_data:
        return []
    else:
        dataset = metrics_to_data[metric_name]
        metric_values = []
        for i in range(len(method_names)):
            if best_type == 'average':
                metric_values.append(np.mean(dataset[i]))
            else:
                metric_values.append(np.max(dataset[i]))
        sort_index = np.argsort(metric_values)
        return [method_names[sort_index[-1]],method_names[sort_index[-2]],method_names[sort_index[-3]],method_names[sort_index[-4]]]

def shape_experimental_data_for_plotting(results,test_sizes):
    metrics_to_data = {}
    method_names = list(classifiers.keys())
    for metric in metrics:
        data_for_metric = []
        for i in range(len(method_names)):
            data_for_metric.append([])
        for test_size in test_sizes:
            metric_data = results[test_size][metric]
            for i in range(len(data_for_metric)):
                data_for_metric[i].append(metric_data[i])
        metrics_to_data[metric] = data_for_metric
    return metrics_to_data,test_sizes

def perform_experiment_on_data(df_from_each_file):
    results = {}
    test_sizes = []
    for num_test_files in range(1,int(len(df_from_each_file)/5)+1):
        test_sizes.append(num_test_files)
        results[num_test_files] = run_with_different_methods(df_from_each_file,num_test_files)
    return shape_experimental_data_for_plotting(results,test_sizes)

def perform_and_plot_experiment_on_data(df_from_each_file):
    stacking_classifier = develop_stacking_classifier(df_from_each_file,'Accuracy','max')
    classifiers.update({'Stacking Classifier':stacking_classifier})
    metrics_to_data,test_sizes = perform_experiment_on_data(df_from_each_file)
    plot_data(metrics_to_data,test_sizes)   

def develop_stacking_classifier(df_from_each_file,metric_name,best_type):
    top_four_classifiers = return_top_four_classifiers(df_from_each_file,metric_name,best_type)
    estimators = top_four_classifiers[1:]
    final_estimator = top_four_classifiers[0]

    return StackingClassifier(estimators,final_estimator=final_estimator)

