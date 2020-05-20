#!/usr/bin/python

import sys
import os
from os import listdir
from os.path import basename, isfile, join
import pandas as pd
import numpy as np
import datetime

fileDir = os.path.dirname(os.path.realpath('__file__'))
#d = datetime.date.today().strftime("%y%m%d")
csv_dir = os.path.join(fileDir,'datasets')
metric_cols = ['NUM_COLLISIONS','NUM_CLUSTERS','NUM_HOSTILES_DETECTED']
metrics_to_classification_label = {'NUM_COLLISIONS':'LIKELY_TO_COLLIDE',
                                   'NUM_CLUSTERS':'',
                                   'NUM_HOSTILES_DETECTED':'LIKELY_TO_DETECT_HOSTILE'}
datasets = [join(csv_dir,f) for f in listdir(csv_dir) if isfile(join(csv_dir,f))]
df_from_each_file = [pd.read_csv(f) for f in datasets]
def prepare_data_for_regression(df, metric_to_use):
    other_cols = [col for col in metric_cols if not col == metric_to_use]
    for col in other_cols:
        if col in df.columns:
            df.drop(col,axis=1,inplace=True)

def remove_decision_cols(df):
    decision_cols = [col for col in df.columns if col.endswith('DECISION')]
    for column in decision_cols:
        df.drop(column,axis=1,inplace=True)
        
def get_columns_for_nodeID(df,nodeID):
    cols = [col for col in df.columns if col.startswith('NODE{0}'.format(nodeID))]
    return cols[:5]

def get_filtered_data_for_nodeID(df,nodeID):
    cols = get_columns_for_nodeID(df,nodeID)
    return df[cols]

def get_data_for_nodeID(nodeID, fileIndex=None):
    df = prepare_data_for_ML(fileIndex)
    return get_filtered_data_for_nodeID(df,nodeID)
    
def prepare_data_for_classification(df, metric_to_use):
    other_cols = [col for col in metric_cols if not col == metric_to_use]
    for col in other_cols:
        if col in df.columns:
            df.drop(col,axis=1,inplace=True)
    
    binary_values = []
    for index, row in df.iterrows():
        value = 0
        if row[metric_to_use] > 0:
            value = 1
        binary_values.append(value)

    classification_label = metrics_to_classification_label[metric_to_use]
    df[classification_label] = binary_values
    df.drop(metric_to_use,axis=1,inplace=True)
    
def prepare_data_for_ML(fileIndex = None):    
    if len(df_from_each_file) == 0:
        return None
    
    if fileIndex is None or fileIndex >= len(datasets):
        df = pd.concat(df_from_each_file, ignore_index=True)
    else:
        df = df_from_each_file[fileIndex]

    drop_cols(df)
   
    return df

def drop_cols(df):
    uuid_cols = [col for col in df.columns if col.endswith('UUID')]
    for column in uuid_cols:
        df.drop(column,axis=1,inplace=True)

    df.drop("TIME",axis=1,inplace=True)
    iff_cols = [col for col in df.columns if col.endswith('IFF')]
    for col in iff_cols:
        df.drop(col,axis=1,inplace=True)
        
    commander_cols = [col for col in df.columns if col.endswith('COMMANDER')]
    for column in commander_cols:
        df.drop(column,axis=1,inplace=True)

    is_following_cols = [col for col in df.columns if col.endswith('ISFOLLOWING')]
    for column in is_following_cols:
        df.drop(column,axis=1,inplace=True)

    closest_enemy_cols = [col for col in df.columns if col.endswith('CLOSESTENEMY')]
    for column in closest_enemy_cols:
        df.drop(column,axis=1,inplace=True)   

def drop_cols_for_each_dataset():
    for df in df_from_each_file:
        drop_cols(df)

def prepare_each_dataset_for_classification(metric_to_use):
    for df in df_from_each_file:
        remove_decision_cols(df)
        prepare_data_for_classification(df,metric_to_use)

def organise_data_into_tuples(nodeID):
    df = get_data_for_nodeID(nodeID)
    cols = df.columns
    states = []
    decisions = []
    for index, row in df.iterrows():
        states.append((row[cols[0]],row[cols[1]],row[cols[2]],row[cols[3]]))
        decisions.append(row[cols[4]])
    decisions_unique = list(np.unique(decisions))
    tuples = []
    for i in range(len(states)-1):
        tuples.append((i,decisions_unique.index(decisions[i+1]),i+1))

    if len(decisions_unique) == 2:
        del decisions_unique[1]
    return states,decisions_unique,tuples
