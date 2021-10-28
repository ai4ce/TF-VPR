import os
import sys
import pickle
import random
import set_path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

import config as cfg

import scipy.io as sio
import torch

#####For training and test data split#####

def construct_dict(df_files, df_indices, filename, pre_dir, k_nearest, k_furthest, traj_len, definite_positives=None):
    pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
    mid_index_range = list(range(-k_nearest, (k_nearest)+1))
    
    queries = {}
    count = 0
    for df_indice in df_indices:
        positive_l = []
        negative_l = list(range(traj_len))

        for index_pos in pos_index_range:
            if (index_pos + df_indice >= 0) and (index_pos + df_indice <= traj_len -1):
                positive_l.append(index_pos + df_indice)
        for index_pos in mid_index_range:
            if (index_pos + df_indice >= 0) and (index_pos + df_indice <= traj_len -1):
                negative_l.remove(index_pos + df_indice)
        if definite_positives is not None:
            positive_l.extend(definite_positives[num][df_indice])
            negative_l = [i for i in negative_l if i not in definite_positives[num][df_indice]]

        positive_l = list(set(positive_l))
        positive_l.remove(df_indice)

        queries[count] = {"query":df_files[count],
                        "positives":positive_l,"negatives":negative_l}
        count = count + 1

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    base_path = "/mnt/NAS/home/yuhang/videomap_v2/"
    runs_folder = "Adrian"
    
    pre_dir = os.path.join(base_path, runs_folder)
    
    # Initialize pandas DataFrame
    k_nearest = 10
    k_furthest = 50

    df_train = pd.DataFrame(columns=['file','positives','negatives'])
    df_test = pd.DataFrame(columns=['file','positives','negatives'])

    df_files_test = []
    df_files_train =[]
    df_files = []

    df_indices_train = []
    df_indices_test = []
    df_indices = []

    all_files = list(sorted(os.listdir(pre_dir)))
    all_files.remove(runs_folder+'.json')
    all_files.remove('trajectory.mp4')
    all_files = [i for i in all_files if not i.endswith(".npy")]

    traj_len = len(all_files)
    
    #n Training 10 testing
    file_index = list(range(traj_len))
    test_index = random.sample(range(traj_len), k=10)
    train_index = list(range(traj_len))
    for ts_ind in test_index:
        train_index.remove(ts_ind)
    
    for indx in range(traj_len):
        file_ = 'panoimg_'+str(indx)+'.png'
        if indx in test_index:
            df_files_test.append(os.path.join(file_))
            df_indices_test.append(indx)        
        else:
            df_files_train.append(os.path.join(file_))
            df_indices_train.append(indx)       
        df_files.append(os.path.join(file_))
        df_indices.append(indx)
    
    if inside == True:
        construct_dict(df_files_train, df_indices_train, "train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len)
        construct_dict(df_files_test, df_indices_test, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len)
        construct_dict(df_files, df_indices, "train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len)

    else:
        construct_dict(df_files_train, df_indices_train, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len, definite_positives=definite_positives)
        construct_dict(df_files_test, df_indices_test, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len, definite_positives=definite_positives)
        construct_dict(df_files, df_indices, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, traj_len, definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(1):
        generate(i)
