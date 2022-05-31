import os
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


def construct_dict(df_files, df_indices, filename, folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=None):
    pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
    mid_index_range = list(range(-k_nearest, (k_nearest)+1))
    pos_index_range.remove(0)

    replace_counts = 0
    queries = {}
    for num in range(folder_num):

        for index in range(len(df_indices)//folder_num):
            df_indice = df_indices[num * (len(df_indices)//folder_num) + index]
            positive_l = []
            negative_l = list(range(num*folder_size,(num+1)*folder_size,1))
            for index_pos in pos_index_range:
                if (index_pos + df_indice >= 0) and (index_pos + df_indice <= folder_size -1):
                    positive_l.append(index_pos + df_indice+folder_size*num)
            for index_pos in mid_index_range:
                if (index_pos + df_indice >= 0) and (index_pos + df_indice <= folder_size -1):
                    negative_l.remove(index_pos + df_indice+folder_size*num)
            if definite_positives is not None:

                if ((np.array(definite_positives[num][df_indice]).ndim) == 2) and (np.array(definite_positives[num][df_indice]).shape[0]!=0):
                    extend_element = definite_positives[num][df_indice][0]
                else:
                    # print("extend_element_pre:"+str(definite_positives[num][df_indice]))
                    extend_element = definite_positives[num][df_indice]

                positive_l.extend(extend_element)
                # print("positive_l:"+str(positive_l))
                positive_l = list(set(positive_l))
                negative_l = [i for i in negative_l if i not in extend_element]

            queries[num * (len(df_indices)//folder_num) + index] = {"query":df_files[num * (len(df_indices)//folder_num) + index],
                          "positives":positive_l,"negatives":negative_l}
    
    #print("replace_counts:"+str(replace_counts))        
    #print("queries:"+str(queries[0][0]))
    
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = cfg.DATASET_FOLDER
    runs_folder = "dm_data/"
    print("cfg.DATASET_FOLDER:"+str(cfg.DATASET_FOLDER))
    
    cc_dir = "/home/cc/"
    all_folders = sorted(os.listdir(os.path.join(cc_dir,runs_folder)))
    
    folders = []
    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    print("Number of runs: "+str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    
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

    folder_num = len(folders)

    for folder in folders:
        all_files = list(sorted(os.listdir(os.path.join(cc_dir,runs_folder,folder))))
        all_files.remove('gt_pose.mat')
        all_files.remove('gt_pose.png')
        
        folder_size = len(all_files)
        test_index = random.sample(range(folder_size), k=2)
        train_index = list(range(folder_size))
        for ts_ind in test_index:
            train_index.remove(ts_ind)
        
        for (indx, file_) in enumerate(all_files): 
            if indx in test_index:
                df_files_test.append(os.path.join(cc_dir,runs_folder,folder,file_))
                df_indices_test.append(indx)
            else:
                df_files_train.append(os.path.join(cc_dir,runs_folder,folder,file_))
                df_indices_train.append(indx)
            df_files.append(os.path.join(cc_dir,runs_folder,folder,file_))
            df_indices.append(indx)

    pre_dir = os.path.join(cc_dir,runs_folder)

    if inside == True:
        construct_dict(df_files_train, df_indices_train, "train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest)
        construct_dict(df_files_test, df_indices_test, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest)
        construct_dict(df_files, df_indices, "train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest)

    else:
        construct_dict(df_files_train, df_indices_train, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)
        construct_dict(df_files_test, df_indices_test, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)
        construct_dict(df_files, df_indices, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_size, folder_num, all_folders, pre_dir, k_nearest, k_furthest, definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(1):
        generate(i)
