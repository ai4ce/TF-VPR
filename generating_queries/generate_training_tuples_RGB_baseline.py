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

def construct_dict(folder_num, df_files, df_files_all, df_indices, filename, pre_dir, k_nearest, k_furthest, traj_len, definite_positives=None):
    pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
    mid_index_range = list(range(-k_nearest, (k_nearest)+1))
    queries = {}
    count = 0
    traj_len = int(len(df_files_all)/folder_num)
    for df_indice in df_indices:
        cur_fold_num = int(df_indice//traj_len)
        file_index = int(df_indice%traj_len)
        positive_l = []
        negative_l = list(range(cur_fold_num*traj_len, (cur_fold_num+1)*traj_len, 1))
        
        cur_indice = df_indice % traj_len

        for index_pos in pos_index_range:
            if (index_pos + cur_indice >= 0) and (index_pos + cur_indice <= traj_len -1):
                positive_l.append(index_pos + df_indice)
        for index_pos in mid_index_range:
            if (index_pos + cur_indice >= 0) and (index_pos + cur_indice <= traj_len -1):
                negative_l.remove(index_pos + df_indice)
        #positive_l.append(df_indice)
        #positive_l.append(df_indice)
        #negative_l.remove(df_indice)

        if definite_positives is not None:
            if len(definite_positives)==1:
                if definite_positives[0][df_indice].ndim ==2:
                    positive_l.extend(definite_positives[0][df_indice][0])
                    negative_l = [i for i in negative_l if i not in definite_positives[0][df_indice][0]]
                else:
                    positive_l.extend(definite_positives[0][df_indice])
                    negative_l = [i for i in negative_l if i not in definite_positives[0][df_indice]]
            else:
                positive_l.extend(definite_positives[df_indice])
                positive_l = list(set(positive_l))
                negative_l = [i for i in negative_l if i not in definite_positives[df_indice]]

        queries[count] = {"query":df_files[count],
                        "positives":positive_l,"negatives":negative_l}
        count = count + 1

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(scene_index, data_index, definite_positives=None, inside=True):
    base_path = "/mnt/NAS/home/cc/data/habitat_4/train/"
    run_folder = cfg.scene_list[scene_index]
    base_path = os.path.join(base_path,run_folder)
    pre_dir = base_path
    '''
    runs_folder = cfg.scene_names[scene_index]
    print("runs_folder2:"+str(runs_folder))

    pre_dir = os.path.join(base_path, runs_folder)
    print("pre_dir:"+str(pre_dir))
    '''
    filename = "gt_pose.mat"
    
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

    fold_list = list(sorted(os.listdir(base_path)))
    all_files = []
    for fold in fold_list:
        files_ = []
        files = list(sorted(os.listdir(os.path.join(base_path, fold))))
        files.remove('gt_pose.mat')
        # print("len(files):"+str(len(files)))
        for ind in range(len(files)):
            file_ = "panoimg_"+str(ind)+".png"
            files_.append(os.path.join(base_path, fold, file_))
        all_files.extend(files_)
    
    traj_len = len(all_files)
    
    #n Training 10 testing
    test_sample = len(fold_list)*10
    file_index = list(range(traj_len))
    test_index = random.sample(range(traj_len), k=test_sample)
    train_index = list(range(traj_len))
    for ts_ind in test_index:
        train_index.remove(ts_ind)
    
    for indx in range(traj_len):
        # file_ = 'panoimg_'+str(indx)+'.png'
        if indx in test_index:
            df_files_test.append(all_files[indx])
            df_indices_test.append(indx)        
        else:
            df_files_train.append(all_files[indx])
            df_indices_train.append(indx)       
        df_files.append(all_files[indx])
        df_indices.append(indx)
        
    if not os.path.exists(cfg.PICKLE_FOLDER):
        os.mkdir(cfg.PICKLE_FOLDER)
    
    if inside == True:
        construct_dict(len(fold_list),df_files_train, df_files,df_indices_train, "train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)))
        construct_dict(len(fold_list), df_files_test, df_files,df_indices_test, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)))
        construct_dict(len(fold_list), df_files,df_files, df_indices, "train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)))

    else:
        construct_dict(len(fold_list), df_files_train,df_files, df_indices_train, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)), definite_positives=definite_positives)
        construct_dict(len(fold_list), df_files_test,df_files, df_indices_test, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)), definite_positives=definite_positives)
        construct_dict(len(fold_list), df_files,df_files, df_indices, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir, k_nearest, k_furthest, int(traj_len/len(fold_list)), definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(1):
        generate(0,i)
