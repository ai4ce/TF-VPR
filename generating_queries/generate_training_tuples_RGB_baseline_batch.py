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

def construct_dict(folder_num, df_f, df_all, df_files, df_files_all, df_indices, df_indices_all, df_locations_x_all, df_locations_y_all, df_locations_x, df_locations_y, filename, pre_dir):
    nn_ind = 0.005
    r_mid = 0.005
    r_ind = 0.015
    
    queries = {}
    count = 0
    
    df_centroids = df_all
    df_folder_index = df_indices

    tree = KDTree(df_centroids[['x','y']])
    ind_r = tree.query_radius(df_centroids[['x','y']], r=r_ind) 
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=nn_ind)
    
    for i in df_indices:
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
                df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        if len(positives)<2:
            positives = []
            positives.append(i+1)
            positives.append(i-1)
            positives = np.array(positives)
        # print(len(positives))
        # assert(len(positives)<=1000)
                
        queries[i] = {"query":df_centroids.iloc[i]['file'],
                "positives":positives,"negatives":negatives}


    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    base_path = "/mnt/NAS/data/cc_data/2D_RGB_real_edited3"
    base_path = os.path.join(base_path)
    pre_dir = base_path
    '''
    runs_folder = cfg.scene_names[scene_index]
    print("runs_folder2:"+str(runs_folder))

    pre_dir = os.path.join(base_path, runs_folder)
    print("pre_dir:"+str(pre_dir))
    '''
    filename = "gt_pose.mat"
    
    # Initialize pandas DataFrame

    df_train = pd.DataFrame(columns=['file','x','y'])
    df_test = pd.DataFrame(columns=['file','x','y'])
    df_all = pd.DataFrame(columns=['file','x','y'])

    df_files_test = []
    df_files_train =[]
    df_files = []

    df_indices_train = []
    df_indices_test = []
    df_indices = []

    df_locations_tr_x = []
    df_locations_tr_y = []
    # df_locations_tr_z = []

    df_locations_ts_x = []
    df_locations_ts_y = []
    # df_locations_ts_z = []

    df_locations_x = []
    df_locations_y = []
    # df_locations_z = []
    
    fold_list = list(sorted(os.listdir(base_path)))
    all_files = []
    for ind, fold in enumerate(fold_list):
        files_ = []
        files = list(sorted(os.listdir(os.path.join(base_path, fold))))
        files.remove('gt_pose.mat')
        # print("len(files):"+str(len(files)))
        for ind_f in range(len(files)):
            file_ = "panoimg_"+str(ind_f)+".jpg"
            files_.append(os.path.join(base_path, fold, file_))
        df_files.extend(files_)
        df_locations = sio.loadmat(os.path.join(base_path,fold,filename))
        df_locations = df_locations['pose']
        #df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
        # print(df_locations.shape)
        # assert()
        file_index = list(range(df_locations.shape[0]))

        df_locations_x.extend(list(df_locations[file_index,0]))
        # df_locations_z.extend(list(df_locations[file_index,1]))
        df_locations_y.extend(list(df_locations[file_index,1]))
        test_sample = 10
        test_index_temp = random.sample(range(df_locations.shape[0]), k=test_sample)
        test_index = list(np.array(test_index_temp)+ind * df_locations.shape[0])

        df_indices_test.extend(test_index)
        train_index_temp = list(range(df_locations.shape[0]))
        train_index = list(np.array(train_index_temp)+ind * len(files_))
        df_indices.extend(train_index)
        files_ = []
        for ts_ind in test_index:
            train_index.remove(ts_ind)
            file_ = "panoimg_"+str(ts_ind)+".jpg"
            files_.append(os.path.join(base_path, fold, file_))
        df_indices_train.extend(train_index)
        df_files_test.extend(files_)
        files_ = []
        for tr_ind in train_index:
            file_ = "panoimg_"+str(tr_ind)+".jpg"
            files_.append(os.path.join(base_path, fold, file_))
        df_files_train.extend(files_)

        df_locations_tr_x.extend(list(df_locations[train_index_temp,0]))
        # df_locations_tr_z.extend(list(df_locations[train_index_temp,1]))
        df_locations_tr_y.extend(list(df_locations[train_index_temp,1]))
        
        df_locations_ts_x.extend(list(df_locations[test_index_temp,0]))
        # df_locations_ts_z.extend(list(df_locations[test_index_temp,1]))
        df_locations_ts_y.extend(list(df_locations[test_index_temp,1]))
        
    traj_len = len(all_files)
    
    df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                                                       columns =['file','x', 'y'])
    df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                                                       columns =['file','x', 'y'])
    df_all = pd.DataFrame(list(zip(df_files, df_locations_x, df_locations_y)),
                                                                       columns =['file','x', 'y'])
    #n Training 10 testing
    
    if inside == True:
        construct_dict(len(fold_list),df_train,df_all,df_files_train,df_files,df_indices_train, df_indices,df_locations_x,df_locations_y,df_locations_tr_x,df_locations_tr_y,"train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir)
        construct_dict(len(fold_list),df_test,df_all,df_files_test,df_files,df_indices_test,df_indices,df_locations_x,df_locations_y,df_locations_ts_x,df_locations_ts_y, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir)
        construct_dict(len(fold_list),df_all,df_all,df_files,df_files,df_indices,df_indices,df_locations_x,df_locations_y,df_locations_x,df_locations_y,"train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir)

    else:
        construct_dict(len(fold_list),df_train,df_all,df_files_train,df_files,df_indices_train,df_indices,df_locations_x,df_locations_y,df_locations_tr_x,df_locations_tr_y, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", pre_dir)
        construct_dict(len(fold_list),df_test,df_all,df_files_test,df_files, df_indices_test,df_indices,df_locations_x,df_locations_y,df_locations_ts_x,df_locations_ts_y, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", pre_dir)
        construct_dict(len(fold_list),df_all,df_all,df_files,df_files, df_indices, df_indices, df_locations_x,df_locations_y,df_locations_x,df_locations_y, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", pre_dir)

if __name__ == "__main__":
    for i in range(1):
        generate(i)
