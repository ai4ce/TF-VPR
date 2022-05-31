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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = cfg.DATASET_FOLDER

runs_folder = "dm_data/"
filename = "gt_pose.mat"
pointcloud_fols = "/pointcloud_20m_10overlap/"

print("cfg.DATASET_FOLDER:"+str(cfg.DATASET_FOLDER))

cc_dir = "/home/cc/"
all_folders = sorted(os.listdir(os.path.join(cc_dir,runs_folder)))

folders = []

# All runs are used for training (both full and partial)
index_list = range(18)
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)


#####For training and test data split#####


def construct_query_dict(df_centroids, train_index, test_index,  filename_train, filename_test):
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=15)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=50)
    queries = {}
    queries_test = {}
    
    #for i in range(len(ind_nn)):
    for i in train_index:
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":df_centroids.iloc[i]['file'],
                      "positives":positives,"negatives":negatives}
    
    for i in test_index:
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries_test[i] = {"query":df_centroids.iloc[i]['file'],
                      "positives":positives,"negatives":negatives}
    
    with open(filename_train, 'wb') as handle:                                  
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename_test, 'wb') as handle:
        pickle.dump(queries_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename_train)
    print("Done ", filename_test)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','x','y'])
df_test = pd.DataFrame(columns=['file','x','y'])
df_file = pd.DataFrame(columns=['file','x','y'])

df_files_test = []
df_files_train =[]
df_files = []

df_indice_test = []
df_indice_train = []
df_indice = []

df_locations_tr_x = []
df_locations_tr_y = []
df_locations_ts_x = []
df_locations_ts_y = []
df_locations_db_x = []
df_locations_db_y = []

count = 0
for folder in folders:
    df_locations = sio.loadmat(os.path.join(
        cc_dir,runs_folder,folder,filename))
    
    df_locations = df_locations['pose']
    df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

    #2038 Training 10 testing
    test_index = random.sample(range(len(df_locations)), k=10)
    df_indice_test.extend(np.array(test_index)+count*2048)
    train_index = list(range(df_locations.shape[0]))
    df_indice_train.extend(np.array(train_index)+count*2048)
    db_index = list(range(df_locations.shape[0]))
    count=count+1
    for i in test_index:
        train_index.remove(i)
    
    df_locations_tr_x.extend(list(df_locations[train_index,0]))
    df_locations_tr_y.extend(list(df_locations[train_index,1]))
    df_locations_ts_x.extend(list(df_locations[test_index,0]))
    df_locations_ts_y.extend(list(df_locations[test_index,1]))
    df_locations_db_x.extend(list(df_locations[db_index,0]))
    df_locations_db_y.extend(list(df_locations[db_index,1]))
    
    all_files = list(sorted(os.listdir(os.path.join(cc_dir,runs_folder,folder))))
    all_files.remove('gt_pose.mat')
    all_files.remove('gt_pose.png')

    for (indx, file_) in enumerate(all_files):
        if indx in test_index:
            df_files_test.append(os.path.join(cc_dir,runs_folder,folder,file_))
        else:
            df_files_train.append(os.path.join(cc_dir,runs_folder,folder,file_))
        df_files.append(os.path.join(cc_dir,runs_folder,folder,file_))


print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
print("df_files_test:"+str(len(df_files_test)))

df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                               columns =['file','x', 'y'])
df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                               columns =['file','x', 'y'])
df_file = pd.DataFrame(list(zip(df_files, df_locations_db_x, df_locations_db_y)),
                                               columns =['file','x', 'y'])


print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

construct_query_dict(df_file, df_indice_train, df_indice_test, "training_queries_baseline.pickle", "test_queries_baseline.pickle")
#construct_query_dict(df_test,"test_queries_baseline.pickle")
