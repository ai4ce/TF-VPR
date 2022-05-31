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

runs_folder = "dm_data"
filename = "gt_pose.mat"
pointcloud_fols = "/pointcloud_20m_10overlap/"
test_num = 4

evaluate_all = True
print("cfg.DATASET_FOLDER:"+str(cfg.DATASET_FOLDER))

cc_dir = "/home/cc/"
all_folders = sorted(os.listdir(os.path.join(cc_dir,runs_folder)))
file_size_ = 2048
folders = []

# All runs are used for training (both full and partial)
if evaluate_all:
    index_list = list(range(18))
else:
    index_list = [5,6,7,9]
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)


#####For training and test data split#####
def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)

#########################################
def construct_query_dict(df_centroids, df_database, folder_num,  filename_train, filename_test, test=False):
    database_trees = []
    test_trees = []
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=15)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=50)
    queries_sets = []
    database_sets = []
    for folder in range(folder_num):
        queries = {}
        for i in range(len(df_centroids)//folder_num):
            temp_indx = folder*(len(df_centroids)//folder_num) + i
            query = df_centroids.iloc[temp_indx]["file"]
            #print("folder:"+str(folder))
            #print("query:"+str(query))
            queries[len(queries.keys())] = {"query":query,
                "x":float(df_centroids.iloc[temp_indx]['x']),"y":float(df_centroids.iloc[temp_indx]['y'])}
        queries_sets.append(queries)
        test_tree = KDTree(df_centroids[folder*test_num:(folder+1)*test_num][['x','y']])
        test_trees.append(test_tree)

    for folder in range(folder_num):
        dataset = {}
        for i in range(len(df_database)//folder_num):
            temp_indx = folder*len(df_database)//folder_num + i
            data = df_database.iloc[temp_indx]["file"]
            dataset[len(dataset.keys())] = {"query":data,
                     "x":float(df_database.iloc[temp_indx]['x']),"y":float(df_database.iloc[temp_indx]['y'])}
        
        database_sets.append(dataset)
        database_tree = KDTree(df_database[folder*file_size_:(folder+1)*file_size_][['x','y']])
        database_trees.append(database_tree)

    if test:
        for i in range(len(database_sets)):
            tree = database_trees[i]
            for j in range(len(queries_sets)):
                if(i == j):
                    continue
                for key in range(len(queries_sets[j].keys())):
                    coor = np.array(
                        [[queries_sets[j][key]["x"],queries_sets[j][key]["y"]]])
                    index = tree.query_radius(coor, r=25)
                    # indices of the positive matches in database i of each query (key) in test set j
                    queries_sets[j][key][i] = index[0].tolist()
    
    output_to_file(queries_sets, filename_test)
    output_to_file(database_sets, filename_train)

# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','x','y'])
df_test = pd.DataFrame(columns=['file','x','y'])

df_files_test = []
df_files_train =[]

df_locations_tr_x = []
df_locations_tr_y = []
df_locations_ts_x = []
df_locations_ts_y = []

for folder in folders:
    df_locations = sio.loadmat(os.path.join(
        cc_dir,runs_folder,folder,filename))
    
    df_locations = df_locations['pose']
    df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

    #2038 Training 10 testing
    test_index = list(sorted(random.sample(range(len(df_locations)), k=test_num)))
    train_index = list(range(df_locations.shape[0]))
    #for i in test_index:
    #    train_index.pop(i)
        
    df_locations_tr_x.extend(list(df_locations[train_index,0]))
    df_locations_tr_y.extend(list(df_locations[train_index,1]))
    df_locations_ts_x.extend(list(df_locations[test_index,0]))
    df_locations_ts_y.extend(list(df_locations[test_index,1]))

    all_files = list(sorted(os.listdir(os.path.join(cc_dir,runs_folder,folder))))
    all_files.remove('gt_pose.mat')
    all_files.remove('gt_pose.png')

    for (indx, file_) in enumerate(all_files):
        if indx in test_index:
            df_files_test.append(os.path.join(cc_dir,runs_folder,folder,file_))
        df_files_train.append(os.path.join(cc_dir,runs_folder,folder,file_))


df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                               columns =['file','x', 'y'])
df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                               columns =['file','x', 'y'])

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

print("df_train:"+str(len(df_train)))

#construct_query_dict(df_train,len(folders),"evaluation_database.pickle",False)
if not evaluate_all:
    construct_query_dict(df_test, df_train, len(folders),"evaluation_database.pickle", "evaluation_query.pickle", True)
else:
    construct_query_dict(df_test, df_train, len(folders),"evaluation_database_full.pickle", "evaluation_query_full.pickle", True)
