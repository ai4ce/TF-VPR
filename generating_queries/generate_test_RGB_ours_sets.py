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
import json

##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)

#########################################
def construct_query_dict(df_centroids, df_database, folder_num, traj_len,  filename_train, filename_test, nn_ind, r_mid, r_ind, test=False, evaluate_all=False):
    database_trees = []
    test_trees = []
    if not evaluate_all:
        tree = KDTree(df_centroids[['x','y']])
        ind_nn = tree.query_radius(df_centroids[['x','y']],r=nn_ind)
        ind_r = tree.query_radius(df_centroids[['x','y']], r=r_ind)
    queries_sets = []
    database_sets = []

    queries = {}
    for i in range(len(df_centroids)):
        temp_indx = i
        query = df_centroids.iloc[temp_indx]["file"]

        if not evaluate_all:
            queries[len(queries.keys())] = {"query":query,
                "x":float(df_centroids.iloc[temp_indx]['x']),"y":float(df_centroids.iloc[temp_indx]['y'])}
        else:
            queries[len(queries.keys())] = {"query":query}

    queries_sets.append(queries)
    if not evaluate_all:
        test_tree = KDTree(df_centroids[['x','y']])
        test_trees.append(test_tree)
    
    ###############################
    dataset = {}
    for i in range(len(df_database)):
        temp_indx = i
        data = df_database.iloc[temp_indx]["file"]
        if not evaluate_all:
            dataset[len(dataset.keys())] = {"query":data,
                        "x":float(df_database.iloc[temp_indx]['x']),"y":float(df_database.iloc[temp_indx]['y']) }
        else:
            dataset[len(dataset.keys())] = {"query":data}
    database_sets.append(dataset)
    if not evaluate_all:
        database_tree = KDTree(df_database[['x','y']])
        database_trees.append(database_tree)
    ##################################
    if test:
        if not evaluate_all:
            tree = database_trees[0]

            for key in range(len(queries_sets[0].keys())):
                coor = np.array(
                    [[queries_sets[0][key]["x"],queries_sets[0][key]["y"]]])
                index = tree.query_radius(coor, r=r_mid)
                queries_sets[0][key][0] = index[0].tolist()
    else:
        pass
    
    output_to_file(queries_sets, filename_test)
    output_to_file(database_sets, filename_train)

def generate(scene_index, evaluate_all = False, inside=True):
    base_path = "/mnt/NAS/home/cc/data/habitat_4/train"
    runs_folder = cfg.scene_list[scene_index]
    pre_dir = os.path.join(base_path, runs_folder)

    nn_ind = 0.2
    r_mid = 0.2
    r_ind = 0.6

    filename = "gt_pose.mat"

    folders = list(sorted(os.listdir(pre_dir)))
    if evaluate_all == False:
        index_list = list(range(len(folders)))
    else:
        index_list = list(range(len(folders)))

    fold_list = []
    for index in index_list:
        fold_list.append(folders[index])

    all_files = []
    for fold in fold_list:
        files_ = []
        files = list(sorted(os.listdir(os.path.join(pre_dir, fold))))
        files.remove('gt_pose.mat')
        # print("len(files):"+str(len(files)))
        for ind in range(len(files)):
            file_ = "panoimg_"+str(ind)+".png"
            files_.append(os.path.join(pre_dir, fold, file_))
        all_files.extend(files_)

    traj_len = len(all_files)
    file_size = traj_len/len(fold_list)

    # Initialize pandas DataFrame
    if evaluate_all:
        df_train = pd.DataFrame(columns=['file'])
        df_test = pd.DataFrame(columns=['file'])
    else:
        df_train = pd.DataFrame(columns=['file','x','y'])
        df_test = pd.DataFrame(columns=['file','x','y'])

    if not evaluate_all:
        df_files_test = []
        df_files_train =[]

        df_locations_tr_x = []
        df_locations_tr_y = []

        df_locations_ts_x = []
        df_locations_ts_y = []

        df_locations = torch.zeros((traj_len, 3), dtype = torch.float)
        for count, fold in enumerate(fold_list):
            data = sio.loadmat(os.path.join(pre_dir,fold,filename))
            df_location = data['pose']
            df_locations[int(count*file_size):int((count+1)*file_size)] = torch.tensor(df_location, dtype = torch.float)

    else:
        df_files_test = []
        df_files_train =[]

    #n-40 Training 40 testing
    test_sample = len(fold_list)*10
    test_index = random.choices(range(traj_len), k=test_sample)
    train_index = list(range(traj_len))

    if not evaluate_all:
        df_locations_tr_x.extend(list(df_locations[train_index,0]))
        df_locations_tr_y.extend(list(df_locations[train_index,2]))

        df_locations_ts_x.extend(list(df_locations[test_index,0]))
        df_locations_ts_y.extend(list(df_locations[test_index,2]))

    for indx in range(traj_len):
        if indx in test_index:
            df_files_test.append(all_files[indx])
        df_files_train.append(all_files[indx])

    if not evaluate_all:
        df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                                    columns =['file','x', 'y'])
        df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                                    columns =['file','x', 'y'])
    else:
        df_train = pd.DataFrame(list(zip(df_files_train)),
                                                    columns =['file'])
        df_test = pd.DataFrame(list(zip(df_files_test)),
                                                    columns =['file'])
    print("Number of training submaps: "+str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

    #construct_query_dict(df_train,len(folders),"evaluation_database.pickle",False)
    if inside == False:
        if not evaluate_all:
            construct_query_dict(df_train, df_train, len(fold_list), traj_len,"generating_queries/evaluation_database.pickle", "generating_queries/evaluation_query.pickle", nn_ind, r_mid, r_ind, True, evaluate_all) 
        else:
            construct_query_dict(df_train, df_train, len(fold_list), traj_len,"generating_queries/evaluation_database_full.pickle", "generating_queries/evaluation_query_full.pickle", nn_ind, r_mid, r_ind, True, evaluate_all)
    else:
        if not evaluate_all:
            construct_query_dict(df_train, df_train, len(fold_list), traj_len,"evaluation_database.pickle", "evaluation_query.pickle", nn_ind, r_mid, r_ind, True, evaluate_all) 
        else:
            construct_query_dict(df_train, df_train, len(fold_list), traj_len,"evaluation_database_full.pickle", "evaluation_query_full.pickle", nn_ind, r_mid, r_ind, True, evaluate_all)
if __name__ == "__main__":
    generate(1, evaluate_all=False)