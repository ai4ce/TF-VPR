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

def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################


def construct_query_dict(df_files, df_indice, filename):
    queries = {}
    
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":df_centroids.iloc[i]['file'],
                      "positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def direction_filter(positives, df_centroids, index):
    filted_pos = []
    ref_roll = df_centroids.iloc[index]['roll']
    ref_pitch = df_centroids.iloc[index]['pitch']
    ref_yaw = df_centroids.iloc[index]['yaw']
    for pos in positives:
        pos_roll = df_centroids.iloc[pos]['roll']
        pos_pitch = df_centroids.iloc[pos]['pitch']
        pos_yaw = df_centroids.iloc[pos]['yaw']
        if abs(ref_roll- pos_roll)< np.pi /2 and abs(ref_pitch- pos_pitch)< np.pi /2 and abs(ref_roll- pos_roll)< np.pi /2:
            filted_pos.append(pos)
    return positives

def construct_dict(df_files, df_all, df_index, filename, folder_sizes, all_folder_sizes, folder_num, all_folders, pre_dir, definite_positives=None):
    queries = {}
    for num in range(folder_num):
        #print("df_files:"+str(len(df_files)))
        if num == 0:
            overhead = 0
            file_overhead = 0
        else:
            overhead = 0
            file_overhead = 0
            for i in range(num):
                overhead = overhead + all_folder_sizes[i]
                file_overhead = file_overhead + folder_sizes[i]

        df_centroids = df_all[overhead:overhead + all_folder_sizes[num]]
        df_folder_index = df_index[file_overhead: file_overhead + folder_sizes[num]]
        
        tree = KDTree(df_centroids[['x','y','z']])
        ind_r = tree.query_radius(df_centroids[['x','y','z']], r=1.8)

        for ind in range(len(df_folder_index)):
            i = df_folder_index[ind]
            radius = 0.2
            n_radius = 1.8
            ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=radius)
            query = df_centroids.iloc[i]["file"]
            pre_positives = np.setdiff1d(ind_nn[i],[i]).tolist()
            positives = direction_filter(pre_positives, df_centroids, i)
            negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
            
            count = 0
            while(len(positives)<4 or len(positives)>20):
                if len(positives)<4:
                    radius = radius+0.2
                    ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=radius)
                    query = df_centroids.iloc[i]["file"]
                    positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                    positives = direction_filter(positives, df_centroids, i)
                elif len(positives)>20:
                    radius = radius - 0.02
                    ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=radius)
                    query = df_centroids.iloc[i]["file"]
                    positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                    positives = direction_filter(positives, df_centroids, i)
                if count>100:
                    assert(0)

            while(len(negatives)<18):
                n_radius = n_radius - 0.1
                print("n_radius:"+str(n_radius))
                ind_r = tree.query_radius(df_centroids[['x','y','z']], r=n_radius)
                negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
                print("negatives:"+str(negatives))
                assert(n_radius>=0)
            assert(len(positives)>=2)
            assert(len(negatives)>=18)

            queries[ind+file_overhead] = {"query":df_centroids.iloc[i]['file'],
                          "positives":positives,"negatives":negatives}
            #print("query:"+str(query))
            #print("positives:"+str(positives))
            #print("negatives:"+str(len(negatives)))
    #print("queries:"+str(len(queries)))
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(definite_positives=None, inside=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = "/data2/cc_data/AVD/"
    runs_folder = "ActiveVisionDataset/"
    
    pre_dir = os.path.join(base_path, runs_folder)
    
    # Initialize pandas DataFrame

    df_train = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])
    df_test = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])
    df_all = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])
    
    df_files_test = []
    df_files_train =[]
    df_files = []

    df_locations_tr_x = []
    df_locations_tr_y = []
    df_locations_tr_z = []
    df_locations_tr_roll = []
    df_locations_tr_pitch = []
    df_locations_tr_yaw = []

    df_locations_ts_x = []
    df_locations_ts_y = []
    df_locations_ts_z = []
    df_locations_ts_roll = []
    df_locations_ts_pitch = []
    df_locations_ts_yaw = []

    df_locations_x = []
    df_locations_y = []
    df_locations_z = []
    df_locations_roll = []
    df_locations_pitch = []
    df_locations_yaw = []

    all_folders = sorted(os.listdir(pre_dir))
    folder_num = len(all_folders)

    folder_sizes_train = []
    folder_sizes_test = []
    folder_sizes = []
    filename = "gt_pose.mat"

    all_file_index = []
    test_index = []
    train_index = []
    
    for index,folder in enumerate(all_folders):
        df_locations = sio.loadmat(os.path.join(
                                   pre_dir,folder,filename))
        df_locations = df_locations['pose']
        df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
    
        file_index = list(range(df_locations.shape[0]))
        folder_sizes.append(df_locations.shape[0])

        if index == 0:
            overhead = 0
        else:
            overhead = 0
            for i in range(index):
                overhead = overhead + folder_sizes[i]

        #n Training 10 testing
        all_file_index.extend(list(file_index))

        tst_index = random.sample(range(len(df_locations)), k=10)
        test_index.extend(list(tst_index))
        
        trn_index = list(range(df_locations.shape[0]))
        for ts_ind in tst_index:
            trn_index.remove(ts_ind)
        train_index.extend(list(trn_index))

        folder_sizes_train.append(len(trn_index))
        folder_sizes_test.append(10)

        df_locations_tr_x.extend(list(df_locations[trn_index,0]))
        df_locations_tr_y.extend(list(df_locations[trn_index,1]))
        df_locations_tr_z.extend(list(df_locations[trn_index,2]))
        df_locations_tr_roll.extend(list(df_locations[trn_index,3]))
        df_locations_tr_pitch.extend(list(df_locations[trn_index,4]))
        df_locations_tr_yaw.extend(list(df_locations[trn_index,5]))

        df_locations_ts_x.extend(list(df_locations[tst_index,0]))
        df_locations_ts_y.extend(list(df_locations[tst_index,1]))
        df_locations_ts_z.extend(list(df_locations[tst_index,2]))
        df_locations_ts_roll.extend(list(df_locations[tst_index,3]))
        df_locations_ts_pitch.extend(list(df_locations[tst_index,4]))
        df_locations_ts_yaw.extend(list(df_locations[tst_index,5]))
        
        df_locations_x.extend(list(df_locations[file_index,0]))
        df_locations_y.extend(list(df_locations[file_index,1]))
        df_locations_z.extend(list(df_locations[file_index,2]))
        df_locations_roll.extend(list(df_locations[file_index,3]))
        df_locations_pitch.extend(list(df_locations[file_index,4]))
        df_locations_yaw.extend(list(df_locations[file_index,5]))

        all_files = list(sorted(os.listdir(os.path.join(pre_dir,folder,"jpg_rgb"))))

        for (indx, file_) in enumerate(all_files): 
            if indx in tst_index:
                df_files_test.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))
            else:
                df_files_train.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))
            df_files.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))
        
    #print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
    df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y, df_locations_tr_z,df_locations_tr_roll,df_locations_tr_pitch,df_locations_tr_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y, df_locations_ts_z, df_locations_ts_roll, df_locations_ts_pitch, df_locations_ts_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    df_all = pd.DataFrame(list(zip(df_files, df_locations_x, df_locations_y, df_locations_z, df_locations_roll, df_locations_pitch, df_locations_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    

    if inside == True:
        construct_dict(df_train, df_all, train_index, "training_queries_baseline.pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_test, df_all, test_index, "test_queries_baseline.pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_all, df_all, all_file_index, "db_queries_baseline.pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir)
    else:
        construct_dict(df_train, df_all, train_index, "generating_queries/training_queries_baseline.pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_test, df_all, test_index, "generating_queries/test_queries_baseline.pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_all, df_all, all_file_index, "generating_queries/db_queries_baseline.pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)

if __name__ == "__main__":
    generate()
