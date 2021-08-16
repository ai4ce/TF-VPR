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

def construct_dict(df_files, filename, folder_sizes, all_folder_sizes, folder_num, all_folders, pre_dir, definite_positives=None):
    print("len_df_files:"+str(len(df_files)))
    queries = {}
    for num in range(folder_num):
        #print("df_files:"+str(len(df_files)))
        if num == 0:
            overhead = 0
        else:
            overhead = 0
            for i in range(num):
                overhead = overhead + folder_sizes[i]
        
        df_centroids = df_files[overhead:overhead + folder_sizes[num]]
        tree = KDTree(df_centroids[['x','y','z']])
        ind_r = tree.query_radius(df_centroids[['x','y','z']], r=50)

        for i in range(len(df_centroids)):
            radius = 0.5
            ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=radius)
            query = df_centroids.iloc[i]["file"]
            pre_positives = np.setdiff1d(ind_nn[i],[i]).tolist()
            positives = direction_filter(pre_positives, df_centroids, i)
            negatives = np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
            random.shuffle(negatives)

            while(len(positives)<4):
                radius = radius+0.5
                ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=radius)
                query = df_centroids.iloc[i]["file"]
                positives = np.setdiff1d(ind_nn[i],[i]).tolist()
                positives = direction_filter(positives, df_centroids, i)
            
            queries[i+overhead] = {"query":df_centroids.iloc[i]['file'],
                          "positives":positives,"negatives":negatives}
            #print("query:"+str(query))
            #print("positives:"+str(positives))
            #print("negatives:"+str(max(negatives)))
            '''
            max_dis = 0
            for pos in positives:
                x_delta = df_centroids.iloc[pos+overhead]['x'] - df_centroids.iloc[i]['x']
                y_delta = df_centroids.iloc[pos+overhead]['y'] - df_centroids.iloc[i]['y']
                dis = math.sqrt(x_delta**2 + y_delta**2)
                print("dis:"+str(dis))
            if dis > max_dis:
                max_dis = dis
            '''
            #print("negatives:"+str(len(negatives)))
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def generate(data_index, definite_positives=None, inside=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = "/home/chao1804/Desktop/AVD/"
    runs_folder = "ActiveVisionDataset/"
    
    pre_dir = os.path.join(base_path, runs_folder)
    
    # Initialize pandas DataFrame

    df_train = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])
    df_test = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])

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

    for folder in all_folders:
        df_locations = sio.loadmat(os.path.join(
                       pre_dir,folder,filename))
        
        df_locations = df_locations['pose']
        df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
        
        #n Training 10 testing
        file_index = list(range(df_locations.shape[0]))
        test_index = random.sample(range(len(df_locations)), k=10)
        train_index = list(range(df_locations.shape[0]))
        for ts_ind in test_index:
            train_index.remove(ts_ind)
        
        folder_sizes_train.append(len(train_index))
        folder_sizes_test.append(10)
        folder_sizes.append(df_locations.shape[0])

        df_locations_tr_x.extend(list(df_locations[train_index,0]))
        df_locations_tr_y.extend(list(df_locations[train_index,1]))
        df_locations_tr_z.extend(list(df_locations[train_index,2]))
        df_locations_tr_roll.extend(list(df_locations[train_index,3]))
        df_locations_tr_pitch.extend(list(df_locations[train_index,4]))
        df_locations_tr_yaw.extend(list(df_locations[train_index,5]))

        df_locations_ts_x.extend(list(df_locations[test_index,0]))
        df_locations_ts_y.extend(list(df_locations[test_index,1]))
        df_locations_ts_z.extend(list(df_locations[test_index,2]))
        df_locations_ts_roll.extend(list(df_locations[test_index,3]))
        df_locations_ts_pitch.extend(list(df_locations[test_index,4]))
        df_locations_ts_yaw.extend(list(df_locations[test_index,5]))
        
        df_locations_x.extend(list(df_locations[file_index,0]))
        df_locations_y.extend(list(df_locations[file_index,1]))
        df_locations_z.extend(list(df_locations[file_index,2]))
        df_locations_roll.extend(list(df_locations[file_index,3]))
        df_locations_pitch.extend(list(df_locations[file_index,4]))
        df_locations_yaw.extend(list(df_locations[file_index,5]))

        all_files = list(sorted(os.listdir(os.path.join(pre_dir,folder,"jpg_rgb"))))

        for (indx, file_) in enumerate(all_files): 
            if indx in test_index:
                df_files_test.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))
            else:
                df_files_train.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))
            df_files.append(os.path.join(pre_dir,folder,"jpg_rgb",file_))

    #print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
    #print("df_files_test:"+str(len(df_files_test)))
    df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y, df_locations_tr_z,df_locations_tr_roll,df_locations_tr_pitch,df_locations_tr_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y, df_locations_ts_z, df_locations_ts_roll, df_locations_ts_pitch, df_locations_ts_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    df_files = pd.DataFrame(list(zip(df_files, df_locations_x, df_locations_y, df_locations_z, df_locations_roll, df_locations_pitch, df_locations_yaw)),
                                                           columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    

    if inside == True:
        construct_dict(df_train,"train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_test, "train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir)
        construct_dict(df_files, "train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir)
    else:
        construct_dict(df_train, "generating_queries/train_pickle/training_queries_baseline_"+str(data_index)+".pickle", folder_sizes_train, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_test, "generating_queries/train_pickle/test_queries_baseline_"+str(data_index)+".pickle", folder_sizes_test, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)
        construct_dict(df_files, "generating_queries/train_pickle/db_queries_baseline_"+str(data_index)+".pickle", folder_sizes, folder_sizes, folder_num, all_folders, pre_dir, definite_positives=definite_positives)

if __name__ == "__main__":
    for i in range(20):
        generate(i)
