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
base_path = "/home/chao1804/Desktop/AVD/"
runs_folder = "ActiveVisionDataset/"
pre_dir = os.path.join(base_path, runs_folder)

filename = "gt_pose.mat"

all_folders = sorted(os.listdir(pre_dir))
folder_num = len(all_folders)

folders = []

# All runs are used for training (both full and partial)
index_list = [2,4]
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####

def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
            in_test_set = True
            break
    return in_test_set
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)

#########################################
def construct_query_dict(df_centroids, df_database, folder_num,  filename_train, filename_test, test=False):
    database_trees = []
    test_trees = []
    tree = KDTree(df_centroids[['x','y','z']])
    ind_nn = tree.query_radius(df_centroids[['x','y','z']],r=15)
    ind_r = tree.query_radius(df_centroids[['x','y','z']], r=50)
    queries_sets = []
    database_sets = []
    for folder in range(folder_num):
        queries = {}
        for i in range(len(df_centroids)//folder_num):
            temp_indx = folder*len(df_centroids)//folder_num + i
            query = df_centroids.iloc[temp_indx]["file"]
            #print("folder:"+str(folder))
            #print("query:"+str(query))
            queries[len(queries.keys())] = {"query":query,
                "x":float(df_centroids.iloc[temp_indx]['x']),"y":float(df_centroids.iloc[temp_indx]['y']), "z":float(df_centroids.iloc[temp_indx]['z']),
                "roll":float(df_centroids.iloc[temp_indx]['roll']),"pitch":float(df_centroids.iloc[temp_indx]['pitch']), "yaw":float(df_centroids.iloc[temp_indx]['yaw'])}
        queries_sets.append(queries)
        test_tree = KDTree(df_centroids[['x','y','z']])
        test_trees.append(test_tree)

    for folder in range(folder_num):
        dataset = {}
        for i in range(len(df_database)//folder_num):
            temp_indx = folder*len(df_database)//folder_num + i
            data = df_database.iloc[temp_indx]["file"]
            dataset[len(dataset.keys())] = {"query":data,
                     "x":float(df_database.iloc[temp_indx]['x']),"y":float(df_database.iloc[temp_indx]['y']), "z":float(df_database.iloc[temp_indx]['z']),
                     "roll":float(df_database.iloc[temp_indx]['roll']),"pitch":float(df_database.iloc[temp_indx]['pitch']), "yaw":float(df_database.iloc[temp_indx]['yaw']) }
        database_sets.append(dataset)
        database_tree = KDTree(df_database[['x','y','z']])
        database_trees.append(database_tree)

    if test:
        for i in range(len(database_sets)):
            tree = database_trees[i]
            for j in range(len(queries_sets)):
                if(i == j):
                    continue
                for key in range(len(queries_sets[j].keys())):
                    coor = np.array(
                        [[queries_sets[j][key]["x"],queries_sets[j][key]["y"], queries_sets[j][key]["z"]]])
                    index = tree.query_radius(coor, r=25)
                    # indices of the positive matches in database i of each query (key) in test set j
                    queries_sets[j][key][i] = index[0].tolist()
    
    print("queries_sets:"+str(queries_sets))
    print("database_sets:"+str(database_sets))
    output_to_file(queries_sets, filename_test)
    output_to_file(database_sets, filename_train)

# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])
df_test = pd.DataFrame(columns=['file','x','y','z','roll','pitch','yaw'])

df_files_test = []
df_files_train =[]

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


for folder in folders:
    df_locations = sio.loadmat(os.path.join(
        pre_dir,folder,filename))
    
    df_locations = df_locations['pose']
    df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

    #2038 Training 10 testing
    test_index = random.choices(range(len(df_locations)), k=10)
    train_index = list(range(df_locations.shape[0]))
    #for i in test_index:
    #    train_index.pop(i)
    
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


    all_files = list(sorted(os.listdir(os.path.join(pre_dir,folder,"jpg_rgb"))))

    for (indx, file_) in enumerate(all_files):
        if indx in test_index:
            df_files_test.append(os.path.join(pre_dir,folder,file_))
        df_files_train.append(os.path.join(pre_dir,folder,file_))

print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
print("df_files_test:"+str(len(df_files_test)))

df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y, df_locations_tr_z, df_locations_tr_roll, df_locations_tr_pitch, df_locations_tr_yaw)),
                                               columns =['file','x', 'y','z','roll','pitch','yaw'])
df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y, df_locations_ts_z, df_locations_ts_roll, df_locations_ts_pitch, df_locations_ts_yaw)),
                                               columns =['file','x', 'y', 'z', 'roll', 'pitch', 'yaw'])

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

print("df_train:"+str(len(df_train)))

#construct_query_dict(df_train,len(folders),"evaluation_database.pickle",False)
construct_query_dict(df_test, df_train, len(folders),"evaluation_database.pickle", "evaluation_query.pickle", True)
