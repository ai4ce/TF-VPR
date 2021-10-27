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

base_path = "/mnt/NAS/home/yuhang/videomap_v2/"
runs_folder = "Adrian"
pre_dir = os.path.join(base_path, runs_folder)

nn_ind = 15
r_mid = 25
r_ind = 50

filename = "gt_pose.mat"

all_files = list(sorted(os.listdir(pre_dir)))
all_files.remove(runs_folder+'.json')
all_files.remove('trajectory.mp4')
all_files = [i for i in all_files if not i.endswith(".npy")]

traj_len = len(all_files)

##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)

#########################################
def construct_query_dict(df_centroids, df_database, traj_len,  filename_train, filename_test, test=False):
    database_trees = []
    test_trees = []
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=nn_ind)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=r_ind)
    queries_sets = []
    database_sets = []
    # for folder in range(folder_num):
    queries = {}
    for i in range(len(df_centroids)):
        # temp_indx = folder*len(df_centroids)//folder_num + i
        query = df_centroids.iloc[i]["file"]
        #print("folder:"+str(folder))
        #print("query:"+str(query))
        queries[len(queries.keys())] = {"query":query,
            "x":float(df_centroids.iloc[i]['x']),"y":float(df_centroids.iloc[i]['y'])}
    queries_sets.append(queries)
    test_tree = KDTree(df_centroids[['x','y']])
    test_trees.append(test_tree)

    # for folder in range(folder_num):
    dataset = {}
    for i in range(len(df_database)):
        # temp_indx = folder*len(df_database)//folder_num + i
        data = df_database.iloc[i]["file"]
        dataset[len(dataset.keys())] = {"query":data,
                    "x":float(df_database.iloc[i]['x']),"y":float(df_database.iloc[i]['y']) }
    database_sets.append(dataset)
    database_tree = KDTree(df_database[['x','y']])
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
                    index = tree.query_radius(coor, r=r_mid)
                    # indices of the positive matches in database i of each query (key) in test set j
                    queries_sets[j][key][i] = index[0].tolist()
    
    print("queries_sets:"+str(queries_sets))
    print("database_sets:"+str(database_sets))
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


df_locations = sio.loadmat(os.path.join(
    pre_dir,filename))

df_locations = df_locations['pose']
df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

#2038 Training 10 testing
test_index = random.choices(range(traj_len), k=10)
train_index = list(range(traj_len))
#for i in test_index:
#    train_index.pop(i)

df_locations_tr_x.extend(list(df_locations[train_index,0]))
df_locations_tr_y.extend(list(df_locations[train_index,1]))

df_locations_ts_x.extend(list(df_locations[test_index,0]))
df_locations_ts_y.extend(list(df_locations[test_index,1]))

for index in range(traj_len):
    file_ = 'panoimg_'+str(indx)+'.png'
    if indx in test_index:
        df_files_test.append(os.path.join(file_))
    df_files_train.append(os.path.join(file_))


print("df_locations_tr_x:"+str(len(df_locations_tr_x)))
print("df_files_test:"+str(len(df_files_test)))

df_train = pd.DataFrame(list(zip(df_files_train, df_locations_tr_x, df_locations_tr_y)),
                                               columns =['file','x', 'y'])
df_test = pd.DataFrame(list(zip(df_files_test, df_locations_ts_x, df_locations_ts_y)),
                                               columns =['file','x', 'y'])

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

print("df_train:"+str(len(df_train)))

#construct_query_dict(df_train,len(folders),"evaluation_database.pickle",False)
construct_query_dict(df_test, df_train, traj_len,"evaluation_database.pickle", "evaluation_query.pickle", True)
