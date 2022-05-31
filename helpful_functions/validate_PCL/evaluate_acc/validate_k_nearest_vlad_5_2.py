import set_path
import os
import torch
import numpy as np 
from dataset_loader import SimulatedPointCloud
from torch.utils.data import DataLoader

import utils
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

from open3d import read_point_cloud

checkpoint_dir = os.path.join('../../','data')
checkpoint_dir_validate = os.path.join('../../results/2D',"gt_map_validate")
if not os.path.exists(checkpoint_dir_validate):
    os.makedirs(checkpoint_dir_validate)

import math

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    
    return X, Y, Z

def validate(temp_index):
    mode = "5_2"
    if mode == "0":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/0_Unsupervised-PointNetVlad_time_seq2"
    elif mode == "1":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/1_Unsupervised-PointNetVlad_SOTA_FS"
    elif mode == "2":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/2_Unsupervised-PointNetVlad_aug"
    elif mode == "4":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/4_Unsupervised-PointNetVlad_SOTA_AUG"
    elif mode == "5":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/5_Unsupervised-PointNetVlad_SOTA_AUG_FS"
    elif mode == "5_2":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/5_Unsupervised-PointNetVlad_SOTA_AUG_FS_2"
    elif mode == "6":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/6_Supervised-PointNetVlad"

    checkpoint_dir = os.path.join(new_dir, 'results')
    #checkpoint_dir_validate = os.path.join('../../results/2D',"gt_map_validate")
    if not os.path.exists(checkpoint_dir_validate):
        os.makedirs(checkpoint_dir_validate)
    
    save_name = os.path.join(checkpoint_dir,'database'+str(temp_index)+'.npy')
    best_matrix = np.load(save_name)
    best_matrix = torch.tensor(best_matrix, dtype = torch.float64)
    best_matrix = np.array(best_matrix)
    print("best_matrix:"+str(best_matrix.shape))
    
    data_dir = '/home/cc/dm_data/'
    all_folders = sorted(os.listdir(data_dir))

    folders = []
    # All runs are used for training (both full and partial)
    index_list = [5,6,7,9]
    print("Number of runs: "+str(len(index_list)))
    for index in index_list:
        print("all_folders[index]:"+str(all_folders[index]))
        folders.append(all_folders[index])
    print(folders)

    all_folders = folders

    indices = []
    best_matrix_list = []

    for index,folder in enumerate(all_folders):
        nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(best_matrix[index])
        distance, indice = nbrs.kneighbors(best_matrix[index])
        best_matrix_embeded = TSNE(n_components=2).fit_transform(best_matrix[index])
        best_matrix_list.append(best_matrix_embeded)
        indices.append(indice)
    indices = np.array(indices, dtype=np.int64)
    best_matrix_list = np.array(best_matrix_list, dtype=np.float64)
        
    #############
    best_matrix_sec = best_matrix.reshape(best_matrix.shape[0]* best_matrix.shape[1], best_matrix.shape[2])
    total_len = best_matrix_sec.shape[0]
    nbrs_11 = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(best_matrix_sec)
        
    distance_11, indices_11 = nbrs_11.kneighbors(best_matrix_sec)
    distance_10, indices_10 = distance_11[:,1:11], indices_11[:,1:11]
    distance_5, indices_5 = distance_11[:,1:6], indices_11[:,1:6]
    distance_1, indices_1 = distance_11[:,1:2], indices_11[:,1:2]
    #############
    indices_gt = []

    location_ests = []
    ori_ests = []
    for index,folder in enumerate(all_folders):
        data_dir_f = os.path.join(data_dir, folder) 
        gt_file = os.path.join(data_dir_f,'gt_pose.mat')
        gt_pose = sio.loadmat(gt_file)
        gt_pose = gt_pose['pose']
        gt_location = gt_pose[:,0:2]
        orientation = gt_pose[:,-1]
        #pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
        location_est = torch.tensor(gt_location, dtype = torch.float).cpu()
        location_ests.append(gt_location)
        ori_ests.append(orientation)
    location_ests = torch.tensor(location_ests, dtype = torch.float).cpu()
    ori_ests = torch.tensor(ori_ests, dtype=torch.float).cpu()
    location_ests = location_ests.reshape(location_ests.shape[0]*location_ests.shape[1],location_ests.shape[2])
    ori_ests = ori_ests.reshape(ori_ests.shape[0]*ori_ests.shape[1])
    
    tree = KDTree(location_ests)
    indice_ = tree.query_radius(location_ests, r=15)
    
    indice_gt_temp = indice_.copy()
    indices_gt = []
    
    for i in range(len(indice_gt_temp)):
        indice_gt_without_self = list(indice_gt_temp[i]).copy()
        try:
            indice_gt_without_self.remove(i)
        except:
            pass
        assert(len(indice_gt_without_self)!=0)
        indices_gt.append(indice_gt_without_self)
    
    #### Accuracy ######
    
    indices_10_count = 0
    indices_5_count = 0
    indices_1_count = 0
    indices_gt_count = len(indices_gt)

    for i in range(len(indices_gt)):
        indice_gt_per = list(indices_gt[i])
        if len(indice_gt_per) > 0 :
            k_nearest = 10
            pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
            pos_index_range.remove(0)
            for pos_ind in pos_index_range:
                try:
                    indice_gt_per.remove(i+pos_ind)
                except:
                    pass 

        indice_10_per = list(indices_10[i])
        indice_5_per = list(indices_5[i])
        indice_1_per = list(indices_1[i])
        
        gt_set = set(indice_gt_per)
        set_10 = set(indice_10_per)
        set_5 = set(indice_5_per)
        set_1 = set(indice_1_per)
        correct_10 = set_10.intersection(gt_set)
        correct_5 = set_5.intersection(gt_set)
        correct_1 = set_1.intersection(gt_set)
        if len(correct_10) > 0:
            indices_10_count += 1
        if len(correct_5) > 0:
            indices_5_count += 1
        if len(correct_1) > 0:
            indices_1_count += 1
        
    # print("indices_10_std:"+str(indices_10_std))
    print("###############################################")
    print("Recall@10:"+str(indices_10_count/indices_gt_count))   
    print("Recall@5:"+str(indices_5_count/indices_gt_count))   
    print("Recall@1:"+str(indices_1_count/indices_gt_count))   
    print("###############################################")
    
    #print("Done")


if __name__ == "__main__":
    for i in range(100):
        validate(i)