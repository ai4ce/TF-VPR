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
    mode = "4"
    if mode == "0":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/0_Unsupervised-PointNetVlad_SOTA"
    elif mode == "1":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/1_Unsupervised-PointNetVlad_SOTA_FS"
    elif mode == "4":
        new_dir = "/mnt/NAS/home/cc/Our_method_PCL/4_Unsupervised-PointNetVlad_SOTA_AUG"
    checkpoint_dir = os.path.join(new_dir, 'results')
    #checkpoint_dir_validate = os.path.join('../../results/2D',"gt_map_validate")
    if not os.path.exists(checkpoint_dir_validate):
        os.makedirs(checkpoint_dir_validate)
    
    save_name = os.path.join(checkpoint_dir,'database'+str(temp_index)+'.npy')
    best_matrix = np.load(save_name)
    best_matrix = torch.tensor(best_matrix, dtype = torch.float64)
    best_matrix = np.array(best_matrix)
    
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
            indice_gt_without_self.remove(i+pos_i)
        except:
            pass
        assert(len(indice_gt_without_self)!=0)
        indices_gt.append(indice_gt_without_self)
    
    #### Accuracy ######
    
    indices_gt_std_per = []
    indices_10_std_per = []
    indices_5_std_per = []
    indices_1_std_per = []
    
    for i in range(len(indices_gt)):
        indice_gt_per = list(indices_gt[i])
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
        
        ori_est_i = ori_ests[i]
        ind_gt_dir = [0,0,0,0]
        for gt in gt_set:
            ori_gt = ori_ests[gt]
            ori_est_diff = ori_gt - ori_est_i
            roll_diff = ori_gt - ori_est_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) < 3.1415926/4:
                ind_gt_dir[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                ind_gt_dir[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                ind_gt_dir[2] += 1
            else:
                ind_gt_dir[3] += 1
        count = 0
        for ind_gt in ind_gt_dir:
            if ind_gt > 0:
                count += 1
        indices_gt_std_per.append(count)
        
        ind_10_dir = [0,0,0,0]
        for c_10 in correct_10:
            ori_c_10 = ori_ests[c_10]
            roll_diff = ori_c_10-ori_est_i
            
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) > 3.1415926/4:
                ind_10_dir[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                ind_10_dir[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                ind_10_dir[2] += 1
            else:
                ind_10_dir[3] +=1
        count = 0
        for ind_10 in ind_10_dir:
            if ind_10 > 0:
                count += 1
        indices_10_std_per.append(count)
        
        ind_5_dir = [0,0,0,0]
        for c_5 in correct_5:
            ori_c_5 = ori_ests[c_5]
            roll_diff = ori_c_5-ori_est_i
            
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) > 3.1415926/4:
                ind_5_dir[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                ind_5_dir[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                ind_5_dir[2] += 1
            else:                                                                          
                ind_5_dir[3] +=1
        count = 0
        for ind_5 in ind_5_dir:
            if ind_5 > 0:
                count += 1
        indices_5_std_per.append(count)

        ind_1_dir = [0,0,0,0]
        for c_1 in correct_1:
            ori_c_1 = ori_ests[c_1]
            roll_diff = ori_c_1-ori_est_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) > 3.1415926/4:
                ind_1_dir[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                ind_1_dir[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                ind_1_dir[2] += 1
            else:
                ind_1_dir[3] +=1
        
        count = 0
        for ind_1 in ind_1_dir:
            if ind_1 > 0:
                count += 1
        indices_1_std_per.append(count)
        
    indices_gt_std_per = np.array(indices_gt_std_per)
    indices_10_std_per = np.array(indices_10_std_per)
    indices_5_std_per = np.array(indices_5_std_per)
    indices_1_std_per = np.array(indices_1_std_per)

    # print("indices_10_std:"+str(indices_10_std))
    print("###############################################")
    print("indices_10_std:"+str(np.mean(indices_10_std_per/indices_gt_std_per))) 
    print("indices_5_std:"+str(np.mean(indices_5_std_per/indices_gt_std_per)))
    print("indices_1_std:"+str(np.mean(indices_1_std_per/indices_gt_std_per))) 
    print("###############################################")
    
    #print("Done")


if __name__ == "__main__":
    for i in range(100):
        validate(i)
