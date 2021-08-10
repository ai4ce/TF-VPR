import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages/python_pcl-0.3-py3.6-linux-x86_64.egg/')
from pcl import IterativeClosestPoint
import pcl
import torch
import torch.nn as nn

import os
import numpy as np
from open3d import read_point_cloud
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def folder_similar_check(folder_, all_files,df_location, top_k):
    ac_max_sim = 0
    folder_sim_index = []
    for file_ in all_files:
        src_path = os.path.join(folder_, file_)
        pc = read_point_cloud(src_path)
        src_pc = np.asarray(pc.points, dtype=np.float32)
        tgt_pc = []
        tgt_paths = []

        for (indx, file_) in enumerate(all_files):
            file_ = os.path.join(folder_, file_)
            tgt_paths.append(file_)
            pc = read_point_cloud(file_)
            pc = np.asarray(pc.points, dtype=np.float32)
            if(pc.shape[0] != 256):
                print("Error in pointcloud shape")
            tgt_pc.append(pc)
        tgt_pc = np.asarray(tgt_pc, dtype=np.float32)
        sim_index, max_sim = similarity_check(src_pc, tgt_pc, top_k, df_location)
        if float(max_sim) > ac_max_sim:
            ac_max_sim = max_sim
        print("ac_max_sim:"+str(ac_max_sim))
        sim_index = np.asarray(sim_index)
        folder_sim_index.append(sim_index)
    
    folder_sim_index = np.asarray(folder_sim_index)

    return folder_sim_index
 
def rotate_point_cloud(batch_data, rotation_angle):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
    BxNx2 array, original batch of point clouds
    Return:
    BxNx2 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    #rotation_angle = (np.random.uniform()*2*np.pi) - np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval], 
                      [sinval, cosval]])
    
    #for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
    #shape_pc = batch_data[k, ...]
    rotated_data = np.dot(batch_data, rotation_matrix)     
    return rotated_data

def filter_trusted(folder_path, all_files, src_index, compared_index):
    pc = read_point_cloud(os.path.join(folder_path, all_files[src_index]))
    src_pc = np.asarray(pc.points, dtype=np.float32)
    if(src_pc.shape[0] != 256):
        print("Error in pointcloud shape")
    trusted_positive = []
    for c_ind in compared_index:
        pc = read_point_cloud(os.path.join(folder_path, all_files[c_ind]))
        tar_pc = np.asarray(pc.points, dtype=np.float32)
        if(similarity_filter(src_pc, tar_pc, 0.003)):
            trusted_positive.append(c_ind)
    trusted_positive = np.array(trusted_positive, dtype=np.float32)
    return trusted_positive

def similarity_filter(in_pcl, compare_pcl, threshold):
    src = pcl.PointCloud(in_pcl.astype(np.float32))
    icp = src.make_IterativeClosestPoint()
    similarities = []
    angle_range = 15*np.arange(24)
    angle_range = angle_range/180 * np.pi
    min_fitness = np.Inf
    for angle in angle_range:
        tgt = compare_pcl
        tgt[:,:2] = rotate_point_cloud(tgt[:,:2], angle)
        tgt = pcl.PointCloud(tgt.astype(np.float32))
        converged, transf, estimate, fitness = icp.icp(src, tgt, max_iter=1)
        if fitness < min_fitness:
            min_fitness = fitness

    if min_fitness < threshold:
        return True
    else:
        return False


def similarity_check(in_pcl, compare_pcls, top_k, df_location):
    src = pcl.PointCloud(in_pcl.astype(np.float32))   
    icp = src.make_IterativeClosestPoint()
    similarities = []
    angle_range = 15*np.arange(24)
    angle_range = angle_range/180 * np.pi

    for i in range(compare_pcls.shape[0]):
        min_fitness = np.Inf
        for angle in angle_range:
            tgt = compare_pcls[i]
            tgt[:,:2] = rotate_point_cloud(tgt[:,:2], angle)
            print("tgt:"+str(tgt))
            print("src:"+str(in_pcl.astype(np.float32)))
            assert(0)
            tgt = pcl.PointCloud(tgt.astype(np.float32))
            converged, transf, estimate, fitness = icp.icp(
                            src, tgt, max_iter=1)
            if fitness < min_fitness:
                min_fitness = fitness
        similarities.append(min_fitness)
    
    similarities = torch.tensor(similarities, dtype=torch.float32)
    similarities, sim_index = torch.topk(similarities, top_k, largest=False)
    max_sim = max(similarities)

    return sim_index, max_sim
    '''
    if not os.path.exists('/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate'):
        os.makedirs('/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate')
 
    save_dir = '/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate'

    traj = df_location[:,:2]
    len_index = len(sim_index)
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    #x_ind = np.repeat(np.arange(0, N, downsample_size), K)
    y_ind = np.asarray(sim_index,dtype=np.float32)
    x_ind = np.zeros_like(y_ind)
    edges = np.concatenate((traj[x_ind], traj[y_ind]), axis=1).reshape(-1, 2, 2)
    lc = mc.LineCollection(edges, colors='r', linewidths=1)
    ax.add_collection(lc)
    ax.scatter(traj[:, 0], traj[:, 1], zorder=3, s=1, marker='o')
    print("saving figure to "+str(os.path.join(save_dir, 'connection_.png')))
    
    plt.savefig(os.path.join(save_dir, 'connection_.png'), bbox_inches='tight')
    plt.close()

    
    # src plot
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(in_pcl[:, 0], in_pcl[:, 1], zorder=2)
    print("saving src_image.png")
    plt.savefig(os.path.join('/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate', 'src_image.png'), bbox_inches='tight')
    plt.close()

    # tgt plot
    for index in sim_index:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(compare_pcls[index][:, 0], compare_pcls[index][:, 1], zorder=2)
        plt.savefig(os.path.join('/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate', 'tgt_image_'+str(int(index))+'.png'), bbox_inches='tight')
        plt.close()
        print("saving tgt_image_"+str(int(index))+".png")
    '''



if __name__ == "__main__":
    runs_folder = "dm_data/"
    top_k = 50

    cc_dir = "/home/cc/"
    pre_dir = os.path.join(cc_dir,runs_folder)
    all_folders = sorted(os.listdir(os.path.join(cc_dir,runs_folder)))

    folders = []
    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    print("Number of runs: "+str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])

    sim_array = []
    for folder in folders:
        all_files = list(sorted(os.listdir(os.path.join(cc_dir,runs_folder,folder))))
        all_files.remove('gt_pose.mat')
        all_files.remove('gt_pose.png')
        folder_size = len(all_files)
        data_index = list(range(folder_size))

        #GT 
        folder_ = os.path.join(pre_dir,folder)
        gt_mat = os.path.join(folder_, 'gt_pose.mat')
        df_locations = sio.loadmat(gt_mat)
        df_locations = df_locations['pose']
        df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()
        
        folder_sim = folder_similar_check(folder_, all_files, df_locations, top_k)
        sim_array.append(folder_sim)

    sim_array = np.asarray(sim_array)
    save_name = '/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate/validate.npy'
    np.save(save_name, sim_array)
    print("Done")
