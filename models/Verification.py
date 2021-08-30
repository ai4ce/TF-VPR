import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages/python_pcl-0.3-py3.6-linux-x86_64.egg/')
from pcl import IterativeClosestPoint
import pcl
import torch
import torch.nn as nn

import os
import numpy as np
#from open3d import read_point_cloud
import open3d as o3d
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

from PIL import Image
import cv2
#import icp_new

def detect_ISS(pointcloud):
    """
    detect iss keypoints
    :param pointcloud: o3d.geometry.Pointcloud()
    :return:
    """
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pointcloud)
    return keypoints


def folder_similar_check(folder_, all_files,df_location, top_k):
    ac_max_sim = 0
    folder_sim_index = []
    for file_ in all_files:
        src_path = os.path.join(folder_, file_)
        pc = o3d.read_point_cloud(src_path)
        src_pc = np.asarray(pc.points, dtype=np.float32)
        tgt_pc = []
        tgt_paths = []

        for (indx, file_) in enumerate(all_files):
            file_ = os.path.join(folder_, file_)
            tgt_paths.append(file_)
            pc = o3d.read_point_cloud(file_)
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
'''
def compute_fpfh(
                pointcloud_with_normal,
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100)
):
   feature_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pointcloud_with_normal, search_param=search_param)
   return feature_fpfh

def get_matches(feature_target, feature_source):
    """
    get the matching relationship through the nearest neighbor search of the descriptor.
    Only when they are the nearest neighbors will they be recorded
    :param feature_target: open3d.pipeline.registration.Feature
    :param feature_source: open3d.pipeline.registration.Feature
    :return: numpy.ndarray N x 2 [source_index, target_index
    """
    search_tree_target = o3d.geometry.KDTreeFlann(feature_target)
    search_tree_source = o3d.geometry.KDTreeFlann(feature_source)
    _, N = feature_source.data.shape
    matches = []

    for i in range(N):
        query_source = feature_source.data[:, i]
        _, nn_target_index, _ = search_tree_target.search_knn_vector_xd(query_source, 1)

        query_target = feature_target.data[:, nn_target_index[0]] 
        _, nn_source_index, _ = search_tree_source.search_knn_vector_xd(query_target, 1)

        if nn_source_index[0] == i:
            matches.append([i, nn_target_index[0]])
    matches = np.asarray(matches)

    return matches
'''
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

def filter_trusted(folder_path, all_files, src_index, compared_index, threshold=0.1):
    compared_index = list(np.setdiff1d(compared_index,src_index))
    #print("src_index:"+str(src_index))
    k_nearest = 6
    pos_index_range = np.arange(-k_nearest//2, (k_nearest//2)+1)
    pos_index = src_index + pos_index_range

    pos_pc = []
    for pos_i in pos_index:
        if (pos_i>=0 and pos_i<=len(all_files)-1):
            compared_index = list(np.setdiff1d(compared_index,pos_i))
            pc = o3d.read_point_cloud(os.path.join(folder_path, all_files[pos_i]))
            pos_pc.append(np.asarray(pc.points, dtype=np.float32))
    
    #print("compared_index:"+str(compared_index))
    pc = o3d.read_point_cloud(os.path.join(folder_path, all_files[src_index]))
    src_pc = np.asarray(pc.points, dtype=np.float32)
    #pos_pc = np.asarray(pos_pc, dtype=np.float32)
    
    if(src_pc.shape[0] != 256):
        print("Error in pointcloud shape")
    trusted_positive = []
    for c_ind in compared_index:
        pc = o3d.read_point_cloud(os.path.join(folder_path, all_files[c_ind]))
        tar_pc = np.asarray(pc.points, dtype=np.float32)
        if(similarity_filter(src_pc, tar_pc, pos_pc, threshold)):
            trusted_positive.append(c_ind)
    trusted_positive = np.array(trusted_positive, dtype=np.int32)

    #print("trusted_positive:"+str(trusted_positive))
    return trusted_positive

def similarity_filter(in_pcl, compare_pcl, pos_pcs, threshold):
    src = pcl.PointCloud(in_pcl.astype(np.float32))
    icp = src.make_IterativeClosestPoint()
    similarities = []
    angle_range = 15*np.arange(24)
    angle_range = angle_range/180 * np.pi
    min_fitness = np.Inf
    min_transf = None
    min_converged = None
    max_compare_fitness = 0
    min_estimate = None
    best_tgt = None

    for pos_pc in pos_pcs:
        min_compare_fitness = np.Inf
        for angle in angle_range:
            tgt = pos_pc.copy()
            tgt[:,:2] = rotate_point_cloud(tgt[:,:2], angle)
            tgt = pcl.PointCloud(tgt.astype(np.float32))
            converged, transf, estimate, fitness = icp.icp(src, tgt, max_iter=10)
            
            if fitness < min_compare_fitness:
                min_compare_fitness = fitness
        if min_compare_fitness > max_compare_fitness:
            max_compare_fitness = min_compare_fitness
    for angle in angle_range:
        tgt = compare_pcl.copy()
        tgt[:,:2] = rotate_point_cloud(tgt[:,:2], angle)
        tgt = pcl.PointCloud(tgt.astype(np.float32))
        converged, transf, estimate, fitness = icp.icp(src, tgt, max_iter=10)
        if fitness < min_fitness:
            min_fitness = fitness
            min_trans = transf
            min_converged =  converged
            min_estimate = estimate
            best_tgt = rotate_point_cloud(compare_pcl[:,:2], angle).astype(np.float32)
            best_tgt = np.dot(best_tgt, min_trans[:2,:2])
            
            #best_tgt[:,0] += min_trans[0][3]
            #best_tgt[:,1] += min_trans[1][3]

    #print("max_compare_fitness:"+str(max_compare_fitness))
    #print("min_fitness:"+str(min_fitness))
    if threshold*max_compare_fitness > min_fitness:
        return True
    else:
        return False
    '''
    print("min_fitness:"+str(min_fitness))
    print("min_trans:"+str(min_trans))
    print("min_converged:"+str(min_converged))
    print("min_estimate:"+str(min_estimate))
    
    best_tgt_new = np.zeros_like(in_pcl, dtype=np.float32)
    best_tgt_new[:,:2] = best_tgt
    
    #comp_pc = best_tgt_new.clone()
    comp_pc = np.concatenate((in_pcl, best_tgt_new),axis=0)
    x_min, x_max, y_min, y_max = min(comp_pc[:,0]), max(comp_pc[:,0]), min(comp_pc[:,1]), max(comp_pc[:,1])
    x_interval = (x_max - x_min)/128
    y_interval = (y_max - y_min)/128
    x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
    y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
    im_compare_pcl = np.zeros((128,128),dtype=np.float32)
    im_compare_pcl = plot_image(comp_pc, im_compare_pcl, x_range, y_range)
    im_compare_pcl = Image.fromarray(im_compare_pcl)
    if im_compare_pcl.mode != 'RGB':
        im_compare_pcl = im_compare_pcl.convert('RGB')
    im_compare_pcl_gray = cv2.cvtColor(np.array(im_compare_pcl), cv2.COLOR_RGB2GRAY)
    cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/result/im_compare_pcl_"+str(index)+".jpg", im_compare_pcl_gray)

    #target = pcl.PointCloud(best_tgt_new.astype(np.float32))
    T, distance, _  = icp_new.icp(in_pcl, best_tgt_new)
    print("distance:"+str(distance.mean()))
    #print("distance:"+str(distance))
    #print("T:"+str(T))

    #print("min_trans:"+str(min_trans))

    in_pcl_o3d = o3d.geometry.PointCloud()
    in_pcl_o3d.points = o3d.utility.Vector3dVector(in_pcl[:, 0:3])
    in_pcl_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    best_tgt_new = np.zeros_like(in_pcl, dtype=np.float32)
    best_tgt_new[:,:2] = best_tgt
    
    best_pcl_o3d = o3d.geometry.PointCloud()
    best_pcl_o3d.points = o3d.utility.Vector3dVector(best_tgt_new[:, 0:3])
    assert(0)
    keypoints_sounsrce = detect_ISS(in_pcl_o3d)
    assert(0)
    descriptor_source = compute_fpfh(keypoints_source)
    
    keypoints_target = detect_ISS(best_pcl_o3d)
    descriptor_target = compute_fpfh(keypoints_target)

    in_pcl = np.histogram2d(in_pcl[:,0], in_pcl[:,1], bins=[128,128])
    in_pcl = np.array(in_pcl)
    print(np.array(in_pcl[2]).shape)
    assert(0)
    in_pcl = in_pcl - in_pcl.min(axis=0)
    in_pcl = in_pcl / np.abs(in_pcl).max(axis=0)
    print("in_pcl:"+str(in_pcl[2]))
    x_stack = np.concatenate((in_pcl[:,0],compare_pcl[:,0], best_tgt[:,0]),axis=0)
    y_stack = np.concatenate((in_pcl[:,1],compare_pcl[:,1], best_tgt[:,1]),axis=0)

    x_min, x_max, y_min, y_max= min(x_stack), max(x_stack), min(y_stack), max(y_stack)
    x_interval = (x_max - x_min)/128
    y_interval = (y_max - y_min)/128
    x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
    y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)

    #print("x_range:"+str(len(x_range)))
    #print("y_range:"+str(len(y_range)))
    im_in_pcl = np.zeros((128,128),dtype=np.float32)
    im_compare_pcl = np.zeros((128,128),dtype=np.float32)
    im_best_tgt = np.zeros((128,128),dtype=np.float32)
    #im_compare_pcl = Image.fromarray(np.array(compare_pcl))

    for ind_p in range(len(in_pcl)):
        for ind_x in range(len(x_range)-1):
            if (in_pcl[ind_p][0] >= x_range[ind_x] ) and (in_pcl[ind_p][0] < x_range[ind_x+1]):
                index_x = ind_x
        for ind_y in range(len(y_range)-1):
            if (in_pcl[ind_p][1] >= y_range[ind_y] ) and (in_pcl[ind_p][1] < y_range[ind_y+1]):
                index_y = ind_y
        im_in_pcl[index_x][index_y] = 255
   
    for ind_p in range(len(compare_pcl)):
        for ind_x in range(len(x_range)-1):
            if (compare_pcl[ind_p][0] >= x_range[ind_x] ) and (compare_pcl[ind_p][0] < x_range[ind_x+1]):
                index_x = ind_x
        for ind_y in range(len(y_range)-1):
            if (compare_pcl[ind_p][1] >= y_range[ind_y] ) and (compare_pcl[ind_p][1] < y_range[ind_y+1]):
                index_y = ind_y
        im_compare_pcl[index_x][index_y] = 255

    for ind_p in range(len(best_tgt)):
        for ind_x in range(len(x_range)-1):
            if (best_tgt[ind_p][0] >= x_range[ind_x] ) and (best_tgt[ind_p][0] < x_range[ind_x+1]):
                index_x = ind_x
        for ind_y in range(len(y_range)-1):
            if (best_tgt[ind_p][1] >= y_range[ind_y] ) and (best_tgt[ind_p][1] < y_range[ind_y+1]):
                index_y = ind_y
        im_best_tgt[index_x][index_y] = 255

    im_in_pcl = Image.fromarray(im_in_pcl)
    im_compare_pcl = Image.fromarray(im_compare_pcl)
    im_best_tgt = Image.fromarray(im_best_tgt)

    if im_in_pcl.mode != 'RGB':
        im_in_pcl = im_in_pcl.convert('RGB')
    if im_compare_pcl.mode != 'RGB':
        im_compare_pcl = im_compare_pcl.convert('RGB')
    if im_best_tgt.mode != 'RGB':
        im_best_tgt = im_best_tgt.convert('RGB')
    
    im_in_pcl_gray = cv2.cvtColor(np.array(im_in_pcl), cv2.COLOR_RGB2GRAY)
    im_in_pcl_origin= cv2.cvtColor(im_in_pcl_gray, cv2.COLOR_GRAY2BGR)
    im_in_pcl_rgb = cv2.cvtColor(im_in_pcl_origin, cv2.COLOR_BGR2RGB)
    im_compare_pcl_gray = cv2.cvtColor(np.array(im_compare_pcl), cv2.COLOR_RGB2GRAY)
    im_compare_pcl_origin= cv2.cvtColor(im_compare_pcl_gray, cv2.COLOR_GRAY2BGR)
    im_compare_pcl_rgb = cv2.cvtColor(im_compare_pcl_origin, cv2.COLOR_BGR2RGB)
    
    im_best_tgt_gray = cv2.cvtColor(np.array(im_best_tgt), cv2.COLOR_RGB2GRAY)
    im_best_tgt_origin= cv2.cvtColor(im_best_tgt_gray, cv2.COLOR_GRAY2BGR)
    im_best_tgt_rgb = cv2.cvtColor(im_best_tgt_origin, cv2.COLOR_BGR2RGB)
    
    kp_left, des_left = SIFT(im_in_pcl_gray)
    kp_right, des_right = SIFT(im_best_tgt_gray)

    #matches = matcher(kp_left, des_left, im_in_pcl_rgb, kp_left, des_left, im_in_pcl_rgb, threshold=0.5)
    matches = matcher(kp_left, des_left, im_in_pcl_rgb, kp_right, des_right, im_best_tgt_rgb, threshold=0.5)
    print("no of matches:"+str(len(matches)))
    #im_in_pcl.save("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_in_pcl.jpg")
    #im_compare_pcl.save("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_compare_pcl.jpg")

    cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_in_pcl.jpg", im_in_pcl_gray)
    cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_compare_pcl.jpg", im_compare_pcl_gray)
    cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_best_tgt_rgb.jpg", im_best_tgt_gray)
    assert(0)
    '''

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

def SIFT(img):
    siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
    # siftDetector= cv2.SIFT_create()  # depends on OpenCV version
    
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des


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

def plot_image(compare_pcl, im_compare_pcl, x_range, y_range):
    for ind_p in range(len(compare_pcl)):
        for ind_x in range(len(x_range)-1):
            if (compare_pcl[ind_p][0] >= x_range[ind_x] ) and (compare_pcl[ind_p][0] < x_range[ind_x+1]):
                index_x = ind_x
        
        for ind_y in range(len(y_range)-1):
            if (compare_pcl[ind_p][1] >= y_range[ind_y] ) and (compare_pcl[ind_p][1] < y_range[ind_y+1]):
                index_y = ind_y
    
        im_compare_pcl[index_x][index_y] = 255
    return im_compare_pcl

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
        '''
        print("df_locations[0]:"+str(df_locations[0]))
        print("df_locations[1]:"+str(torch.norm(df_locations[1,:2]-df_locations[0,:2])))
        print("df_locations[1545]:"+str(torch.norm(df_locations[1545,:2]-df_locations[0,:2])))
        print("df_locations[1545]:"+str(df_locations[1545]))

        print("df_locations[2]:"+str(df_locations[2]))
        print("df_locations[4]:"+str(torch.norm(df_locations[4,:2]-df_locations[2,:2])))
        print("df_locations[523:"+str(torch.norm(df_locations[523,:2]-df_locations[2,:2])))
        print("df_locations[1546:"+str(torch.norm(df_locations[1546,:2]-df_locations[2,:2])))
        print("check:"+str(df_locations[1546,:2]-df_locations[2,:2]))
        print("df_locations[523]:"+str(df_locations[523]))
        print("df_locations[1546]:"+str(df_locations[1546]))
        print("df_locations[2:"+str(torch.norm(df_locations[2]-df_locations[0])))
        print("df_locations[3:"+str(torch.norm(df_locations[3]-df_locations[0])))
        print("df_locations[4:"+str(torch.norm(df_locations[4]-df_locations[0])))
        print("df_locations[5:"+str(torch.norm(df_locations[5]-df_locations[0])))
        
        print("df_locations[10:"+str(torch.norm(df_locations[10]-df_locations[0])))
        print("df_locations[12:"+str(torch.norm(df_locations[12]-df_locations[0])))
        print("df_locations[13:"+str(torch.norm(df_locations[13]-df_locations[0])))
        print("df_locations[47:"+str(torch.norm(df_locations[47]-df_locations[0])))
        '''
        #print("all_files[0]:"+str(all_files[0]))
        pc = o3d.io.read_point_cloud(os.path.join(pre_dir,folder,all_files[0]))
        pc = np.asarray(pc.points, dtype=np.float32)
        results = []
        for index, all_file in enumerate(all_files):
            print("all_file:"+str(all_file))
            comp_pc = o3d.io.read_point_cloud(os.path.join(pre_dir,folder,all_file))
            comp_pc = np.asarray(comp_pc.points, dtype=np.float32)
            result = similarity_filter(pc, comp_pc, 0.5)
            if result==True:
                results.append(index)
                print("results:"+str(results))
        assert(0)
        #folder_sim = folder_similar_check(folder_, all_files, df_locations, top_k)
        #sim_array.append(folder_sim)

    #sim_array = np.asarray(sim_array)
    #save_name = '/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate/validate.npy'
    #np.save(save_name, sim_array)
    print("Done")
