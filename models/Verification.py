import sys
# sys.path.append('/usr/local/lib/python3.6/dist-packages/python_pcl-0.3-py3.6-linux-x86_64.egg/')
# from pcl import IterativeClosestPoint
# import pcl
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
import copy
from sklearn.neighbors import NearestNeighbors
from math import atan2
sys.path.insert(0, '../')
from numpy import linalg as LA
import config as cfg

#import icp_new

def point2plane_metrics_2D(p,q,v):
    """
    Point-to-plane minimization
    Chen, Y. and G. Medioni. “Object Modelling by Registration of Multiple Range Images.” 
    Image Vision Computing. Butterworth-Heinemann . Vol. 10, Issue 3, April 1992, pp. 145-155.
    
    Args:
        p: Nx2 matrix, moving point locations
        q: Nx2 matrix, fixed point locations
        v:Nx2 matrix, fixed point normal
    Returns:
        R: 2x2 matrix
        t: 2x1 matrix
    """
    assert q.shape[1] == p.shape[1] == v.shape[1] == 2, 'points must be 2D'
    
    p,q,v = np.array(p),np.array(q),np.array(v)
    c = np.expand_dims(np.cross(p,v),-1)
    cn = np.concatenate((c,v),axis=1)  # [ci,nix,niy]
    C = np.matmul(cn.T,cn)
    if np.linalg.cond(C)>=1/sys.float_info.epsilon:
        # handle singular matrix
        raise ArithmeticError('Singular matrix')
    
#     print(C.shape)
    qp = q-p
    b = np.array([
        [(qp*cn[:,0:1]*v).sum()],
        [(qp*cn[:,1:2]*v).sum()],
        [(qp*cn[:,2:]*v).sum()],
    ])

    X = np.linalg.solve(C, b)
    cos_ = np.cos(X[0])[0]
    sin_ = np.sin(X[0])[0]
    R = np.array([
        [cos_,-sin_],
        [sin_,cos_]
    ])
    t = np.array(X[1:])
    return R,t

def estimate_normal_eig(data):
    """
    Computes the vector normal to the k-dimensional sample points
    """
    data -= np.mean(data,axis=0)
    data = data.T
    A = np.cov(data)
    w,v = np.linalg.eig(A)
    idx = np.argmin(w)
    v = v[:,idx]
    v /= np.linalg.norm(v,2)
    return v

def surface_normal(pc,n_neighbors=6):
    """
    Estimate point cloud surface normal
    Args:
        pc: Nxk matrix representing k-dimensional point cloud
    """
    
    n_points,k = pc.shape
    v = np.zeros_like(pc)
    
    # nn search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(pc)
    _, indices = nbrs.kneighbors(pc)
    neighbor_points = pc[indices]
    for i in range(n_points):
        # estimate surface normal
        v_tmp = estimate_normal_eig(neighbor_points[i,])
        v_tmp[abs(v_tmp)<1e-5] = 0
        if v_tmp[0] < 0:
            v_tmp *= -1
        v[i,:] = v_tmp
    return v

def icp(src,dst,nv=None,n_iter=100,init_pose=[0,0,0],torlerance=1e-6,metrics='point',verbose=False):
    '''
    Currently only works for 2D case
    Args:
        src: <Nx2> 2-dim moving points
        dst: <Nx2> 2-dim fixed points
        n_iter: a positive integer to specify the maxium nuber of iterations
        init_pose: [tx,ty,theta] initial transformation
        torlerance: the tolerance of registration error
        metrics: 'point' or 'plane'
        
    Return:
        src: transformed src points
        R: rotation matrix
        t: translation vector
        R*src + t
    '''
    n_src = src.shape[0]
    if metrics == 'plane' and nv is None:
        nv = surface_normal(dst)

    #src = np.matrix(src)
    #dst = np.matrix(dst)
    #Initialise with the initial pose estimation
    R_init = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2])],
                   [np.sin(init_pose[2]), np.cos(init_pose[2])] 
                      ])
    t_init = np.array([[init_pose[0]],
                   [init_pose[1]]
                      ])  
    
    #src =  R_init*src.T + t_init
    src = np.matmul(R_init,src.T) + t_init
    src = src.T
    
    R,t = R_init,t_init

    prev_err = np.inf
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
    for i in range(n_iter):
        # Find the nearest neighbours
        _, indices = nbrs.kneighbors(src)

        # Compute the transformation
        if metrics == 'point':
            R0,t0 = rigid_transform_kD(src,dst[indices[:,0]])
        elif metrics=='plane':
            try:
                R0,t0 = point2plane_metrics_2D(src,dst[indices[:,0]], nv[indices[:,0]]) 
            except ArithmeticError:
                #print('Singular matrix')
                return None,src,R,t
        else:
            raise ValueError('metrics: {} not recognized.'.format(metrics))
        # Update dst and compute error
        src = np.matmul(R0,src.T) + t0
        src = src.T

        R = np.matmul(R0,R)
        t = np.matmul(R0,t) + t0
        #R = R0*R
        #t = R0*t + t0
        current_err = np.sqrt((np.array(src-dst[indices[:,0]])**2).sum()/n_src)

        if verbose:
            print('iter: {}, error: {}'.format(i,current_err))
            
        if  np.abs(current_err - prev_err) < torlerance:
            break
        else:
            prev_err = current_err
            
    return current_err,src,R,t

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

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
def cal_T_matrix(rotation_angle):
    rotated_data = np.zeros((4,4), dtype=np.float32)
    rotated_data[3,3] = 1
    rotated_data[2,2] = 1
    #rotation_angle = (np.random.uniform()*2*np.pi) - np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval], 
                      [sinval, cosval]])
    
    rotated_data[:2,:2] = rotation_matrix

    #for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
    #shape_pc = batch_data[k, ...]
    #rotated_data = np.dot(batch_data, rotation_matrix)     
    return rotated_data

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

def filter_trusted(folder_path, all_files, src_index, compared_index, threshold_ratio=0.1, cal_thresholds=None):
    compared_index = list(np.setdiff1d(compared_index,src_index))
    #print("src_index:"+str(src_index))
    k_nearest = 6
    pos_index_range = np.arange(-k_nearest//2, (k_nearest//2)+1)
    pos_index = src_index + pos_index_range

    pos_pc = []
    for pos_i in pos_index:
        if (pos_i>=0 and pos_i<=len(all_files)-1):
            compared_index = list(np.setdiff1d(compared_index,pos_i))
    
    if cal_thresholds is None:
        for pos_i in pos_index:
            if (pos_i>=0 and pos_i<=len(all_files)-1):
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
            if cal_thresholds is None:
                cal_thresholds, boolean_value = similarity_filter(src_pc, tar_pc, pos_pc, threshold_ratio)
            else:
                _, boolean_value = similarity_filter(src_pc, tar_pc, pos_pc, threshold_ratio, cal_thresholds=cal_thresholds)
            if(boolean_value):
                trusted_positive.append(c_ind)
        trusted_positive = np.array(trusted_positive, dtype=np.int32)
    else:
        pc = o3d.read_point_cloud(os.path.join(folder_path, all_files[src_index]))
        src_pc = np.asarray(pc.points, dtype=np.float32)
        trusted_positive = []
        for c_ind in compared_index:
            pc = o3d.read_point_cloud(os.path.join(folder_path, all_files[c_ind]))
            tar_pc = np.asarray(pc.points, dtype=np.float32)
            _, boolean_value = similarity_filter(src_pc, tar_pc, None, threshold_ratio, cal_thresholds=cal_thresholds)
            if(boolean_value):
                trusted_positive.append(c_ind)
        trusted_positive = np.array(trusted_positive, dtype=np.int32)
    return cal_thresholds,trusted_positive

def filter_trusted_pos(all_files, src_index, compared_index, threshold_ratio=1, cal_thresholds=None):
    compared_index = list(np.setdiff1d(compared_index,src_index))
    #print("src_index:"+str(src_index))
    k_nearest = 6
    pos_index_range = np.arange(-k_nearest//2, (k_nearest//2)+1)
    pos_index = src_index + pos_index_range
    folder_num = src_index // 2048

    # print("folder_num:"+str(folder_num))

    pos_pc = []
    for pos_i in pos_index:
        if (pos_i>=folder_num*2048 and pos_i<=(folder_num+1)*2048-1):
            compared_index = list(np.setdiff1d(compared_index,pos_i))
    
    if cal_thresholds is None:
        for pos_i in pos_index:
            if (pos_i>=folder_num*2048 and pos_i<=(folder_num+1)*2048-1):
                pc = o3d.read_point_cloud(os.path.join(all_files[pos_i]))
                pos_pc.append(np.asarray(pc.points, dtype=np.float32))

        #print("compared_index:"+str(compared_index))
        pc = o3d.read_point_cloud(os.path.join(all_files[src_index]))
        src_pc = np.asarray(pc.points, dtype=np.float32)
        #pos_pc = np.asarray(pos_pc, dtype=np.float32)
        
        #assert(0)
        '''
        if(src_pc.shape[0] != 256):
            print("Error in pointcloud shape")
        '''
        trusted_positive = []
        for c_ind in compared_index:
            pc = o3d.read_point_cloud(os.path.join(all_files[c_ind]))
            tar_pc = np.asarray(pc.points, dtype=np.float32)
            if cal_thresholds is None:
                cal_thresholds, boolean_value = similarity_filter(src_pc, tar_pc, pos_pc, threshold_ratio)
            else:
                _, boolean_value = similarity_filter(src_pc, tar_pc, pos_pc, threshold_ratio, cal_thresholds=cal_thresholds)
            
            if(boolean_value):
                trusted_positive.append(c_ind)
        trusted_positive = np.array(trusted_positive, dtype=np.int32)
    else:
        pc = o3d.read_point_cloud(os.path.join(all_files[src_index]))
        src_pc = np.asarray(pc.points, dtype=np.float32)
        trusted_positive = []
        for c_ind in compared_index:
            pc = o3d.read_point_cloud(os.path.join(all_files[c_ind]))
            tar_pc = np.asarray(pc.points, dtype=np.float32)
            _, boolean_value = similarity_filter(src_pc, tar_pc, None, threshold_ratio, cal_thresholds=cal_thresholds)
            if(boolean_value):
                trusted_positive.append(c_ind)
        trusted_positive = np.array(trusted_positive, dtype=np.int32)
    return cal_thresholds,trusted_positive

def threshold_cal(in_pcl, pos_pcs, threshold_ratio):
    # src = o3d.geometry.PointCloud()
    # src.points = o3d.utility.Vector3dVector(in_pcl)
    src = in_pcl.astype(np.float32)[:,:2]

    # icp = src.make_IterativeClosestPoint()
    
    similarities = []
    
    angle_range = 30*np.arange(12)
    angle_range = angle_range/180 * np.pi
    min_fitness = np.Inf
    min_transf = None
    min_converged = None
    max_compare_fitness = 0
    overall_min_compare_fitness = np.Inf
    min_estimate = None    
    best_tgt = None

    for pos_index, pos_pc in enumerate(pos_pcs):
        min_compare_fitness = np.Inf

        tgt = pos_pc.astype(np.float32)[:,:2]
        # rotated_tgt = rotate_point_cloud(tgt[:,:2], angle)
        current_err,_,R,t = icp(src,tgt,metrics='plane')

        if current_err is None:
            current_err,_,R,t = icp(tgt,src,metrics='plane')
            R = np.linalg.inv(R)
            t = -t
            if current_err is None:
                assert(0)

        # print("reg_p2p.transformation:"+str(reg_p2p.transformation))
        if (current_err is not None) and (current_err < min_compare_fitness):
            min_compare_fitness = current_err
            min_R = R
            min_t = t
            # min_angle = angle

        # print("min_transf:"+str(min_transf))
        if min_compare_fitness > max_compare_fitness:
            max_compare_fitness = min_compare_fitness
        if min_compare_fitness < overall_min_compare_fitness:
            overall_min_compare_fitness = min_compare_fitness
        #print("min_compare_fitness:"+str(min_compare_fitness))
        #print("max_compare_fitness:"+str(max_compare_fitness))
        #print("overall_min_compare_fitness:"+str(overall_min_compare_fitness))
    # cosval = np.cos(min_angle)
    # sinval = np.sin(min_angle)
    # min_R = np.array([[cosval, -sinval], 
    #                     [sinval, cosval]])

    return overall_min_compare_fitness * threshold_ratio, max_compare_fitness * threshold_ratio, min_transf, min_R, min_t

def similarity_filter(in_pcl, compare_pcl, pos_pcs, threshold_ratio, cal_thresholds=None):
    src = in_pcl.astype(np.float32)[:,:2]
    # icp = src.make_IterativeClosestPoint()
    similarities = []
    angle_range = 30*np.arange(12)
    angle_range = angle_range/180 * np.pi
    min_fitness = np.Inf
    min_transf = None
    min_converged = None
    max_compare_fitness = 0
    min_estimate = None
    best_tgt = None

    if cal_thresholds is None:
        for pos_pc in pos_pcs:
            min_compare_fitness = np.Inf
            for angle in angle_range:
                tgt = pos_pc.astype(np.float32)[:,:2]
                rotated_tgt = rotate_point_cloud(tgt[:,:2], angle)
                current_err,_,R,t = icp(src,rotated_tgt,metrics='plane')
                if current_err is None:
                    current_err,_,R,t = icp(rotated_tgt,src,metrics='plane')
            
                if current_err < min_compare_fitness:
                    min_compare_fitness = current_err
            if min_compare_fitness > max_compare_fitness:
                max_compare_fitness = min_compare_fitness
    
    for angle in angle_range:
        tgt = compare_pcl.copy()
        rotated_tgt = rotate_point_cloud(tgt[:,:2], angle)
        current_err,_,R,t = icp(src,rotated_tgt,metrics='plane')
        if current_err is None:
            current_err,_,R,t = icp(rotated_tgt,src,metrics='plane')
            if current_err is None:
                current_err = np.Inf
            # R = np.linalg.inv(R)
            # t = -t
        # tgt = pcl.PointCloud(tgt.astype(np.float32))
        # converged, transf, estimate, fitness = icp.icp(src, tgt, max_iter=1)
        if current_err < min_fitness:
            min_fitness = current_err
            # min_t = t
            # min_angle = angle
            # angle_offset = atan2(-R[0,1],R[0,0])
            # min_angle = min_angle + angle_offset

        #     cosval = np.cos(min_angle)
        #     sinval = np.sin(min_angle)
        #     min_R = np.array([[cosval, -sinval], 
        #                       [sinval, cosval]])

            # best_tgt = rotate_point_cloud(compare_pcl[:,:2], angle).astype(np.float32)
            # best_tgt = np.dot(best_tgt, min_trans[:2,:2])
            
            #best_tgt[:,0] += min_trans[0][3]
            #best_tgt[:,1] += min_trans[1][3]
    '''
    print("min_fitness:"+str(min_fitness))
    print("cal_thresholds:"+str(cal_thresholds))
    print("threshold_ratio:"+str(threshold_ratio))
    '''
    #print("max_compare_fitness:"+str(max_compare_fitness))
    #print("min_fitness:"+str(min_fitness))
    if cal_thresholds is None:
        if threshold_ratio*max_compare_fitness > min_fitness:
            return threshold_ratio*max_compare_fitness,True
        else:
            return threshold_ratio*max_compare_fitness,False
    else:
        if threshold_ratio*cal_thresholds > min_fitness:
            return cal_thresholds,True
        else:
            return cal_thresholds,False
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

def Compute_positive(flag, db_vec, index, potential_positives, potential_distributions, trusted_positives, folders, thresholds, all_files_reshape, weight, indice, epoch):
    print("index:"+str(index))
    print("all_files_reshape:"+str(len(all_files_reshape)))
    if flag:
        trusted_positive = []

        # for index2 in range(2):
        for index2 in range(db_vec.shape[1]):
            #print("indice[index][index2]:"+str(indice[index][index2]))
            #print("weight[index][index2]:"+str(weight[index][index2]))
            if (index == 1) and (index2 == 2034):
                print("pre_trusted_positive:"+str(pre_trusted_positive))
            pre_trusted_positive = np.array(indice[index][index2])[np.argsort(weight[index][index2])[::-1][:(cfg.INIT_TRUST)]]
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive,index2)
            '''
            if index2 == 0:
                print("flag is true")
                print("index2 == 0")
                print("pre_trusted_positive:"+str(pre_trusted_positive))
            '''
            folder_path = os.path.join(cfg.DATASET_FOLDER,folders[index])
            '''
            print("index2:"+str(index2))
            print("pre_trusted_positive:"+str(pre_trusted_positive))
            '''
            trusted_pos = pre_trusted_positive
            #_, trusted_pos = filter_trusted_pos(all_files_reshape, index*db_vec.shape[1]+index2, pre_trusted_positive, cal_thresholds=thresholds[index][index2])
            # print("trusted_pos:"+str(trusted_pos.tolist()))
            '''
            if index2 == 0:
                print("trusted_pos:"+str(trusted_pos))
            '''
            trusted_positive.append(trusted_pos.tolist())
        # assert(0)
        return potential_positives, potential_distributions, trusted_positive
    else:
        new_potential_positive = []
        new_potential_distribution = []
        new_trusted_positive = []
        
        print("db_vec:"+str(db_vec.shape[1]))
        # assert(0)
        for index2 in range(db_vec.shape[1]):
            if np.array(potential_positives[index][index2]).ndim == 2:
                pos_set = list(potential_positives[index][index2][0])
                pos_dis = list(potential_distributions[index][index2][0])
            else:
                pos_set = list(potential_positives[index][index2])
                pos_dis = list(potential_distributions[index][index2])
            for count,inc in enumerate(indice[index][index2]):
                if inc not in pos_set:
                    pos_set.append(inc)
                    pos_dis.append(weight[index][index2][count])
                else:
                    pos_dis[list(pos_set).index(inc)] = pos_dis[list(pos_set).index(inc)] + weight[index][index2][count] # if the element exists, distribution +1
            #print("np.argsort:",np.argsort(pos_dis)[::-1][:5])
            #print("pos_set[np.argsort(pos_dis)[:5]:]"+str(np.array(pos_set)[np.argsort(pos_dis)[::-1][:5]]))
            #assert(0)
            new_potential_positive.append(pos_set)
            new_potential_distribution.append(pos_dis)
            '''
            if data_index-1 >=5:
                data_index_constraint = 6
            else:
                data_index_constraint = data_index
            '''
            folder_path = os.path.join(cfg.DATASET_FOLDER,folders[index])
            all_files = list(sorted(os.listdir(folder_path)))
            all_files.remove('gt_pose.mat')
            all_files.remove('gt_pose.png')
            
            previous_trusted_positive = trusted_positives[index][index2]
            # print("previous_trusted_positive:"+str(previous_trusted_positive))
            if ((np.array(previous_trusted_positive).ndim) == 2) and (np.array(previous_trusted_positive).shape[0]!=0):
                    previous_trusted_positive = previous_trusted_positive[0]
            else:
                pass
            # print("previous_trusted_positive[0]:"+str(previous_trusted_positive))
            pre_trusted_positive = np.array(pos_set)[np.argsort(pos_dis)[::-1][:(cfg.INIT_TRUST)]]

            pre_trusted_positive = np.setdiff1d(pre_trusted_positive, previous_trusted_positive)
            '''
            if index2 == 0:
                print("flag is false")
                print("index2 == 0")
                print("pre_trusted_positive:"+str(pre_trusted_positive))
            '''
            # assert(0)
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive, index2)
            
            filtered_trusted_positive = pre_trusted_positive
            #_, filtered_trusted_positive = filter_trusted_pos(all_files_reshape, index*db_vec.shape[1]+index2, pre_trusted_positive, cal_thresholds=thresholds[index][index2])
            '''
            if index2 == 0:
                print("filtered_trusted_positive:"+str(filtered_trusted_positive))
            '''
            #filtered_trusted_positive = pre_trusted_positive

            if len(filtered_trusted_positive) == 0:
                trusted_positive = previous_trusted_positive
            else:
                trusted_positive = list(previous_trusted_positive)
                trusted_positive.extend(list(filtered_trusted_positive))
                trusted_positive = np.array(list(set(trusted_positive)),dtype=np.int32)
            
            new_trusted_positive.append(trusted_positive)
        # print("new_trusted_positive:"+str(len(new_trusted_positive)))
        # print("new_trusted_positive[0]:"+str(len(new_trusted_positive[0])))

        # assert(0)
        return new_potential_positive, new_potential_distribution, new_trusted_positive

def similarity_check(in_pcl, compare_pcls, top_k, df_location):
    src = pcl.PointCloud(in_pcl.astype(np.float32))   
    icp = src.make_IterativeClosestPoint()
    similarities = []
    angle_range = 30*np.arange(12)
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
    k_nearest = 6

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
    thresholds = []
    min_thresholds = []
    # for folder in folders:

    # for folder_num, folder in enumerate(folders):
    folder_num = 0
    folder = folders[0]
    # print("folder_num:"+str(folder_num))
    threshold = []
    min_threshold = []
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
    print("df_locations[0]:"+str(df_locations[0]))
    print("df_locations[1029]:"+str(LA.norm(df_locations[1029,:2]-df_locations[0,:2])))
    print("df_locations[1031]:"+str(LA.norm(df_locations[1031,:2]-df_locations[0,:2])))

    print("df_locations[518]:"+str(LA.norm(df_locations[518,:2]-df_locations[0,:2])))
    print("df_locations[1545]:"+str(LA.norm(df_locations[1545,:2]-df_locations[0,:2])))
    print("df_locations[521]:"+str(LA.norm(df_locations[521,:2]-df_locations[0,:2])))
    print("df_locations[1550]:"+str(LA.norm(df_locations[1550,:2]-df_locations[0,:2])))
    assert(0)
        
    for index, all_file in enumerate(all_files):
    # all_file = all_files[1605]
    # index = 1605

        file_len = len(all_files) 
        neigh_range = np.arange(-k_nearest//2, (k_nearest//2)+1)
        neigh_range = np.setdiff1d(neigh_range,0)
        neigh_index = []

        pc_pcd = o3d.read_point_cloud(os.path.join(pre_dir,folder,all_file))
        pc = np.asarray(pc_pcd.points, dtype=np.float32)
        
        comp_pcs = []
        for neigh in neigh_range:
            if (neigh+index>=0 and neigh+index<=file_len-1):
                # print("neigh+index:"+str(neigh+index))
                neigh_index.append(neigh+index)
                comp_pc = o3d.read_point_cloud(os.path.join(pre_dir,folder,all_files[neigh+index]))
                comp_pc = np.asarray(comp_pc.points, dtype=np.float32)
                comp_pcs.append(comp_pc)
                
        comp_pcs = np.asarray(comp_pcs, dtype=np.float32)

        min_result, max_result, min_transform, min_R, min_t = threshold_cal(pc, comp_pcs, 1)
        print("min_result:"+str(min_result))
        print("max_result:"+str(max_result))
        print("min_t:"+str(min_t))

        folder_num = 7
        folder = folders[7]
        # print("folder_num:"+str(folder_num))
        all_files = list(sorted(os.listdir(os.path.join(cc_dir,runs_folder,folder))))
        all_files.remove('gt_pose.mat')
        all_files.remove('gt_pose.png')
        folder_size = len(all_files)
        data_index = list(range(folder_size))
        test_comp_pcs = []
        test_comp_pc = o3d.read_point_cloud(os.path.join(pre_dir,folder,all_files[521]))
        test_comp_pc = np.asarray(test_comp_pc.points, dtype=np.float32)
        test_comp_pcs.append(test_comp_pc)
        test_comp_pcs = np.asarray(test_comp_pcs, dtype=np.float32)

        test_min_result, test_max_result, test_min_transform, test_min_R, test_min_t = threshold_cal(pc, test_comp_pcs, 1)
        folder_ = os.path.join(pre_dir,folder)
        gt_mat = os.path.join(folder_, 'gt_pose.mat')
        df_locations_ = sio.loadmat(gt_mat)
        df_locations_ = df_locations_['pose']
        df_locations_ = torch.tensor(df_locations_, dtype = torch.float).cpu()
        print("test_min_result:"+str(test_min_result))
        print("test_min_R:"+str(test_min_R))
        print("test_min_t:"+str(test_min_t))
        print("df_locations[2][77]:"+str(df_locations_[521]))


        test_comp_pcs_copy = pc.copy()
        x_min, x_max, y_min, y_max = min(test_comp_pcs_copy[:,0]), max(test_comp_pcs_copy[:,0]), min(test_comp_pcs_copy[:,1]), max(test_comp_pcs_copy[:,1])
        x_interval = (x_max - x_min)/128
        y_interval = (y_max - y_min)/128
        x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
        y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
        im_target_pcl = np.zeros((128,128),dtype=np.float32)
        im_target_pcl = plot_image(test_comp_pcs_copy, im_target_pcl, x_range, y_range)
        im_target_pcl = Image.fromarray(im_target_pcl)
        if im_target_pcl.mode != 'RGB':
            im_target_pcl = im_target_pcl.convert('RGB')
        im_target_pcl = cv2.cvtColor(np.array(im_target_pcl), cv2.COLOR_RGB2GRAY)
        cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_source_pcl.jpg", im_target_pcl)
        
        test_comp_pcs_copy = test_comp_pcs[0].copy()
        x_min, x_max, y_min, y_max = min(test_comp_pcs_copy[:,0]), max(test_comp_pcs_copy[:,0]), min(test_comp_pcs_copy[:,1]), max(test_comp_pcs_copy[:,1])
        x_interval = (x_max - x_min)/128
        y_interval = (y_max - y_min)/128
        x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
        y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
        im_target_pcl = np.zeros((128,128),dtype=np.float32)
        im_target_pcl = plot_image(test_comp_pcs_copy, im_target_pcl, x_range, y_range)
        im_target_pcl = Image.fromarray(im_target_pcl)
        if im_target_pcl.mode != 'RGB':
            im_target_pcl = im_target_pcl.convert('RGB')
        im_target_pcl = cv2.cvtColor(np.array(im_target_pcl), cv2.COLOR_RGB2GRAY)
        cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_transformed_pcl.jpg", im_target_pcl)

        threshold.append(max_result)
        min_threshold.append(min_result)
        assert(0)




        thresholds.append(threshold)
        min_thresholds.append(min_threshold)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    min_thresholds = np.asarray(min_thresholds, dtype=np.float32)
    print("thresholds:"+str(thresholds.shape))
    print("np.min(values):"+str(np.min(thresholds)))
    sio.savemat("max_thresholds.mat",{'data':thresholds})
    sio.savemat("min_thresholds.mat",{'data':min_thresholds})
        
    # folder_sim = folder_similar_check(folder_, all_files, df_locations, top_k)
    #     sim_array.append(folder_sim)

    # sim_array = np.asarray(sim_array)
    # save_name = '/home/cc/Unsupervised-PointNetVlad_ongoing_developping/results/pcl_validate/validate.npy'
    # np.save(save_name, sim_array)
    print("Done")
            #pc_xyz = pcd2xyz(pc)
            # pc_xyz_copy = pc.copy()
            # x_min, x_max, y_min, y_max = min(pc_xyz_copy[:,0]), max(pc_xyz_copy[:,0]), min(pc_xyz_copy[:,1]), max(pc_xyz_copy[:,1])
            # x_interval = (x_max - x_min)/128
            # y_interval = (y_max - y_min)/128
            # x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
            # y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
            # im_source_pcl = np.zeros((128,128),dtype=np.float32)
            # im_source_pcl = plot_image(pc_xyz_copy, im_source_pcl, x_range, y_range)
            # im_source_pcl = Image.fromarray(im_source_pcl)
            # if im_source_pcl.mode != 'RGB':
            #     im_source_pcl = im_source_pcl.convert('RGB')
            # im_source_pcl = cv2.cvtColor(np.array(im_source_pcl), cv2.COLOR_RGB2GRAY)
            # cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_source_pcl.jpg", im_source_pcl)
            
            #comp_pc_xyz = pcd2xyz(comp_pc)
            # pc_pcd = o3d.read_point_cloud(os.path.join(pre_dir,folder,all_file))

            # # VOXEL_SIZE=0.001
            # # pc_pcd_raw = pc_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

            

            # # min_R = min_R_offset * min_R           
            # min_transform = np.zeros((4,4),dtype=np.float32)
            # min_transform[2,2] = 1
            # min_transform[3,3] = 1
            # min_transform[:2,:2] = min_R
            # min_transform[0,3] = min_t[0]
            # min_transform[1,3] = min_t[1]

            # comp_pc2_copy = comp_pcs[0].copy()
            # x_min, x_max, y_min, y_max = min(comp_pc2_copy[:,0]), max(comp_pc2_copy[:,0]), min(comp_pc2_copy[:,1]), max(comp_pc2_copy[:,1])
            # x_interval = (x_max - x_min)/128
            # y_interval = (y_max - y_min)/128
            # x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
            # y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
            # im_target_pcl = np.zeros((128,128),dtype=np.float32)
            # im_target_pcl = plot_image(comp_pc2_copy, im_target_pcl, x_range, y_range)
            # im_target_pcl = Image.fromarray(im_target_pcl)
            # if im_target_pcl.mode != 'RGB':
            #     im_target_pcl = im_target_pcl.convert('RGB')
            # im_target_pcl = cv2.cvtColor(np.array(im_target_pcl), cv2.COLOR_RGB2GRAY)
            # cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_target_pcl.jpg", im_target_pcl)

            
            # comp_pc2 = copy.deepcopy(pc_pcd).transform(min_transform)
            # comp_pc2 = pcd2xyz(comp_pc2)
            # comp_pc2_copy = comp_pc2.T.copy()
            # x_min, x_max, y_min, y_max = min(comp_pc2_copy[:,0]), max(comp_pc2_copy[:,0]), min(comp_pc2_copy[:,1]), max(comp_pc2_copy[:,1])
            # x_interval = (x_max - x_min)/128
            # y_interval = (y_max - y_min)/128
            # x_range = np.arange(x_min, x_max+0.5*x_interval, x_interval)
            # y_range = np.arange(y_min, y_max+0.5*y_interval, y_interval)
            # im_transform_pcl = np.zeros((128,128),dtype=np.float32)
            # im_transform_pcl = plot_image(comp_pc2_copy, im_transform_pcl, x_range, y_range)
            # im_transform_pcl = Image.fromarray(im_transform_pcl)
            # if im_transform_pcl.mode != 'RGB':
            #     im_transform_pcl = im_transform_pcl.convert('RGB')
            # im_transform_pcl = cv2.cvtColor(np.array(im_transform_pcl), cv2.COLOR_RGB2GRAY)
            # cv2.imwrite("/home/cc/netvlad_project/simulated_2D/Unsupervised-PointNetVlad_w_loop/models/im_transform_pcl.jpg", im_transform_pcl)
            # assert(0)



