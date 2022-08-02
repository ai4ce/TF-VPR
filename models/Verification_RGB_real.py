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
from shutil import copyfile

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

def threshold_cal_RGB(in_index, k_nearest, pre_path):
    fold_list = ["run_0", "run_1", "run_2", "run_3", "run_4", "run_5", "run_6"]
    neigh_range = list(range(-int(k_nearest//2),int(k_nearest//2)+1))
    max_compare_fitness = 0
    overall_min_compare_fitness = np.Inf
    for index in range(in_index):
        fold_index = int(index // 2137)
        file_index = int(index % 2137)
        in_image = cv2.imread(os.path.join(pre_path,fold_list[fold_index],"panoimg_"+str(file_index)+".png"))
        #print("in_image:"+str(in_image.shape))
        input_image = Image.fromarray(in_image)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        input_image_gray = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2GRAY)
        #print("input_image_gray:"+str(input_image_gray.shape))
        kp_left, des_left = SIFT(input_image_gray)

        for neigh in neigh_range:
            if (neigh + index >= 0) and (neigh + index <= in_index -1):
                fold_index = int((neigh + index) // 2137)
                file_index = int((neigh + index) % 2137)
                comp_image = cv2.imread(os.path.join(pre_path,fold_list[fold_index],"panoimg_"+str(file_index)+".png"))
                #print("comp_image:"+str(comp_image.shape))
                compare_image = Image.fromarray(comp_image)
                if compare_image.mode != 'RGB':
                    compare_image = compare_image.convert('RGB')
                compare_image_gray = cv2.cvtColor(np.array(compare_image), cv2.COLOR_RGB2GRAY)
                kp_right, des_right = SIFT(compare_image_gray)
                matches = matcher(kp_left, des_left, input_image, kp_right, des_right, compare_image, threshold=0.5)
                #print("matches:"+str(matches))
                #print("no of matches:"+str(len(matches)))
                #assert(0)
                if max_compare_fitness < len(matches):
                    max_compare_fitness = len(matches)
                if overall_min_compare_fitness > len(matches):
                    overall_min_compare_fitness = len(matches)
            if neigh % 20 == 0:
                print("max_compare_fitness:"+str(max_compare_fitness))
                print("overall_min_compare_fitness:"+str(overall_min_compare_fitness))
    return overall_min_compare_fitness, max_compare_fitness

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
            '''
            if current_err is None:
                current_err,R,t = None,None,None
            '''
        # print("reg_p2p.transformation:"+str(reg_p2p.transformation))
        if (current_err is not None) and (current_err < min_compare_fitness):
            min_compare_fitness = current_err
            min_R = R
            min_t = t

        if min_compare_fitness > max_compare_fitness:
            max_compare_fitness = min_compare_fitness
        if min_compare_fitness < overall_min_compare_fitness:
            overall_min_compare_fitness = min_compare_fitness

    return overall_min_compare_fitness * threshold_ratio, max_compare_fitness * threshold_ratio, min_transf, min_R, min_t

def Verify_image(query_index, pre_trusted_positive, pre_path):
    trusted_index = []
    fold_list = ["run_0"]#, "run_1", "run_2", "run_3", "run_4", "run_5", "run_6"]
    fold_index = 0
    file_index = int(query_index)
    in_image = cv2.imread(os.path.join(pre_path,fold_list[fold_index],"panoimg_"+str(file_index)+".jpg"))
    input_image = Image.fromarray(in_image)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    input_image_gray = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2GRAY)
    kp_left, des_left = SIFT(input_image_gray)
    for pre_trust in pre_trusted_positive:
        fold_index = 0
        file_index = int(pre_trust)
        comp_image = cv2.imread(os.path.join(pre_path,fold_list[fold_index],"panoimg_"+str(file_index)+".jpg"))
        compare_image = Image.fromarray(comp_image)
        if compare_image.mode != 'RGB':           
            compare_image = compare_image.convert('RGB')
        compare_image_gray = cv2.cvtColor(np.array(compare_image), cv2.COLOR_RGB2GRAY)
        kp_right, des_right = SIFT(compare_image_gray)
        matches = matcher(kp_left, des_left, input_image_gray, kp_right, des_right, compare_image_gray, threshold=0.5)
        if len(matches)>=15:
            trusted_index.append(pre_trust)
    return trusted_index

def similarity_filter(in_pcl, compare_pcl, pos_pcs, threshold_ratio, cal_thresholds=None):
    src = in_pcl.astype(np.float32)[:,:2]
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

        if current_err < min_fitness:
            min_fitness = current_err
            
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

def draw_plot(trusted_positive,input_path):
    fold_list = ["run_0", "run_1", "run_2", "run_3", "run_4", "run_5", "run_6"]
    output_path = os.path.join("results/images")
    trusted_index = trusted_positive[0]
    query_index = 0
    query_image_path = os.path.join(input_path,"run_0","panoimg_0.png")
    copyfile(query_image_path, os.path.join(output_path,"query_0.png"))
    for trust_ind in trusted_index:
        folder_ind = int(trust_ind//2137)
        file_ind = int(trust_ind%2137)
        trust_image_path = os.path.join(input_path,fold_list[folder_ind],"panoimg_"+str(file_ind)+".png")
        copyfile(trust_image_path, os.path.join(output_path,"evaluate_"+str(trust_ind)+".png"))

def Compute_positive(flag, db_vec, potential_positives, potential_distributions, trusted_positives, weight, sort_weight, indice, epoch):
    if flag:
        trusted_positive = []
        for index2 in range(db_vec.shape[0]*db_vec.shape[1]):
            index_range = list(range(-int(cfg.NEIGHBOR)//2, (int(cfg.NEIGHBOR)//2)+1))
            threshold = 1
            for ind in index_range:
                if (ind + index2 >= 0) and (ind + index2 < db_vec.shape[0]*db_vec.shape[1]):
                    if sort_weight[index2][ind + index2] < threshold:
                        threshold = sort_weight[index2][ind + index2]
            
            pre_trusted_positive = np.array(list(range(len(indice[index2]))))[np.array(sort_weight[index2])>=threshold]
            k_nearest = 10
            ind_range = neigh_range = np.arange((-k_nearest//2)+index2, ((k_nearest//2)+1)+index2)
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive,ind_range)
            if len(pre_trusted_positive) >= cfg.INIT_TRUST:
                pre_trusted_positive = np.array(indice[index2])[np.argsort(weight[index2])[::-1][:(cfg.INIT_TRUST+k_nearest)]]
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive,ind_range)
            
            folder_path = os.path.join(cfg.DATASET_FOLDER_RGB_REAL)
            trusted_pos = Verify_image(index2,pre_trusted_positive, cfg.DATASET_FOLDER_RGB_REAL)
            trusted_positive.append(trusted_pos)
        return potential_positives, potential_distributions, trusted_positive
    else:

        new_trusted_positive = []
        
        for index2 in range(db_vec.shape[0]*db_vec.shape[1]):
            folder_path = os.path.join(cfg.DATASET_FOLDER_RGB_REAL)
            
            trusted_positives = np.squeeze(trusted_positives)
            previous_trusted_positive = trusted_positives[index2]
            
            if ((np.array(previous_trusted_positive).ndim) == 2) and (np.array(previous_trusted_positive).shape[0]!=0):
                previous_trusted_positive = previous_trusted_positive[0]
            else:
                pass
            
            index_range = list(range(-int(cfg.NEIGHBOR)//2, (int(cfg.NEIGHBOR)//2)+1))
            threshold = 1
            for ind in index_range:
                if (ind + index2 >= 0) and (ind + index2 < db_vec.shape[0]*db_vec.shape[1]):
                    if sort_weight[index2][ind + index2] < threshold:
                        threshold = sort_weight[index2][ind + index2]
            
            pre_trusted_positive = np.array(list(range(len(indice[index2]))))[np.array(sort_weight[index2])>=threshold]
            k_nearest = 10
            ind_range = neigh_range = np.arange((-k_nearest//2)+index2, ((k_nearest//2)+1)+index2)
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive,ind_range)

            if len(pre_trusted_positive)>= cfg.INIT_TRUST:
                pre_trusted_positive = np.array(indice[index2])[np.argsort(weight[index2])[::-1][:(cfg.INIT_TRUST+k_nearest)]]
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive,ind_range)
            pre_trusted_positive = np.setdiff1d(pre_trusted_positive, previous_trusted_positive)
            
            filtered_trusted_positive = Verify_image(index2, pre_trusted_positive, cfg.DATASET_FOLDER_RGB_REAL)

            if len(filtered_trusted_positive) == 0:
                trusted_positive = previous_trusted_positive
            else:
                trusted_positive = list(previous_trusted_positive)
                trusted_positive.extend(list(filtered_trusted_positive))
                trusted_positive = np.array(list(set(trusted_positive)),dtype=np.int32)
            
            new_trusted_positive.append(trusted_positive)

        return [], [], new_trusted_positive

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
    runs_folder = "2D_real/"
    k_nearest = 6

    cc_dir = "/mnt/NAS/data/cc_data_2/"
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
    folder_size = len(all_files)
    data_index = list(range(folder_size))

    #GT 
    folder_ = os.path.join(pre_dir,folder)
    gt_mat = os.path.join(folder_, 'gt_pose.mat')
    df_locations = sio.loadmat(gt_mat)
    df_locations = df_locations['pose']
    df_locations = torch.tensor(df_locations, dtype = torch.float).cpu()

    for index, all_file in enumerate(all_files):

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
 
        threshold.append(max_result)
        min_threshold.append(min_result)

        thresholds.append(threshold)
        min_thresholds.append(min_threshold)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    min_thresholds = np.asarray(min_thresholds, dtype=np.float32)
    print("thresholds:"+str(thresholds.shape))
    print("np.min(values):"+str(np.min(thresholds)))
    sio.savemat("max_thresholds.mat",{'data':thresholds})
    sio.savemat("min_thresholds.mat",{'data':min_thresholds})
    
    print("Done")



