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
    #mode = "baseline"
    #mode = "Supervised_NetVlad"
    #mode = "Our_method_RGB"
    #mode = "Our_method_RGB_w_temporal"
    mode = "Our_method_RGB_ongoing2"
    #mode = "Our_method_RGB_w_temporal3"
    #mode = "Method_2_RGB_no_feat_w_temporal"
   # mode = "Method_2_RI_RGB_no_feat__w_temporal"

    if mode == "baseline":
        new_dir = "/home/cc/baseline_RGB/results/Nimmons"
    elif mode == "Supervised_NetVlad":
        new_dir = "/home/cc/Supervised_NetVlad/results/Nimmons"
    elif mode == "Our_method_RGB":
        new_dir = "/home/cc/Our_method_RGB/results/Micanopy"
    elif mode == "Our_method_RGB_w_temporal":
        new_dir = "/home/cc/Our_method_RGB_w_temporal/results/"
    elif mode == "Our_method_RGB_w_temporal2":
        new_dir = "/home/cc/Our_method_RGB_w_temporal2/results/"
    elif mode == "Our_method_RGB_w_temporal3":
        new_dir = "/home/cc/Our_method_RGB_w_temporal3/results/"
    elif mode == "Our_method_RGB_ongoing2":
        new_dir = "/home/cc/Our_method_RGB_ongoing2/results/"
    elif mode == "Method_2_RGB_no_feat_w_temporal":
        new_dir = "/home/cc/Method_2_RGB_no_feat_w_temporal/results/Micanopy"
    elif mode == "Method_2_RI_RGB_no_feat__w_temporal":
        new_dir = "/home/cc/Method_2_RI_RGB_no_feat__w_temporal/results/Micanopy"
    save_name = os.path.join(new_dir,"Goffs",'database'+str(temp_index)+'.npy')
    best_matrix = np.load(save_name)
    best_matrix = torch.tensor(best_matrix, dtype = torch.float64)
    best_matrix = np.array(best_matrix)
    data_dir = '/mnt/NAS/home/yiming/habitat_3/train/Goffs/'
    all_folders = sorted(os.listdir(data_dir))

    folders = []
    # All runs are used for training (both full and partial)
    index_list = [0,1,2,3,4,5,6]
    print("Number of runs: "+str(len(index_list)))
    for index in index_list:
        print("all_folders[index]:"+str(all_folders[index]))
        folders.append(all_folders[index])
    print(folders)
    folder_sizes = []
    #all_folder_sizes = []

    for folder_ in folders:
        all_files = list(sorted(os.listdir(os.path.join(data_dir,folder_))))
        all_files.remove("gt_pose.mat")
        folder_sizes.append(len(all_files))
    
    '''
    for folder_ in all_folders:
        all_files = list(sorted(os.listdir(os.path.join(data_dir,folder_,"jpg_rgb"))))
        all_folder_sizes.append(len(all_files))
    '''
    all_folders = folders
    indices_10 = []
    indices_25 = []
    indices_100 = []
    best_matrix_list = []
    
    #print("all_folder_sizes:"+str(all_folder_sizes))
    print("len(index_list):"+str(len(index_list)))
    #############
    '''
    for j, index in enumerate(index_list):
        folder = all_folders[j]
        if j == 0:
            overhead = 0
        else:
            overhead = 0
            for i in range(j):
                overhead = overhead + folder_sizes[i]
        #print("best_matrix:"+str(best_matrix.shape))
        #best_matrix_sec = best_matrix[overhead:overhead+folder_sizes[j]]
        #best_matrix_sec = best_matrix[j]
        '''
    best_matrix_sec = best_matrix.reshape(best_matrix.shape[0]* best_matrix.shape[1], best_matrix.shape[2])
    #print("best_matrix_sec:"+str(best_matrix_sec.shape))
    #print("best_matrix_sec[0]:"+str(best_matrix_sec[0]))

    total_len = best_matrix_sec.shape[0]
    #nbrs_1 = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(best_matrix_sec)
    #nbrs_5 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(best_matrix_sec)

    nbrs_11 = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(best_matrix_sec)
        
    #distance_1, indices_1 = nbrs_1.kneighbors(best_matrix_sec)
    #distance_5, indices_5 = nbrs_5.kneighbors(best_matrix_sec)
    distance_11, indices_11 = nbrs_11.kneighbors(best_matrix_sec)
    distance_10, indices_10 = distance_11[:,1:11], indices_11[:,1:11]
    distance_5, indices_5 = distance_11[:,1:6], indices_11[:,1:6]
    distance_1, indices_1 = distance_11[:,1:2], indices_11[:,1:2]
    #print("distance_5:"+str(distance_5[0]))
    #print("distance_1:"+str(distance_1.shape))
    #assert(0)
    
    '''
        indices_10.append(indice_10)
        indices_25.append(indice_25)
        indices_100.append(indice_100)
    indices_10 = np.array(indices_10, dtype=np.int64)
    indices_25 = np.array(indices_25, dtype=np.int64)
    indices_100 = np.array(indices_100, dtype=np.int64)
    '''
    #############
    init_pose = None

    indices_gt = []
    #indices_gt_true = []

    location_ests = []
    ori_ests = []
    for index,folder in enumerate(all_folders):
        data_dir_f = os.path.join(data_dir, folder) 
        if not os.path.exists(data_dir_f):
            os.makedirs(data_dir_f)
        checkpoint_dir_validate_f = os.path.join(checkpoint_dir_validate, folder)
        if not os.path.exists(checkpoint_dir_validate_f):
            os.makedirs(checkpoint_dir_validate_f)
        gt_file = os.path.join(data_dir_f,'gt_pose.mat')
        gt_pose = sio.loadmat(gt_file)
        gt_pose = gt_pose['pose']
        gt_location = gt_pose[:,0:3:2]
        orientation = gt_pose[:,-4:]
        #pose_est = torch.tensor(gt_pose, dtype = torch.float).cpu()
        location_est = torch.tensor(gt_location, dtype = torch.float).cpu()
        location_ests.append(gt_location)
        ori_ests.append(orientation)
    location_ests = torch.tensor(location_ests, dtype = torch.float).cpu()
    ori_ests = torch.tensor(ori_ests, dtype=torch.float).cpu()
    location_ests = location_ests.reshape(location_ests.shape[0]*location_ests.shape[1],location_ests.shape[2])
    ori_ests = ori_ests.reshape(ori_ests.shape[0]*ori_ests.shape[1], ori_ests.shape[2])
    euler_ests = np.zeros((ori_ests.shape[0], 3), dtype=np.float32)
    
    rolls = []
    for i in range(ori_ests.shape[0]):
        w, x, y, z = ori_ests[i][0], ori_ests[i][1], ori_ests[i][2], ori_ests[i][3]
        roll = math.atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z);
        rolls.append(roll)
        #pitch = math.atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z);
        #yaw = math.asin(2*x*y + 2*z*w);
        #print("roll:"+str(roll))
        #assert(pitch==0)
        #assert(yaw==0)
        #print("pitch:"+str(pitch))
        #print("yaw:"+str(yaw))
        #print("quaternion_to_euler_angle:"+str(quaternion_to_euler_angle(w,x,y,z)))


    #nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(location_ests)
    #nbrs_16 = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(location_est)
    tree = KDTree(location_ests)
    indice_ = tree.query_radius(location_ests, r=0.5)
    #distance_, indice_ = nbrs.kneighbors(location_ests)
    #distance_16, indice_16 = nbrs_16.kneighbors(location_est)
    #indices_gt = np.array(indice_, dtype=np.int64)
    indices_gt = indice_
    #assert(0)
    #indice_16 = np.array(indice_16, dtype=np.int64)

    #indices_gt_true.extend(indice_16)
    #utils.draw_graphs(location_est, indices[index], 1, checkpoint_dir_validate_f, downsample_size = 32)
    '''
    print("best_matrix_sec[14957]:"+str(best_matrix_sec[14957]))
    print("2135:"+str(best_matrix_sec[2135]))
    print("426:"+str(best_matrix_sec[426]))
    print("425:"+str(best_matrix_sec[425]))
    print("839:"+str(best_matrix_sec[839]))
    print("227:"+str(best_matrix_sec[227]))
    print("1394:"+str(best_matrix_sec[1394]))
    print("1395:"+str(best_matrix_sec[1395]))
    print("#############################################")
    # eval:  8866 13247 11752 13661 13660  8865 13246 14956
    print("8866:"+str(best_matrix_sec[8866]))
    print("13247:"+str(best_matrix_sec[13247]))
    print("11752:"+str(best_matrix_sec[11752]))
    print("13661:"+str(best_matrix_sec[13661]))
    print("13660:"+str(best_matrix_sec[13660]))
    print("8865:"+str(best_matrix_sec[8865]))
    print("13246:"+str(best_matrix_sec[13246]))
    assert(0)
    '''
    #indice_gt_temp = np.array(indices_gt, dtype=np.int64) 
    indice_gt_temp = indices_gt.copy()
    #indice_gt_temp = indice_gt_temp.reshape(indice_gt_temp.shape[0]* indice_gt_temp.shape[1],indice_gt_temp.shape[2])
    indices_gt = []
    #indices_gt = np.zeros((indice_gt_temp.shape[0],10),dtype=np.int32)
    #not_indice_gt = np.zeros((indices_gt.shape[0],total_len-11),dtype=np.int32)
    
    for i in range(len(indice_gt_temp)):
        indice_gt_without_self = list(indice_gt_temp[i]).copy()
        try:
            indice_gt_without_self.remove(i)
        except:
            pass
            #indice_gt_without_self.remove(indice_gt_without_self[-1])    
        #indices_gt[i] = np.array(indice_gt_without_self, dtype=np.float32)
        print("len(indice_gt_without_self)!=0:"+str(len(indice_gt_without_self)!=0))
        assert(len(indice_gt_without_self)!=0)
        indices_gt.append(indice_gt_without_self)
        #a = set(range(total_len))
        #b = set(indice_gt_temp[i])
        #not_indice_gt[i] = np.array(list(a.difference(b)))
    #indices_gt_true = np.array(indices_gt_true, dtype=np.int64)
    #print("indices_gt:"+str(indices_gt.shape))
    #print("indices_gt_true:"+str(indices_gt_true.shape))
    #assert(0)
    
    #### Accuracy ######
    '''
    indices_10_std = np.zeros((len(indices_gt)),dtype=np.float32)
    indices_5_std = np.zeros((len(indices_gt)),dtype=np.float32)
    indices_1_std = np.zeros((len(indices_gt)),dtype=np.float32)
    '''
    indices_10_std_per = []
    indices_5_std_per = []
    indices_1_std_per = []
    indices_gt_std_per = []

    for i in range(len(indices_gt)):
        indice_gt_per = list(indices_gt[i])
        indice_10_per = list(indices_10[i])
        indice_5_per = list(indices_5[i])
        indice_1_per = list(indices_1[i])
        print("indice_gt_per:"+str(indice_gt_per))
        gt_set = set(indice_gt_per)
        set_10 = set(indice_10_per)
        set_5 = set(indice_5_per)
        set_1 = set(indice_1_per)
        correct_10 = set_10.intersection(gt_set)
        correct_5 = set_5.intersection(gt_set)
        correct_1 = set_1.intersection(gt_set)
        roll_i = rolls[i]
        
        indice_gt = [0,0,0,0]
        for gt in gt_set:
            roll_gt = rolls[gt]
            roll_diff = roll_gt - roll_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) < 3.1415926/4:
                indice_gt[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                indice_gt[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                indice_gt[2] += 1
            else:
                indice_gt[3] += 1
        print("indice_gt:"+str(indice_gt))
        print("gt_set:"+str(gt_set))
        count = 0
        for ind_gt in indice_gt:
            if ind_gt > 0:
                count += 1
        indices_gt_std_per.append(count)
        print("gt_count:"+str(count))

        indice_10 = [0,0,0,0]
        for c_10 in correct_10:
            roll_c_10 = rolls[c_10]
            roll_diff = roll_c_10-roll_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) > 3.1415926/4:
                indice_10[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                indice_10[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                indice_10[2] += 1
            else:
                indice_10[3] +=1
        print("indice_10:"+str(indice_10))
        count = 0
        for ind_10 in indice_10:
            if ind_10 > 0:
                count += 1
        indices_10_std_per.append(count)
        print("10_count:"+str(count))

        #print("np.std(indices_10_std_per):"+str(np.std(indices_10_std_per)))
        '''
        if len(indices_10_std_per) == 0:
            indices_10_std[i] = 0
        else:
            indices_10_std[i] = np.std(indices_10_std_per)
        '''
        indice_5 =  [0,0,0,0]
        for c_5 in correct_5:
            roll_c_5 = rolls[c_5]
            roll_diff = roll_c_5-roll_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            #if abs(roll_diff) > 1.5707963:
            if abs(roll_diff) > 3.1415926/4:           
                indice_5[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                indice_5[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                indice_5[2] += 1
            else:
                indice_5[3] += 1
        print("indice_5:"+str(indice_5))
        count = 0
        for ind_5 in indice_5:
            if ind_5 > 0:
                count += 1
        indices_5_std_per.append(count)
        print("5_count:"+str(count))

        #indices_5_std.append(indices_5_std_per)
        '''
        if len(indices_5_std_per) == 0:
            indices_5_std[i] = 0
        else:
            indices_5_std[i] = np.std(indices_5_std_per)
        '''
        indice_1 = [0,0,0,0]
        for c_1 in correct_1:
            roll_c_1 = rolls[c_1]
            roll_diff = roll_c_1-roll_i
            while roll_diff > 3.1415926:
                roll_diff -= 3.1415926
            while roll_diff < -3.1415926:
                roll_diff += 3.1415926
            if abs(roll_diff) > 3.1415926/4:
                indice_1[0] += 1
            elif (abs(roll_diff) >= 3.1415926/4) and (abs(roll_diff) < 3.1415926/2):
                indice_1[1] += 1
            elif (abs(roll_diff) <= 3.1415926/2) and (abs(roll_diff) > 3 * 3.1415926/4):
                indice_1[2] += 1
            else:
                indice_1[3] += 1
        print("indices_1:"+str(indice_1))
        count = 0
        for ind_1 in indice_1:
            if ind_1 > 0:
                count += 1
        indices_1_std_per.append(count)
        print("1_count:"+str(count))

        #indices_1_std.append(indices_1_std_per)
        '''
        if len(indices_1_std_per) == 0:
            indices_1_std[i] =0
        else:
            indices_1_std[i] = np.std(indices_1_std_per)
        '''
    #assert(0)

    #print("gt_set:"+str(gt_set))
    #print("10_set:"+str(set_10))
    #print("diff:"+str(set_10.intersection(gt_set)))
    
    # print("indices_10_std:"+str(indices_10_std))
    indices_gt_std_per = np.array(indices_gt_std_per)
    indices_10_std_per = np.array(indices_10_std_per)
    indices_5_std_per = np.array(indices_5_std_per)
    indices_1_std_per = np.array(indices_1_std_per)

    print("indices_10_std:"+str(np.mean(indices_10_std_per/indices_gt_std_per)))   
    print("indices_5_std:"+str(np.mean(indices_5_std_per/indices_gt_std_per)))   
    print("indices_1_std:"+str(np.mean(indices_1_std_per/indices_gt_std_per)))   
    '''
    B_10, P_10 = indices_10.shape
    B_25, P_25 = indices_25.shape
    B_100, P_100 = indices_100.shape
    B_gt, P_gt = indices_gt.shape
    NB_gt, NP_gt = not_indice_gt.shape

    tp_count_10 = 0.0
    tp_count_25 = 0.0
    tp_count_100 = 0.0
    
    tn_count_10 = 0.0
    tn_count_25 = 0.0
    tn_count_100 = 0.0

    fp_count_10 = 0.0
    fp_count_25 = 0.0
    fp_count_100 = 0.0

    fn_count_10 = 0.0
    fn_count_25 = 0.0
    fn_count_100 = 0.0
    
    for b in range(B_10):
        for p in range(P_10):
            if indices_10[b,p] in indices_gt[b]:
                tp_count_10 = tp_count_10 + 1
            else:
                fp_count_10 = fp_count_10 + 1
    for b in range(B_25):
        for p in range(P_25):
            if indices_25[b,p] in indices_gt[b]:
                tp_count_25 = tp_count_25 + 1
            else:
                fp_count_25 = fp_count_25 + 1
    
    for b in range(B_100):
        for p in range(P_100):
            if indices_100[b,p] in indices_gt[b]:
                tp_count_100 = tp_count_100 + 1
            else:
                fp_count_100 = fp_count_100 + 1
    for b in range(B_gt):
        for p in range(P_gt):
            if (indices_gt[b,p] not in indices_10[b]):
                fn_count_10 = fn_count_10 + 1
            if (indices_gt[b,p] not in indices_25[b]):
                fn_count_25 = fn_count_25 + 1
            if (indices_gt[b,p] not in indices_100[b]):
                fn_count_100 = fn_count_100 + 1
            
    for nb in range(NB_gt):
        not_indice_10 = set(range(total_len)).difference(set(indices_10[nb]))
        not_indice_25 = set(range(total_len)).difference(set(indices_25[nb]))
        not_indice_100 = set(range(total_len)).difference(set(indices_100[nb]))
        tn_count_10_temp = len(set(not_indice_gt[nb]).intersection(not_indice_10))
        tn_count_25_temp = len(set(not_indice_gt[nb]).intersection(not_indice_25))
        tn_count_100_temp = len(set(not_indice_gt[nb]).intersection(not_indice_100))
        tn_count_10 += tn_count_10_temp
        tn_count_25 += tn_count_25_temp
        tn_count_100 += tn_count_100_temp
    '''
    '''
            if (b==0) and (indices[b,n] not in indices_gt_true[b]):
                absolute_false_positive.append(indices[b,n])
            if (b==0) and (indices[b,n] not in indices_gt[b]):
                false_positive.append(indices[b,n])
            if (b==0) and (indices_gt_true[b,n] not in indices[b]):
                absolute_false_negative.append(indices_gt_true[b,n])
            if (b==0) and (indices_gt[b,n] not in indices[b]):
                false_negative.append(indices_gt[b,n])
            
    print("trained nearest 16 neigbours for packet 0:"+str(indices[0]))
    print("top 16 nearest neighbours: :"+str(indices_gt_true[0]))
    print("top nearest 48 neigbours:"+str(indices_gt[0]))
    print("false_positives for packet 0:"+str(false_positive))
    print("false_positives with loose constraint for packet 0:"+str(absolute_false_positive))
    print("false_negatives for packet 0:"+str(false_negative))
    print("false_negatives with loose constraint for packet 0:"+str(absolute_false_negative))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train"))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt"))

    if not os.path.exists(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true")):
        os.makedirs(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true"))

    for index_ in indices_gt_true[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt_true", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()

    for index_ in indices_gt[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_gt", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()

    for index_ in indices[0,0]:
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        #print("index_:"+str('{0:04}'.format(index_)))
        pcl = read_point_cloud("/home/cc/dm_data/v0_pose05/00000"+str('{0:04}'.format(index_))+".pcd")
        pcl = np.asarray(pcl.points, dtype=np.float32)
        ax.scatter(pcl[:, 0], pcl[:, 1], zorder=2)
        #print("saving figure to "+str(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train", 'unsupervised_'+str(index_)+'.png')))
        plt.savefig(os.path.join(checkpoint_dir_validate, "pcl_map_Similarity_train", 'unsupervised_'+str(index_)+'.png'), bbox_inches='tight')
        plt.close()
    
    '''
    '''
    print("################################################")
            #print("indices[b,p,n]:"+str(indices[b,p,n]))
            #print("indices_gt[b,p]:"+str(indices_gt[b,p]))
    print("The Recall_10: "+str(acc_count_10/(fn_count_10+acc_count_10)))
    print("The Recall_25: "+str(acc_count_25/(fn_count_25+acc_count_25)))
    print("The Recall_100: "+str(acc_count_100/(fn_count_100+acc_count_100)))
    '''
    '''
    print("################################################")
    print("The TPR_10: "+str(tp_count_10/(fn_count_10+tp_count_10)))
    print("The TPR_25: "+str(tp_count_25/(fn_count_25+tp_count_25)))
    print("The TPR_100: "+str(tp_count_100/(fn_count_100+tp_count_100)))
    print("################################################")
    print("The FPR_10: "+str(fp_count_10/(fp_count_10+tn_count_10)))
    print("The FPR_25: "+str(fp_count_25/(fp_count_25+tn_count_25)))
    print("The FPR_100: "+str(fp_count_100/(fp_count_100+tn_count_100)))
    print("################################################")
    save_name = os.path.join(checkpoint_dir,'best_n_trained.npy')

    np.save(save_name,indices_100[0])
    '''
    print("Done")


if __name__ == "__main__":
    for i in range(100):
        validate(i)
