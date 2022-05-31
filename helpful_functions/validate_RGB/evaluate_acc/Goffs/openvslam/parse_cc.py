import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

def slam2world(t, r):
    return -np.matmul(r.transpose(), t)

start_flag = False
count = 0
matrix_count = 0
location_np = np.zeros((21378,2),dtype=np.float32)
matrix_44 = np.zeros((4,4),dtype=np.float32)

mode = "openvslam"
nn_ind = 0.2
r_mid = 0.2
r_ind = 0.6

if mode == "openvslam":
    pre_path = "/mnt/NAS/data/cc_data/2D_RGB_real_full_anbang"

data_folder = "/mnt/NAS/home/cc/data/habitat/Goffs/"
data_folder_json = "/mnt/NAS/home/cc/data/habitat/Goffs/Goffs.json"
f = open(data_folder_json)
data = json.load(f)
poses = []
count = 0

file_list = os.listdir(data_folder)
file_list.remove("trajectory.mp4")
file_list.remove("Goffs.json")

for j in range(len(file_list)):
    pose = data['pose_list'][j]
    poses.append(pose)
poses = np.array(poses)
pose_gt = np.zeros((poses.shape[0],2),dtype=np.float32)
pose_gt[:,0] = poses[:,0]
pose_gt[:,1] = poses[:,2]

tree = KDTree(pose_gt)
gt_neighbor = tree.query_radius(pose_gt, r=r_mid)

for line in open('cc.txt','r'):
    if line.split("\n")[0] == "[2022-04-25 22:42:57.495] [I] start global optimization module":
        start_flag = True
        continue
    if line[0] == '[':
        continue
    if line[0] == 'm':
        break
    if start_flag == True:
        if line.split("\n")[0] == str(count):
            count = count + 1
        else:
            test_list = line.split("\n")[0].strip().split(" ")
            while("" in test_list) :
                test_list.remove("")
            if matrix_count == 0:
                matrix_count += 1
                matrix_44[0,0] = float(test_list[0])
                matrix_44[0,1] = float(test_list[1])
                matrix_44[0,2] = float(test_list[2])
                matrix_44[0,3] = float(test_list[3])

                #assert(0)
            elif matrix_count == 1:
                matrix_count += 1
                matrix_44[1,0] = float(test_list[0])
                matrix_44[1,1] = float(test_list[1])
                matrix_44[1,2] = float(test_list[2])
                matrix_44[1,3] = float(test_list[3])
            elif matrix_count == 2:
                matrix_count += 1
                matrix_44[2,0] = float(test_list[0])
                matrix_44[2,1] = float(test_list[1])
                matrix_44[2,2] = float(test_list[2])
                matrix_44[2,3] = float(test_list[3])
            elif matrix_count == 3:
                matrix_count = 0
                matrix_44[3,0] = float(test_list[0])
                matrix_44[3,1] = float(test_list[1])
                matrix_44[3,2] = float(test_list[2])
                matrix_44[3,3] = float(test_list[3])
            
            rot = matrix_44[:3,:3]
            trans = matrix_44[0:3,3]
            pos = slam2world(trans, rot)
            location_np[count-1, 0] = pos[0]
            location_np[count-1, 1] = pos[1]

nbrs = NearestNeighbors(n_neighbors=31, algorithm='ball_tree').fit(location_np)
distance, location_neigh = nbrs.kneighbors(location_np)
k_nearest = 10
pos_index_range = list(range(-k_nearest//2, (k_nearest//2)+1))
location_neigh_temp = np.zeros((location_neigh.shape[0], 10), dtype=np.int32)

for i in range(location_neigh.shape[0]):
    location_neigh_each = list(location_neigh[i])
    for pos_index in pos_index_range:
        try:
            location_neigh_each.remove(pos_index+i)
        except:
            pass
    location_neigh_temp[i] = location_neigh_each[:10]

failure_count = 0
for i in range(21378):
    if abs(location_np[i][0]) < 0.000000000001:
        failure_count += 1
    else:
        compare_a = gt_neighbor[i]
        compare_b = location_neigh_temp[i]
        compare_a = set(compare_a)
        compare_b = set(compare_b)
        if len(list(compare_a.intersection(compare_b))) == 0:
            failure_count += 1
print("success_rate:"+str((21378-failure_count)/21378))
