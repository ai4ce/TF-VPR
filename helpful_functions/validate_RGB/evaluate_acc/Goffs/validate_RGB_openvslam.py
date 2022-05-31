import scipy.io as sio
import numpy as np
import glob, os
import json
import msgpack
import collections
import natsort

from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R

def slam2world(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix().transpose(), t)

mode = "openvslam"
nn_ind = 0.2
r_mid = 0.2
r_ind = 0.6

if mode == "1":
    pre_path = "/mnt/NAS/home/cc/Our_method_RGB/Our_method_RGB_w_temporal/results/Goffs"
elif mode == "distance":
    pre_path = "/mnt/NAS/home/cc/Our_method_RGB/Our_method_RGB_w_distance_check/results/Goffs"
elif mode == "openvslam":
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
print("gt_neighbor:"+str(gt_neighbor))
assert(0)

with open(os.path.join(pre_path,"CC.msg"), "rb") as data_file:
    byte_data = data_file.read()
    data_loaded = msgpack.unpackb(byte_data, use_list=False, raw=False)

data = data_loaded
landmarks = data['landmarks']
keyframes = collections.OrderedDict(natsort.natsorted(data['keyframes'].items()))

point3d=[]
key=[]

print("landmarks.items():"+str(len(landmarks.items())))
print("keyframes.items():"+str(len(keyframes.items())))

for id, point in landmarks.items():
    pos = point["pos_w"]
    point3d.append([pos[0], 0, pos[2]])

for id, point in keyframes.items():
    trans = point["trans_cw"]
    rot = point['rot_cw']
    pos = slam2world(trans, rot)
    print("pos:"+str(pos))
    key.append([pos[0], pos[2]])
    
assert(0)

recalls_1 = []
recalls_5 = []
recalls_10 = []

print("all_txt:"+str(len(glob.glob(os.path.join(pre_path,"*.txt")))))
file_len = len(glob.glob(os.path.join(pre_path,"*.txt")))

for i in range(file_len):
    file_name = "results_" + str(i) + ".txt"
    with open(os.path.join(pre_path,file_name)) as f:
        lines = f.readlines()
        for index,line in enumerate(lines):
            if line.split(":")[0] == "Average Recall @1":
                recall_1 = float(lines[index+1])
                recalls_1.append(recall_1)
            if line.split(":")[0] == "Average Recall @5":
                recall_5 = float(lines[index+1])
                recalls_5.append(recall_5)
            if line.split(":")[0] == "Average Recall @10":
                recall_10 = float(lines[index+1])
                recalls_10.append(recall_10)

sio.savemat('log_'+str(mode)+'.mat',{'recall_1':np.array(recalls_1), 'recall_5':np.array(recalls_5), 'recall_10':np.array(recalls_10)})
