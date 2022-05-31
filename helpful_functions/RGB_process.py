import os
import scipy.io as sio
import numpy as np
import math
from shutil import copyfile
import json

basedir = '/mnt/NAS/home/yiming'
#gt_poses = []
indir = os.path.join(basedir,"habitat/Goffs")
f=open(os.path.join(indir,"Goffs.json"))
data = json.load(f)
outdir = os.path.join(basedir,"habitat_3/train/Goffs")

file_size = 2137
file_num = 7

for i in range(7):
    gt_poses = []
    for j in range(file_size):
        in_file_index = i* file_size + j
        out_file_index = j
        in_file = os.path.join(indir,"panoimg_"+str(in_file_index)+".png")
        out_file = os.path.join(outdir,"run_"+str(i),"panoimg_"+str(out_file_index)+".png")
        copyfile(in_file, out_file)
        gt_poses.append(data['pose_list'][in_file_index])
        #print("count:"+str(data['pose_list'][in_file_index]))
        #assert(0)
    sio.savemat(os.path.join(outdir,"run_"+str(i),'gt_pose.mat'), {'pose':gt_poses})
        
#gt_poses = np.array(gt_poses)
#sio.savemat('gt_pose.mat', {'pose':gt_poses})
