import scipy.io as sio
import os
import numpy as np

data_path = "/data2/cc_data/AVD/ActiveVisionDataset/"

gt_file = "image_structs.mat"
folders = os.listdir(data_path)

for folder in folders:
    mat_contents = sio.loadmat(os.path.join(data_path, folder, gt_file))
    mat_contents = mat_contents['image_structs']
    world_pos = mat_contents['world_pos'][0]
    direction = mat_contents['direction'][0]
    gt = np.zeros((len(world_pos), 6), dtype = np.float32)
    for i in range(len(world_pos)):
        gt[i,0] = world_pos[i][0]
        gt[i,1] = world_pos[i][1]
        gt[i,2] = world_pos[i][2]
        gt[i,3] = direction[i][0]
        gt[i,4] = direction[i][1]
        gt[i,5] = direction[i][2]
    gt_dict = {"pose":gt}
    sio.savemat(os.path.join(data_path, folder, "gt_pose.mat"), gt_dict)

    #gt_dir = np.array(direction, dtype = np.float32)
print("folders:"+str(folders))
