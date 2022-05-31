import os
import scipy.io as sio
import numpy as np
import glob, os

mode = "2"

if mode == "1":
    pre_path = "/mnt/NAS/home/cc/Our_method_RGB_real/Our_method_RGB_w_temporal/results/Goffs"
elif mode == "2":
    pre_path = "/mnt/NAS/home/cc/Our_method_RGB_real/Our_method_RGB_w_temporal2/results/Goffs"

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
