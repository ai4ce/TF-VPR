import os
import numpy as np
from scipy.io import savemat

path = "/home/cc/DeepMapping-plus/script/helpful_functions/validate_RGB/evaluate_acc"
file_name_pre = "log_distance"
file_name = file_name_pre+".txt"

myfile = open(os.path.join(path,file_name), "r")
variance_10 = []
variance_5 = []
variance_1 = []
while myfile:
    line  = myfile.readline()
    if line.split("ecall@10:")[0] == "R":
        variance_10.append(float(line.split("ecall@10:")[1]))
    elif line.split("ecall@5:")[0] == "R":
        variance_5.append(float(line.split("ecall@5:")[1]))
    elif line.split("ecall@1:")[0] == "R":
        variance_1.append(float(line.split("ecall@1:")[1]))
    if line == "":
        break

myfile.close()

index = list(np.arange(len(variance_10)))
print("len of index:"+str(len(index)))

mdict = {"Recall_10":variance_10, "Recall_5":variance_5, "Recall_1":variance_1, "index":index}
savemat(file_name_pre+".mat", mdict)
