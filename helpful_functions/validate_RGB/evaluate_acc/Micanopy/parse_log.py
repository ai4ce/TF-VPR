import os
import numpy as np
from scipy.io import savemat

path = "/home/cc/DeepMapping-plus/script/helpful_functions/validate_RGB/evaluate_acc"
file_name_pre = "log_PCL"
file_name = file_name_pre+".txt"

myfile = open(os.path.join(path,file_name), "r")
variance_10 = []
variance_5 = []
variance_1 = []
while myfile:
    line  = myfile.readline()
    #line = line.split("\n")[0]
    if line.split("ecall_10:")[0] == "The R":
        variance_10.append(float(line.split("ecall_10:")[1]))
    elif line.split("ecall_5:")[0] == "The R":
        variance_5.append(float(line.split("ecall_5:")[1]))
    elif line.split("ecall_1:")[0] == "The R":
        variance_1.append(float(line.split("ecall_1:")[1]))
    if line == "":
        break

myfile.close()

index = list(np.arange(len(variance_10)))
print("len of index:"+str(len(index)))

mdict = {"Recall_10":variance_10, "Recall_5":variance_5, "Recall_1":variance_1, "index":index}
savemat(file_name_pre+".mat", mdict)
