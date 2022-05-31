import os
import numpy as np
from scipy.io import savemat

path = "/home/cc/DeepMapping-plus/script/helpful_functions/validate_RGB_real/Scene1/evaluate_dir/"
file_name_pre = "log_PCL"
file_name = file_name_pre+".txt"

myfile = open(os.path.join(path,file_name), "r")
variance_10 = []
variance_5 = []
variance_1 = []
recall_10 = []
recall_5 = []
recall_1 = []
while myfile:
    line  = myfile.readline()
    #indices_10_std
    if line.split("he Recall_10: ")[0] == "T":
        recall_10.append(float(line.split("he Recall_10: ")[1]))
    elif line.split("he Recall_5: ")[0] == "T":
        recall_5.append(float(line.split("he Recall_5: ")[1]))
    elif line.split("he Recall_1: ")[0] == "T":
        recall_1.append(float(line.split("he Recall_1: ")[1]))

    elif line.split("ndices_10_std:")[0] == "i":
        variance_10.append(float(line.split("ndices_10_std:")[1]))
    elif line.split("ndices_5_std:")[0] == "i":
        variance_5.append(float(line.split("ndices_5_std:")[1]))
    elif line.split("ndices_1_std:")[0] == "i":
        variance_1.append(float(line.split("ndices_1_std:")[1]))
    if line == "":
        break

myfile.close()

index = list(np.arange(len(variance_10)))
print("len of index:"+str(len(index)))

mdict = {"OD_10":variance_10, "OD_5":variance_5, "OD_1":variance_1, "recall_10":recall_10, "recall_5":recall_5, "recall_1":recall_1, "index":index}
savemat(file_name_pre+".mat", mdict)
