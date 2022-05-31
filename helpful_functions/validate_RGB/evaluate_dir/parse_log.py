import os
import numpy as np
from scipy.io import savemat

path = "/home/cc/DeepMapping-plus/script/helpful_functions/validate_RGB/evaluate_dir/"
file_name_pre = "log"
file_name = file_name_pre+".txt"

myfile = open(os.path.join(path,file_name), "r")

accuracy_10 = []
accuracy_5 = []
accuracy_1 = []
variance_10 = []
variance_5 = []
variance_1 = []
scene = None
scene_target = "Goffs"
while myfile:
    line = myfile.readline()
    line = line.split("\n")[0]
    #indices_10_std
    
    if line.split("cene:")[0] == "S":
        scene = line.split("cene:")[1]
    elif (line.split("he Recall_10:")[0] == "T") and (scene == scene_target):
        accuracy_10.append(float(line.split("he Recall_10:")[1]))
    elif (line.split("he Recall_5:")[0] == "T") and (scene == scene_target):
        accuracy_5.append(float(line.split("he Recall_5:")[1]))
    elif (line.split("he Recall_1:")[0] == "T") and (scene == scene_target):
        accuracy_1.append(float(line.split("he Recall_1:")[1]))
    elif (line.split("ndices_10_std:")[0] == "i") and (scene == scene_target):
        variance_10.append(float(line.split("ndices_10_std:")[1]))
    elif (line.split("ndices_5_std:")[0] == "i") and (scene == scene_target):
        variance_5.append(float(line.split("ndices_5_std:")[1]))
    elif (line.split("ndices_1_std:")[0] == "i") and (scene == scene_target):
        variance_1.append(float(line.split("ndices_1_std:")[1]))
    if line == "":
        break

myfile.close()

index = list(np.arange(len(variance_10)))
print("len of index:"+str(len(index)))

mdict = {"accuracy_10":accuracy_10, "accuracy_5":accuracy_5, "accuracy_1":accuracy_1, "OD_10":variance_10, "OD_5":variance_5, "OD_1":variance_1, "index":index}
savemat(file_name_pre+scene_target+".mat", mdict)
