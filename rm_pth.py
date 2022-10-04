import os
import shutil
filename = os.path.join("work_dirs_HuBMAP_mulit_5fold")
for i in os.listdir(filename):
    if os.path.isdir(os.path.join(filename, i)):
    #print(i)
        for j in os.listdir(os.path.join(filename, i)):
            if j == "model_predict_color":
                print(os.path.join(filename, i, j))
                shutil.rmtree(os.path.join(filename, i, j))
            if j[-3:] == "pth" and j != "latest.pth" and int(j.split("_")[1][:-4]) !=30000 and int(j.split("_")[1][:-4]) !=45000:
                print(os.path.join(filename, i, j))
                os.remove(os.path.join(filename, i, j))