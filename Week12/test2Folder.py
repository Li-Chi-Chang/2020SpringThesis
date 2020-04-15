import os,shutil

# this py file is trying to make a 20-record dataset

DatasetDir = "/home/chen/LiChiChang/2020SpringThesis/dataset/mnist/mnist" #pls indicate your path
test2Dir = "/home/chen/LiChiChang/2020SpringThesis/dataset/mnist/test2"

if os.path.exists(test2Dir):
    shutil.rmtree(test2Dir)

os.mkdir(test2Dir)

for i in range(30):
    modI = i % 10
    qI = i / 10
    shutil.copy(os.path.join(os.path.join(DatasetDir,str(modI)),str(i)+".jpg"),os.path.join(test2Dir,str(i)+".jpg"))
    