import os
import glob
path="dataset"
i=1
for img in glob.glob(path + "/*.jpg"):
    dest=str(i)+".jpg"
    os.rename(img,dest)
    i=i+1
