import imutils
import numpy as np
import os
import cv2
from imutils.paths import list_images
import argparse
import pickle
from tqdm import tqdm
import test as ap
from flask import Flask, render_template, url_for, request, redirect, Response



def describe(image):
    hist = cv2.calcHist([image], [0, 1, 2],
                        None, [8,8,8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()
path ="dataset"

index = {}
for imagePath in tqdm(os.listdir(path)):
    k = imagePath[0:imagePath.rfind("."):]
    image = cv2.imread(os.path.join(path, imagePath))
    features = describe(image)
    index[k] = features
    f = open("index.pickle", "wb")
    f.write(pickle.dumps(index))
    f.close()
    print("[INFO] done...indexed {} images".format(len(index)))
