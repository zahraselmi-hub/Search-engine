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
from io import BytesIO
import re, time, base64

on_line = Flask(__name__)
@on_line.route('/')
def interface():
    return render_template("interface.html")

@on_line.route('/receiver', methods=['POST'])
def worker():
    print('Incoming..')
    #get the img from the js
    jsdata = request.form['javascript_data']
    jada = json.loads(jsdata)[0]
    image = cv2.imread(convertToIMG(jada['img']))
    result = search(image)
    print(result)
    return str(result), 200
# convert the img base64 to jpg img
def convertToIMG(data):
    base64_data = re.sub('^data:image/.+;base64,', '', data)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    img.save(str(t) + '.jpg', "JPEG")
    return str(t) + '.jpg'

def search(queryFeatures):
    results = {}
    for (k, features) in index.items():
        print(k)
        print(features)
        d = chi2_distance(features, queryFeatures)
        results[k] = d
    print("res")
    print(results)
    results = sorted([(v, k) for (k, v) in results.items()])
    return results


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    return d


def describe(image):
    hist = cv2.calcHist([image], [0, 1, 2],
                        None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


x = open("index.pickle", "rb")
index = pickle.load(x)
print(index)
queryImage = cv2.imread("1.jpg")
width = 400
height = 166
dim = (width, height)
queryImage = cv2.resize(queryImage, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Query", queryImage)
montageA = np.zeros((166 * 5, 400, 3), dtype="uint8")
montageB = np.zeros((166 * 5, 400, 3), dtype="uint8")
cv2.waitKey(0)
results = search(describe(queryImage))
print(search(describe(queryImage)))
path = "dataset"
for j in range(0, 10):
    (score, imageName) = results[j]
    path1 = os.path.join(path, imageName)
    result = cv2.imread(path1 + ".jpg")
    width = 400
    height = 166
    dim = (width, height)
    result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)
    print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
    if j < 5:
        montageA[j * 166:(j + 1) * 166, :] = result
    else:
        montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

cv2.imshow("Results 1-5", montageA)
cv2.waitKey(0)

if __name__ == "__main__":
    on_line.run(port=4555, debug=True)
