# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt
from skimage import transform, io, data, img_as_float, color, exposure
import os.path
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import time
from sklearn import svm
import base64
from flask import Flask
from flask import request
def ImageDivision(img_hsv, h, w):
    # h = heigh of the image, w = width of the image
    n = 3      # Division 3*3
    ratio = 1/n
    i = 0
    j = 0
    Image_sub = []
    for i in range(n):
        for j in range(n):
            x = int(i * ratio * h)
            y = int(j * ratio * w)
            img = img_hsv[x:int(x + ratio * h)+1, y:int(y + ratio * w)+1, :]
            Image_sub.append(img)
    return Image_sub
def ImageRead(filename):
    Img_ori = cv2.imread(filename, cv2.IMREAD_COLOR)
    ImageSize = Img_ori.shape
    ImageHeight = ImageSize[0]
    ImageWidth = ImageSize[1]
    return Img_ori,ImageHeight, ImageWidth


def ImageHSV(Image_sub):
    N = 81 #num of features
    features = []
    for i in range(N):
        features.append(0)
    features = np.array(features)
    i = 0
    for i in range(Image_sub.shape[0]): # 3*3 sub-Image
        for h in range(Image_sub[i].shape[0]): # height of ith sub-Image
            for w in range(Image_sub[i].shape[1]): # width of ith sub-Image
                ############## H division 0 < H < 180##################
                if Image_sub[i][h][w][0] in range(60):
                    features[9 * i] += 1
                elif Image_sub[i][h][w][0] in range(60,120):
                    features[9 * i + 1] += 1
                elif Image_sub[i][h][w][0] in range(120,180):
                    features[9 * i + 2] += 1

                ############## S division 0 < S < 255##################
                if Image_sub[i][h][w][1] in range(85):
                    features[9 * i + 3] += 1
                elif Image_sub[i][h][w][1] in range(85,170):
                    features[9 * i + 4] += 1
                elif Image_sub[i][h][w][1] in range(170,256):
                    features[9 * i + 5] += 1

                ############## V division 0 < V < 255##################
                if Image_sub[i][h][w][2] in range(85):
                    features[9 * i + 6] += 1
                elif Image_sub[i][h][w][2] in range(85,170):
                    features[9 * i + 7] += 1
                elif Image_sub[i][h][w][2] in range(170,256):
                    features[9 * i + 8] += 1
    return features

def Predi_Data_process(filename):
    predi = [[]*81]*1
    Image_ori, ImageHeight, ImageWidth = ImageRead(filename)
    ###########################RGB 2 HSV####################################
    Image_hsv = cv2.cvtColor(Image_ori, cv2.COLOR_BGR2HSV)
    #########################3 * 3 Division################################
    Image_sub = ImageDivision(Image_hsv, ImageHeight, ImageWidth)
    Image_sub = np.array(Image_sub)
    #########################81 features##################################
    predi[0] = ImageHSV(Image_sub)
    return predi
def Read_Charc():
    charc = [[0 for col in range(81)] for row in range(100)]
    with open('list.txt') as f:
        m = 0
        for line in f:
            n = 80
            tra = line
            i = len(tra) - 3
            while i >= 0:
                sum = 0
                j = 0
                while tra[i] != '*' and i >= 0:
                    sum += ((int(tra[i])) * (10 ** j))
                    j += 1
                    i -= 1
                i -= 1
                charc[m][n] = sum
                n -= 1
            m += 1
    return charc

app = Flask(__name__)
@app.route('/') # just for test
def api_root():
    return 'This is server side, plz upload data to api /isrealface?img="base64code". api will return 1 for realFace and 0 for feak.'
@app.route('/isrealface', methods=['POST']) #this main apt,input:base64 encoded img data, output:1 for real & 0 for feak
def api_isRealFace():
    charc = Read_Charc()  # 从文件读取之前获取的所有样本的特征特征
    label = [] * 100
    for i in range(50):
        label.append(-1)
    for i in range(50, 100):
        label.append(1)
    charc = [[0 for col in range(81)] for row in range(100)]
    predi = [[] * 81] * 1
    if request.form.get('imgEncodeByBase64'):
        # print(request.form.get('imgEncodeByBase64'))
        imgEncodeByJPEG = base64.b64decode(request.form.get('imgEncodeByBase64'))
        file = open('predict.jpg', 'wb')
        file.write(imgEncodeByJPEG)
        file.close()
        # 开始训练数据
        clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        x = np.array(charc)
        y = np.array(label)
        clf.fit(x, y)
        #  预测结果
        img = Image.open('predict.jpg')
        img.thumbnail((60, 60))
        img.save('predict.jpg')
        predi = Predi_Data_process('predict.jpg')
        z = np.array(predi)
        res = clf.predict(z)
        return str(res)   # 返回给前端预测的结果
    else:
        return "-2"       #如何没有接受的base64编码或编码错误，则返回错误值
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)