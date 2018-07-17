from Vnet2d.vnet_model import Vnet2dModule
import numpy as np
import pandas as pd
import cv2


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('PROMISE2012Mask.csv')
    csvimagedata = pd.read_csv('PROMISE2012Image.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "model\\Vnet2dModule.pd", "log\\", 0.001, 0.5, 100000, 4)


def predict():
	image = cv2.imread("1.bmp",0)
    unet2d = Vnet2dModule(512, 512, 1)
    predictvalue = unet2d.prediction("model\\Vnet2dModule.pd", image)
	cv2.imwrite("mask.bmp", predictvalue)


train()
# predict()
