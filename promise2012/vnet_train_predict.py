from promise2012.Vnet2d.vnet_model import Vnet2dModule
import numpy as np
import pandas as pd
import cv2
from promise2012.Vnet2d.util import convertMetaModelToPbModel


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('promise2012vnet2dMask.csv')
    csvimagedata = pd.read_csv('promise2012vnet2dimage.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "model\\Vnet2dModule.pd", "log\\", 0.001, 0.5, 2, 4)


def predict():
    image = cv2.imread("1.bmp", 0)
    unet2d = Vnet2dModule(512, 512, 1, model_path="model\\Vnet2dModule.pd")
    predictvalue = unet2d.prediction(image)
    cv2.imwrite("mask.bmp", predictvalue)


def meta2pd():
    convertMetaModelToPbModel(meta_model="model\\Vnet2dModule.pd", pb_model="model")


# train()
# predict()
meta2pd()
