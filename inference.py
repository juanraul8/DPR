'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *
from utils_shtools import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

# ---------------- Create dense SH coefficients ---------------
coeffs = shtools_dense()

# ---------------- Create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))

#---------------------------Create Pretrained model----------------------------------

modelFolder = 'trained_model/'
#device = torch.device('cpu')
device = torch.device('cuda:0')

# load model
from defineHourglass_512_gray_skip import *
my_network = HourglassNet().to(device)
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
my_network.train(False)

#---------------------------Processing Input/Ouput------------------------------------

saveFolder = 'result'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

img = cv2.imread('data/FelixMeissen.jpg')
row, col, _ = img.shape
img = cv2.resize(img, (512, 512))
Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

inputL = Lab[:,:,0]
inputL = inputL.astype(np.float32)/255.0
inputL = inputL.transpose((0,1))
inputL = inputL[None,None,...]
inputL = Variable(torch.from_numpy(inputL).to(device))

#-------------------------Juan Code------------------------------------------------
for i in range(len(coeffs)):

    #Load SH coeffients
    sh = coeffs[i]
    sh = sh[0:9]
    sh = sh * 0.7

    #-----------------------rendering half-sphere
    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    #Gray Scale to RGBA Format
    shading_img = np.stack((shading, shading, shading), 2)

    cv2.imwrite(os.path.join(saveFolder, 'lights/light_{:02d}.png'.format(i)), shading_img)

    #--------------------------Rendering images using the network

    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).to(device))

    outputImg, outputSH = my_network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)

    # save image
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))
    cv2.imwrite(os.path.join(saveFolder, 'imgs/img_{:02d}.jpg'.format(i)), resultLab)
