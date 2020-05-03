import cmath as cm

import cv2

import csv

from datetime import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Masking
from keras.callbacks import Callback

import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from numpy import matlib
from numpy import interp
import numpy.ma as ma

import os
import os.path
#import oct2py
os.environ['OCTAVE_EXECUTABLE']='C:\\Octave\\Octave-4.2.1\\bin\\octave-cli.exe'
from oct2py import Oct2Py, octave
octave.addpath('C:\\Octave\\Octave-4.2.1\\share\\octave\\4.2.1\\m')

from pandas import read_csv

from scipy.interpolate import interp1d
from scipy.ndimage import convolve
#from scipy import stats

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

from urllib import request

#(469,40)

def lookBack(x,y,n):
    if n>0:
        xAux = []
        yAux = []
        for i in range(n,x.shape[0]+1):
            xAux.append(x[i-n:i,:])
            yAux.append(y[i-1])
    else:
        xAux = x
        yAux = y
    
    return xAux, yAux
    
def preparingData():
    P = np.asarray(np.loadtxt('./partial_output_files/P_PY.csv'))
    T = np.asarray(np.loadtxt('./partial_output_files/T_PY.csv'))
    P = np.transpose(P)
    T = T.reshape(-1,1)
    scaler = StandardScaler()

    pn = scaler.fit_transform(P)
    tn = scaler.fit_transform(T)

    seriesSize = len(tn)
    
    #print('\nArquivos salvos em '+path+' :\n')
    np.savetxt('./partial_output_files/pn_PY.csv', pn)
    #print('pn_PY.csv\n')

    np.savetxt('./partial_output_files/tn_PY.csv', tn)
    #print('tn_PY.csv\n')
    
    mask_value = 2
    tn[np.where(np.isnan(tn))] = mask_value
    
    return p, t