from __future__ import division

import six
import numpy as np
import pandas as pd
import cv2
import glob
import random

np.random.seed(2016)
random.seed(2016)

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,
AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

