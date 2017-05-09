from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import processing_images

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K

# coding: utf-8

# Public Leader-board of 0.89094
# ====================================================
# derived code from
# https://www.kaggle.com/the1owl/intel-mobileodt-cervical-cancer-screening/artificial-intelligence-for-cc-screening/comments
# Save train and test images to normalized numpy arrays once for running multiple neural network configuration tests


def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count()-1)
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2(path):
    #img = cv2.imread(path)
    img = processing_images.process_img(path, rgb=True)
    #resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR) #use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, img]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

def process_data():
    train = glob.glob('/data/kaggle/train/**/*.jpg') + glob.glob('/data/kaggle/additional/**/*.jpg')
    train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = ['type','image','path'])[::5] #limit for Kaggle Demo
    train = im_stats(train)
    train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
    train_data = normalize_image_features(train['path'])
    np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)
    print(le.classes_) #in case not 1 to 3 order
    np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)

    test = glob.glob('../input/test/*.jpg')
    test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path']) #[::20] #limit for Kaggle Demo
    test_data = normalize_image_features(test['path'])
    np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)

    test_id = test.image.values
    np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)

# Start your neural network high performance engines

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32, 32))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

def main():
    process_run = False
    model_run = True
    print(cpu_count())
    np.random.seed(17)

    if process_run:
        process_data()
    #    K.set_image_dim_ordering('th')

    if model_run:
        # read in data
        train_data = np.load('train.npy')
        train_target = np.load('train_target.npy')

        # split data
        x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

        # Image preprocessing, rotating images and performing random zooms
        datagen = ImageDataGenerator(rotation_range=0.9, zoom_range=0.3)
        datagen.fit(train_data)

        # Create Image model
        K.set_floatx('float32')
        model = create_model()
        model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=200, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))

        # Load processed data
        test_data = np.load('test.npy')
        test_id = np.load('test_id.npy')

        # run classification
        pred = model.predict_proba(test_data)
        df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
        df['image_name'] = test_id
        df.to_csv('~/kaggle_intel_mobileODT_cervix_classification/submission.csv', index=False)

if __name__=='__main__':
    main()



