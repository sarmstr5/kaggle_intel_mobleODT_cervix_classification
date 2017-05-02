import numpy as np
import pandas as pd
import os, platform
import matplotlib.pyplot as plt
import cv2

# derived code from:
# https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster
def load_rgb_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2RGB)
    return img

def load_gry_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2GRAY)
    return img

def show_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_load_img(abspath_img))
    matplotlib.pyplot.show()

# Orient images to be portriate
def orient_img(img):
    if img.shape[0] >= img.shape[1]:
        return img
    else:
        return np.rot90(img)

# make all images same size
def resize_img_same_ratio(img):
    if img.shape[0] / 640.0 >= img.shape[1] / 480.0:
        # (640, *, 3)
        img_resized = cv2.resize(img, (int(640.0 * img.shape[1] / img.shape[0]), 640)) 
    else:
        # (*, 480, 3)
        img_resized = cv2.resize(img, (480, int(480.0 * img.shape[0] / img.shape[1]))) 
    return img_resized

# fill in blank space with black
def fill_img(img):
    if img.shape[0] == 640:
        int_resize_1 = img.shape[1]
        int_fill_1 = (480 - int_resize_1 ) // 2
        int_fill_2 =  480 - int_resize_1 - int_fill_1
        numpy_fill_1 = np.zeros((640, int_fill_1, 3),dtype=np.uint8)
        numpy_fill_2 = np.zeros((640, int_fill_2, 3), dtype=np.uint8)
        img_filled = np.concatenate((numpy_fill_1, img, numpy_fill_1), axis=1)

    elif img.shape[1] == 480:
        int_resize_0 = img.shape[0]
        int_fill_1 = (640 - int_resize_0 ) // 2
        int_fill_2 = 640 - int_resize_0 - int_fill_1
        numpy_fill_1 = np.zeros((int_fill_1, 480, 3), dtype=np.uint8)
        numpy_fill_2 = np.zeros((int_fill_2, 480, 3), dtype=np.uint8)
        img_filled = np.concatenate((numpy_fill_1, img, numpy_fill_1), axis=0)

    else:
        raise ValueError

    return img_filled

def process_img(img_fn, rgb=False):
    if rgb:
        img = load_rgb_img(img_fn)
    else:
        img = load_gry_img(img_fn)
    img = orient_img(img)
    img = resize_img_same_ratio(img)
    img = fill_img(img)
    return img

def save_files(org_fn, img):
    fn = '~/kaggle_intel_mobileODT_cervix_classification/data'
    fn_segments = org_fn.split('/')
    # test files dont have a type dir
    if fn_segments[-2] == 'test':
        for seg in fn_segments[-2:]:
            fn = os.path.join(fn, seg)
    else:
        for seg in fn_segments[-3:]:
            fn = os.path.join(fn, seg)
    print(fn)
    cv2.imwrite(fn, img)

def main():
    print(os.listdir())
    if 'c001' in platform.node():
        print('in if')
        abspath_img = '/data/kaggle/test/5.jpg'
    print(os.path.isfile(abspath_img))
    print(cv2.imread(abspath_img))


    img = load_rgb_img(abspath_img)
    plt.imshow(img)
    plt.show()

    plt.imshow(sub_func_orient_img_if_need(img))
    plt.show()

if __name__ == '__main__':
    main()
