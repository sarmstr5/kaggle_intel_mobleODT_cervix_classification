import numpy as np
import pandas as pd
import os, platform
import matplotlib.pyplot as plt
import cv2


# code gotten from https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster
def load_rgb_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2RGB)
    return img

def load_gry_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2GRAY)
    return img

def show_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_load_img(abspath_img))
    matplotlib.pyplot.show()


# Unify sidelong images into vertically long images
def orient_img(img):
    if img.shape[0] >= img.shape[1]:
        return img
    else:
        return np.rot90(img)



def resize_img_same_ratio(img):
    if img.shape[0] / 640.0 >= img.shape[1] / 480.0:
        # (640, *, 3)
        img_resized = cv2.resize(img, (int(640.0 * img.shape[1] / img.shape[0]), 640)) 
    else:
        # (*, 480, 3)
        img_resized = cv2.resize(img, (480, int(480.0 * img.shape[0] / img.shape[1]))) 
    return img_resized

def fill_img(img):
    if img.shape[0] == 640:
        int_resize_1 = img.shape[1]
        int_fill_1 = (480 - int_resize_1 ) // 2
        int_fill_2 =  480 - int_resize_1 - int_fill_1
        numpy_fill_1 = numpy.zeros((640, int_fill_1, 3),dtype=numpy.uint8)
        numpy_fill_2 = numpy.zeros((640, int_fill_2, 3), dtype=numpy.uint8)
        img_filled = numpy.concatenate((numpy_fill_1, img, numpy_fill_1), axis=1)

    elif img.shape[1] == 480:
        int_resize_0 = img.shape[0]
        int_fill_1 = (640 - int_resize_0 ) // 2
        int_fill_2 = 640 - int_resize_0 - int_fill_1
        numpy_fill_1 = numpy.zeros((int_fill_1, 480, 3), dtype=numpy.uint8)
        numpy_fill_2 = numpy.zeros((int_fill_2, 480, 3), dtype=numpy.uint8)
        img_filled = numpy.concatenate((numpy_fill_1, img, numpy_fill_1), axis=0)

    else:
        raise ValueError

    return img_filled


def process_img(img_fn, rgb=False):
    if rgb:
        img = load_rgb_img(img_fn)
    else:
        img = load_gry_img(img_fn)
    img = rotate_img(img)
    img = resize_img_same_ratio(img)
    img = fill_img(img)

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

    plt.imshow(sub_func_rotate_img_if_need(img))
    plt.show()

if __name__ == '__main__':
    main()
