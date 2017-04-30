import platform
import os
import pandas as pd




def get_file_paths():
    if 'c001' in platform.node(): 
        # platform.node() => 'c001' or like 'c001-n030' on Colfax
        abspath_dataset_dir_train_1 = '/data/kaggle/train/Type_1'
        abspath_dataset_dir_train_2 = '/data/kaggle/train/Type_2'
        abspath_dataset_dir_train_3 = '/data/kaggle/train/Type_3'
        abspath_dataset_dir_test    = '/data/kaggle/test/'
        abspath_dataset_dir_add_1   = '/data/kaggle/additional/Type_1'
        abspath_dataset_dir_add_2   = '/data/kaggle/additional/Type_2'
        abspath_dataset_dir_add_3   = '/data/kaggle/additional/Type_3'
    elif '.local' in platform.node():
        # platform.node() => '*.local' on my local MacBook Air
        abspath_dataset_dir_train_1 = '/abspath/to/train/Type_1'
        abspath_dataset_dir_train_2 = '/abspath/to/train/Type_2'
        abspath_dataset_dir_train_3 = '/abspath/to/train/Type_3'
        abspath_dataset_dir_test    = '/abspath/to/test/'
        abspath_dataset_dir_add_1   = '/abspath/to/additional/Type_1'
        abspath_dataset_dir_add_2   = '/abspath/to/additional/Type_2'
        abspath_dataset_dir_add_3   = '/abspath/to/additional/Type_3'
    else:
        # For kaggle's kernels environment (docker container?)
        abspath_dataset_dir_train_1 = '/kaggle/input/train/Type_1'
        abspath_dataset_dir_train_2 = '/kaggle/input/train/Type_2'
        abspath_dataset_dir_train_3 = '/kaggle/input/train/Type_3'
        abspath_dataset_dir_test    = '/kaggle/input/test/'
        abspath_dataset_dir_add_1   = '/kaggle/input/additional/Type_1'
        abspath_dataset_dir_add_2   = '/kaggle/input/additional/Type_2'
        abspath_dataset_dir_add_3   = '/kaggle/input/additional/Type_3'

    return abspath_dataset_dir_train_1, abspath_dataset_dir_train_2, abspath_dataset_dir_train_3, abspath_dataset_dir_test, abspath_dataset_dir_add_1, abspath_dataset_dir_add_2, abspath_dataset_dir_add_3


def get_list_abspath_img(abspath_dataset_dir):
    list_abspath_img = []
    for str_name_file_or_dir in os.listdir(abspath_dataset_dir):
        if ('.jpg' in str_name_file_or_dir) == True:
            list_abspath_img.append(os.path.join(abspath_dataset_dir, str_name_file_or_dir))
    list_abspath_img.sort()
    return list_abspath_img

def get_fn_df():

    abspath_dataset_dir_train_1, abspath_dataset_dir_train_2,
    abspath_dataset_dir_train_3, abspath_dataset_dir_test,
    abspath_dataset_dir_add_1, abspath_dataset_dir_add_2,
    abspath_dataset_dir_add_3 = get_file_paths()
    
    list_abspath_img_train_1 = get_list_abspath_img(abspath_dataset_dir_train_1)
    list_abspath_img_train_2 = get_list_abspath_img(abspath_dataset_dir_train_2)
    list_abspath_img_train_3 = get_list_abspath_img(abspath_dataset_dir_train_3)
    list_abspath_img_train   = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3

    list_abspath_img_test    = get_list_abspath_img(abspath_dataset_dir_test)

    list_abspath_img_add_1   = get_list_abspath_img(abspath_dataset_dir_add_1)
    list_abspath_img_add_2   = get_list_abspath_img(abspath_dataset_dir_add_2)
    list_abspath_img_add_3   = get_list_abspath_img(abspath_dataset_dir_add_3)
    list_abspath_img_add     = list_abspath_img_add_1   + list_abspath_img_add_2   + list_abspath_img_add_3

    # 0: Type_1, 1: Type_2, 2: Type_3
    list_answer_train        = [0] * len(list_abspath_img_train_1) + [1] * len(list_abspath_img_train_2) + [2] * len(list_abspath_img_train_3)
    list_answer_add          = [0] * len(list_abspath_img_add_1) + [1] * len(list_abspath_img_add_2) + [2] * len(list_abspath_img_add_3)

#   print(list_abspath_img_train_1[0:2])

    pandas_columns = ['Number of image files']
    pandas_index = ['train_1', 'train_2','train_3', 'train', 'test', 'add_1',
                    'add_2', 'add_3', 'add', 'train + add', 'total']
    pandas_data = [len(list_abspath_img_train_1), len(list_abspath_img_train_2),
                    len(list_abspath_img_train_3), len(list_abspath_img_train),
                    len(list_abspath_img_test), len(list_abspath_img_add_1),
                    len(list_abspath_img_add_2), len(list_abspath_img_add_3),
                    len(list_abspath_img_add), len(list_abspath_img_train) +
                    len(list_abspath_img_add), len(list_abspath_img_train) +
                    len(list_abspath_img_test) + len(list_abspath_img_add)]
    # counting number of image files
    return pd.DataFrame(pandas_data, index=pandas_index, columns=pandas_columns)

def main():

    df = get_fn_df()
    # show type ratios
    pandas_columns = ['Type_1', 'Type_2', 'Type_3']
    pandas_index   = ['train', 'test', 'add']

    ratio_train    = [x / len(list_abspath_img_train) for x in
                      [len(list_abspath_img_train_1),
                       len(list_abspath_img_train_2),
                       len(list_abspath_img_train_3)]]
    ratio_test     = ['?', '?', '?']
    ratio_add      = [x / len(list_abspath_img_add) for x in
                      [len(list_abspath_img_add_1), len(list_abspath_img_add_2),
                       len(list_abspath_img_add_3)]]

    pandas_data    = [ratio_train, ratio_test, ratio_add]

    df2 = pd.DataFrame(pandas_data, index = pandas_index, columns = pandas_columns)
    print(df2)

if __name__ == '__main__':
    main()
