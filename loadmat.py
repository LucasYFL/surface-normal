import scipy
import numpy as np
import os
crop_size_h = 481
crop_size_w = 681
def myfunc(x):
    try:
        data_dic = scipy.io.loadmat(x)
        data_img = data_dic['img']
        data_norm = data_dic['norm']
    except:
        return None
        data_img = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        data_norm = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)


    return data_img, data_norm
for x in os.listdir("./large"):
    print(myfunc("./large/"+x))