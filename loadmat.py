import scipy
import numpy as np
crop_size_h = 481
crop_size_w = 681
def myfunc(x):
    try:
        data_dic = scipy.io.loadmat(x)
        data_img = data_dic['img']
        #print "aaaaa"
        data_depth = data_dic['depth']
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        depth_mask[np.where(data_depth < 0.1)] = 0.0
        depth_mask[np.where(data_depth >= 0.1)] = 1.0
        data_norm = data_dic['norm']
        data_mask = data_dic['mask']
        grid = data_dic['grid']
    except:
        data_img = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        data_depth = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_norm = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        grid = np.zeros((crop_size_h, crop_size_w,3), dtype=np.float32)

    return data_img, data_depth, data_norm, data_mask,depth_mask,grid