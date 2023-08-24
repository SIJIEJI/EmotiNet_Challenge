

# search the best threshold


import torch
from torchvision import transforms
import numpy as np
#from PIL import Image
from ResNet import resnet18, resnet50, resnet101
import collections
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import os, sys, shutil
import random as rd

from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_imgs(image_list_file):
    #import pdb; pdb.set_trace()
    imgs = list()
    with open(image_list_file, 'r') as imf:
        for line in imf:
            # pdb.set_trace()
            line = line.strip()
            line = line.split()
            img_name = line[0]
            label_arr = line[1:]
            img_path = '/home/jisijie/Code/img_dataset/' + img_name
            # path, label, dentity_level = line.split(' ',2)
            # label = int(label)
            imgs.append((img_path,label_arr))
            #print(imgs)  #imgs here actually is label of imgs
    return imgs

def load_models():
    model = resnet18(end2end=True)
    pretrained_net_dict = torch.load('model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        crop_size = 110
        crop_height = crop_width = crop_size
        crop_center_y_offset = 15
        crop_center_x_offset = 0
        if self.phase == 'train':
            scale_aug = 0.02
            trans_aug = 0.01
        else:
            scale_aug = 0.0
            trans_aug = 0.0
        
        # computed parameters
        randint = rd.randint
        scale_height_diff = (randint(0,1000)/500-1)*scale_aug
        crop_height_aug = crop_height*(1+scale_height_diff)
        scale_width_diff = (randint(0,1000)/500-1)*scale_aug
        crop_width_aug = crop_width*(1+scale_width_diff)


        trans_diff_x = (randint(0,1000)/500-1)*trans_aug
        trans_diff_y = (randint(0,1000)/500-1)*trans_aug


        center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
                 (img.height/2 + crop_center_y_offset)*(1+trans_diff_y))

        
        if center[0] < crop_width_aug/2:
            crop_width_aug = center[0]*2-0.5
        if center[1] < crop_height_aug/2:
            crop_height_aug = center[1]*2-0.5
        if (center[0]+crop_width_aug/2) >= img.width:
            crop_width_aug = (img.width-center[0])*2-0.5
        if (center[1]+crop_height_aug/2) >= img.height:
            crop_height_aug = (img.height-center[1])*2-0.5

        crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
                    center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

        mid_img = img.crop(crop_box)
        res_img = img.resize( (final_width, final_height) )
        #import pdb; pdb.set_trace()
        return res_img

# def sigmoid(x):
#     return 1/(1+np.exp(-x))


def main_caculate():
    imgs_train = load_imgs('label.txt')
    Train_gt = []
    #import pdb; pdb.set_trace()
    total_cnt_train = len(imgs_train)
    for j in range(total_cnt_train):
              
         gt_train = imgs_train[j][1]
         Train_gt.append(gt_train)

    Cnt1 = []
    Cnt0 = []
    Cnt999 = []
    for k in range(23):
        #import pdb; pdb.set_trace()
        col = [x[k] for x in Train_gt]
        cnt1 = col.count('1')
        cnt0 = col.count('0')
        cnt999 = col.count('999')
        Cnt1.append(cnt1)
        Cnt0.append(cnt0)
        Cnt999.append(cnt999)
    
    import pdb; pdb.set_trace()
    c = np.array(Cnt1)
    d = np.array(Cnt0)
    fratest = c / d

    testcnt1 = [152, 53, 208, 100, 479, 44, 1244, 546, 317, 26, 453, 10, 768, 963, 214, 97, 78, 294, 302, 122, 55, 124, 104]
    a = np.array(testcnt1)
    testcnt0 =  [1773, 1876, 1776, 1834, 1447, 1904, 730, 927, 1645, 1859, 1509, 1979, 1187, 916, 1755, 1868, 1806, 1707, 1699, 1879, 1946, 1877, 1897]
    b = np.array(testcnt0)
    testcnt999 =  [76, 72, 17, 67, 75, 53, 27, 528, 39, 116, 39, 12, 46, 122, 32, 36, 117, 0, 0, 0, 0, 0, 0]    
    fra10 = a / b
    final = (fra10+fratest)/2
    tensor_final = torch.FloatTensor(final)
    print(tensor_final)


def main():

    weight = [0.0758, 0.0296, 0.1270, 0.0462, 0.2840, 0.0224, 1.7504, 0.6051, 0.2095, 0.0184, 0.2404, 0.0053, 0.7263, 1.0102, 0.1073, 0.0527, 0.0357, 0.1925,  
        0.1933, 0.1004, 0.0563, 0.0661, 0.0622]
    Nor_weight = [float(i)/sum(weight) for i in weight]
    Nor_weight = np.array(Nor_weight)
    #import pdb; pdb.set_trace()
    Final = Nor_weight * 0.1
    Final = Final.tolist()
    print(Final)
# def mainmmm():

#     #print(model.eval())
#     for i in range(total_cnt):


#         path = imgs[i][0]
#         img = Image.open(path).convert("RGB")
#         caffe_crop = CaffeCrop('test')
#         transform_sj= transforms.Compose([caffe_crop,transforms.ToTensor()])
#         img_tensor = transform_sj(img)
#         img_tensor = torch.unsqueeze(img_tensor, 0)
#         #import pdb; pdb.set_trace()
#         preds = model(img_tensor)
#         prediction = torch.sigmoid(preds)

#         # best_threshold_over = []
#         # for i in range(len(prediction)):
#         #     avgF_optimal_thresholds = []
#         #     prediction_per = prediction[i]
#         #     for j in range(prediction_per.shape[1]):
                
#         gt = imgs[i][1]
#         #import pdb; pdb.set_trace()
#         Whole_pre.append(prediction)
#         Whole_gt.append(gt)

#     # for j in range(total_cnt_train):
              
#     #     gt_train = imgs[j][1]
#     #     import pdb; pdb.set_trace()
#     #     Train_gt.append(gt_train)

    
#     # cnt0 = Trian_gt.count(0)
#     # cnt999 = Trian_gt.count(999)
#     print ("cnt1: ", Cnt1)
#     print ("cnt0: ", Cnt0)
#     print ("cnt999: ", Cnt999)
    
#         # Fx = prediction
#         # Fy = gt
#         # #compute Avg F score
#         # TP = ((Fx == 1) & (Fy == 1)).sum()
#         #     # TN    predict 和 label 同时为
#         # TN = ((Fx == 0) & (Fy == 0)).sum()
#         # # FN    predict 0 label 1
#         # FN = ((Fx == 0) & (Fy == 1)).sum()
#         # # FP    predict 1 label 0
#         # FP = ((Fx == 1) & (Fy == 0)).sum()
#         # Fpre  = TP.float() / (TP.float() + FP.float())
#         # Frec = TP.float() / (TP.float() + FN.float())
#         # F05 = (1+0.25) * Fpre * Frec / (0.25*Fpre + Frec)
#         # F1 = (1+1) * Fpre * Frec / (1*Fpre + Frec)
#         # F2 = (1+4) * Fpre * Frec / (4*Fpre + Frec)
#         # AvgF = ( F05 + F1 + F2 ) / 3
#         # #prediction[prediction<=0.5] =0
#         # #prediction[prediction>0.5] =1


# for k in range(len(prediction)):
#                     f1_optimal_thresholds = []
#                     merged_preds_per_model = merged_preds[i]
#                     for j in range(merged_preds_per_model.shape[1]):
#                         precision, recall, thresholds = precision_recall_curve(labels[i][:, j].astype(np.int),merged_preds[i][:, j])
#                         f1_optimal_thresholds.append(thresholds[np.abs(precision-recall).argmin(0)])
#                     f1_optimal_thresholds = np.array(f1_optimal_thresholds)


    #print(Whole_gt.shape())
if __name__ == '__main__':
    main_caculate()
