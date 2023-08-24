

# search the best threshold


import torch
from torchvision import transforms
import numpy as np
#from PIL import Image
#from ResNet import resnet18, resnet50, resnet101
#from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import collections
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import os, sys, shutil
import random as rd
#from AUrecognitionCP import image2AUvect

from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from model_irse_224 import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
            img_path = '/home/jisijie/Code/img_dataset_align' + img_name
            # path, label, dentity_level = line.split(' ',2)
            # label = int(label)
            imgs.append((img_path,label_arr))
            #print(imgs)  #imgs here actually is label of imgs
    return imgs

# for 112
# def load_models():
#     model = IR_50([112,112])
    
#     pretrained_net_dict = torch.load('model_best.pth.tar')
#     new_state_dict = OrderedDict()
    
#     for k, v in pretrained_net_dict['state_dict'].items():
        
#         name = k[7:] # remove `module.`
#         #print(name)
#         new_state_dict[name] = v
#     #import pdb; pdb.set_trace()
#     #model.load_state_dict(new_state_dict)
#     model.load_state_dict(new_state_dict, strict = False)
#     model.eval()
#     return model

#for 224

def load_models():
    model = IR_50([224,224])
    pretrained_net_dict = torch.load('affectnet_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict, strict = True)
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
        final_size = 112
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

RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)

def sigmoid(x):
    return 1/(1+np.exp(-x))



def main():
    imgs= load_imgs('label_test_align.txt')
    #imgs_train = load_imgs('label.txt')
    Whole_pre = []
    Whole_gt  = []
    #Train_gt = []
    #import pdb; pdb.set_trace()
    total_cnt = len(imgs)
    #total_cnt_train = len(imgs_train)
    Whole_acc = []
    Whole_F1 = []
    Whole_avg = []
    model = load_models()
    model.eval()
    
    #print(model.eval())
    for i in range(total_cnt):

        #import pdb; pdb.set_trace()
        path = imgs[i][0]
        
        target_arr = np.array(imgs[i][1],dtype='int32').T
        target_tensor = torch.tensor(target_arr)
        target = target_tensor
        
        #gt = np.array(imgs[i][1]).astype(np.int)
        #img = Image.open(path).convert("RGB")
        #import pdb; pdb.set_trace()
        pred_score = image2AUvect(path,model)
        # = pred_score.detach().numpy()
        
        Whole_pre.append(pred_score)
        target= target.gt(0)
        target= target.float()
        correct_num = sum(pred_score == target).sum().item()
        mean_acc = correct_num/23
        #import pdb; pdb.set_trace()
        #accuracy_score(target, pred_score)
        Whole_acc.append(mean_acc)
        
        Fx = pred_score
        Fy = target
        #compute Avg F score
        TP = ((Fx == 1) & (Fy == 1)).sum()
            # TN    predict 和 label 同时为
        TN = ((Fx == 0) & (Fy == 0)).sum()
        # FN    predict 0 label 1
        FN = ((Fx == 0) & (Fy == 1)).sum()
        # FP    predict 1 label 0
        FP = ((Fx == 1) & (Fy == 0)).sum()
        Fpre  = TP.float() / (TP.float() + FP.float())
        Frec = TP.float() / (TP.float() + FN.float())
        F1 = (1+1) * Fpre * Frec / (1*Fpre + Frec)
        #Avg = ( mean_acc + F1.item() ) / 2
        Whole_F1.append(F1.item())
        #print(i)
        
        #import pdb; pdb.set_trace()
        #Whole_avg.append(Avg)
    #import pdb; pdb.set_trace()
    FFFFF1 = np.array(Whole_F1)
    where_are_nan = np.isnan(FFFFF1)
    FFFFF1[where_are_nan] = 0
    #FFFFF1 = [x for x in FFFFF1 if str(x) != 'nan'] 
    FFF1 = np.mean(FFFFF1)
    ACCC = np.mean(Whole_acc)
    AVG = (FFF1 + ACCC) / 2
    print('ACC',ACCC)
    print('F1',FFF1)
    print('AVG',AVG)

    
    # 'F1\t''AVG\t'.format(,Whole_F1,Whole_avg)
    
         # preds = preds.detach().numpy().flatten()
        # gt = imgs[i][1]
        #import pdb; pdb.set_trace()
        
        # Whole_pre.append(preds)
        # Whole_gt.append(gt)
    
        # #[  34,   96,   99,  173,  177,  181,  190,  227,  283,  319,  328,
    #     331,  378,  442,  494,  503,  552,  576,  582,  601,  632,  717,
    #     813,  835,  840,  848,  864,  905,  943,  965,  985, 1065, 1097,
    #    1182, 1194, 1202, 1223, 1232, 1244, 1267, 1274, 1283, 1286, 1329,
    #    1330, 1334, 1337, 1338, 1345, 1351, 1362, 1383, 1390, 1422, 1425,
    #    1453, 1457, 1471, 1589, 1666, 1669, 1672, 1720, 1785, 1817, 1876,
    #    1910, 1926, 1931, 1932, 1937, 1940, 1942, 1944, 1963, 1996])

    # merged_preds = np.array(Whole_pre) # the prediction value
    # labels = np.array(Whole_gt) # the grountruth value
    # labels = labels.astype(np.int)
    # labels = labels.tolist()
    # f1_optimal_thresholds = []    
    # for q in range(23):


    #     col = [x[q] for x in labels] # extract col 
    #     col = np.array(col).astype(np.int)
    #     idx = np.argwhere(col == 999).flatten()
    #     col = [i for i in col if i != 999]
   

    #     col_pre = np.array(merged_preds[:,q]).flatten()
    #     col_pre = col_pre.tolist()
        
    #     cnt = 0
    #     for i in range(len(idx)):
    #         col_pre.pop(idx[i]-cnt)
    #     #import pdb; pdb.set_trace()
    #         cnt = cnt + 1

    
    #     precision, recall, thresholds = precision_recall_curve(col,col_pre)
    #     #import pdb; pdb.set_trace()
    #     f1_optimal_thresholds.append(thresholds[np.abs(precision-recall).argmin(0)])
    
    # print(f1_optimal_thresholds)
        
        # Fx = prediction
        # Fy = gt
        # #compute Avg F score
        # TP = ((Fx == 1) & (Fy == 1)).sum()
        #     # TN    predict 和 label 同时为
        # TN = ((Fx == 0) & (Fy == 0)).sum()
        # # FN    predict 0 label 1
        # FN = ((Fx == 0) & (Fy == 1)).sum()
        # # FP    predict 1 label 0
        # FP = ((Fx == 1) & (Fy == 0)).sum()
        # Fpre  = TP.float() / (TP.float() + FP.float())
        # Frec = TP.float() / (TP.float() + FN.float())
        # F05 = (1+0.25) * Fpre * Frec / (0.25*Fpre + Frec)
        # F1 = (1+1) * Fpre * Frec / (1*Fpre + Frec)
        # F2 = (1+4) * Fpre * Frec / (4*Fpre + Frec)
        # AvgF = ( F05 + F1 + F2 ) / 3
        # #prediction[prediction<=0.5] =0
        # #prediction[prediction>0.5] =1

def image2AUvect(image,model):
    
    #import pdb; pdb.set_trace()
    #img = Image.open(path).convert("RGB")
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    #img = cv2.resize(img,(112,112),interpolation=cv2.INTER_CUBIC)
    transform_kw = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([224, 224]), # smaller side resized
        #transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD)])
    #import pdb; pdb.set_trace()
    img_tensor = transform_kw(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    #import pdb; pdb.set_trace()

    #import time

    # pause for 2 second for testing on latency
    #time.sleep(0.01)
    #prediction = F.sigmoid(model(img_tensor))#1*23
    prediction = model(img_tensor)
    prediction[prediction<0.5] =0
    prediction[prediction>=0.5] =1
    prediction= prediction.float()

    return prediction


if __name__ == '__main__':
    main()
