# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 08:59:48 2018
Output template for EmotioNet recognition

@author: Qianli
"""
# -------------------------------------------------------track 1 AU recognition
import torch
from torchvision import transforms
from PIL import Image
from ResNet import resnet18, resnet50, resnet101

from model_irse_224 import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import collections
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import numpy as np

import os, sys, shutil
import random as rd

from PIL import Image
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def load_imgs(image_list_file):
    imgs = list()
    error_img = []
    with open(image_list_file, 'r') as imf:
        for line in imf:
            line = line.strip()
            #import pdb; pdb.set_trace()
            img_path = '/home/jisijie/Code/test_imgs2_align/' + line #root_path换成你自己的path
            img = cv2.imread(img_path)
            if img is not None:
                line_no_jpg = line.split('.')[0].split('_')[-1]
                imgs.append((img_path,line_no_jpg))
            else:
                import pdb; pdb.set_trace()
                error_img.append(line)
    return imgs


class MsCelebDataset(data.Dataset):
    def __init__(self, image_list_file, transform=None):
        self.imgs = load_imgs(image_list_file)
        self.transform = transform

    def __getitem__(self, index):
        path, tags = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, tags
    
    def __len__(self):
        return len(self.imgs)

INPUT_SIZE = [224,224]
RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)

val_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([224, 224]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])


val_list_file = 'test_imgs2_align.txt'# 换成你自己的
val_dataset =  MsCelebDataset(val_list_file, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

def load_224_models():
    model_224 = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('align_224_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_224.load_state_dict(new_state_dict, strict = True)
    model_224.eval()
    return model_224


def load_affect_models():
    model_affect_all = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('affectnet_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_affect_all.load_state_dict(new_state_dict, strict = True)
    model_affect_all.eval()
    return model_affect_all

def load_imbalance_models():
    model_imbalance_all = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('imbalanced_sample_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_imbalance_all.load_state_dict(new_state_dict, strict = True)
    model_imbalance_all.eval()
    return model_imbalance_all


def load_152weighted_models():
    model_224_all = IR_152([224,224]).cuda()
    pretrained_net_dict = torch.load('ir152_weighted_loss_64_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_224_all.load_state_dict(new_state_dict, strict = True)
    model_224_all.eval()
    return model_224_all

def load_152imbalanced_models():
    model_224_all = IR_152([224,224]).cuda()
    pretrained_net_dict = torch.load('ir152_imbalanced_sample_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_224_all.load_state_dict(new_state_dict, strict = True)
    model_224_all.eval()
    return model_224_all


def main():
    #imgs,error_img = load_imgs('test_imgs2_align.txt')
    #import pdb; pdb.set_trace()
    model_224 = load_224_models()
    model_224.eval()
    model_affect = load_affect_models()
    model_affect.eval()
    model_imbalance = load_imbalance_models()
    model_imbalance.eval()
    model_152imbalanced = load_152imbalanced_models()   
    model_152imbalanced.eval()
    model_152_weighted = load_152weighted_models()
    model_152_weighted.eval()

    result_dict = {}
    threshold =[0.23372596502304077, 0.1990925371646881, 0.3834395110607147, 0.14819832146167755, 0.4715411067008972, 0.5796836614608765, 0.5434879064559937, 0.41309654712677, 0.3481535017490387, 0.2778008282184601, 0.3311555087566376, 0.7886293530464172, 0.6664986610412598, 0.36747920513153076, 0.22083085775375366, 0.2278563678264618, 0.4715306758880615, 0.23590746521949768, 0.2270904779434204, 0.1872551143169403, 0.16792409121990204, 0.10406456887722015, 0.10069383680820465]


    for i, (input, tags) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)#记得检查tags是不是六位编码
    # prediction = model(input_var)
    # prediction = F.sigmoid(prediction)
    # prediction[prediction>=0.5] = 1
    # prediction[prediction<0.5] = 0


        prediction1 = model_224(input_var.cuda())
        prediction2 = model_affect(input_var.cuda())
        prediction3 = model_imbalance(input_var.cuda())
        prediction4 = model_152_weighted(input_var.cuda())
        prediction5 = model_152imbalanced(input_var.cuda())
    #prediction = prediction1 
        prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
        prediction = torch.sigmoid(prediction)
    #prediction_thres = prediction
        prediction = prediction.cpu()
        prediction = prediction.detach().numpy()
    
    #threshold =[0.23372596502304077, 0.1990925371646881, 0.3834395110607147, 0.14819832146167755, 0.4715411067008972, 0.5796836614608765, 0.5434879064559937, 0.41309654712677, 0.3481535017490387, 0.2778008282184601, 0.3311555087566376, 0.7886293530464172, 0.6664986610412598, 0.36747920513153076, 0.22083085775375366, 0.2278563678264618, 0.4715306758880615, 0.23590746521949768, 0.2270904779434204, 0.1872551143169403, 0.16792409121990204, 0.10406456887722015, 0.10069383680820465]
        for j in range(23):
            if prediction[0,j] < threshold[j]:
                prediction[0,j] = 0
            else:
                prediction[0,j] = 1

        #import pdb; pdb.set_trace()
        result = [999] * 60
        result[0] = int(prediction[0][0].item())   # AU01
        result[1] = int(prediction[0][1].item())   # AU02      
        result[3] = int(prediction[0][2].item())  # AU04
        result[4] = int(prediction[0][3].item())   # AU05
        result[5] = int(prediction[0][4].item())   # AU06
        result[8] = int(prediction[0][5].item())   # AU09
        result[9] = int(prediction[0][6].item())   # AU10
        result[11] = int(prediction[0][7].item())  # AU12
        result[14] = int(prediction[0][8].item())  # AU15
        result[16] = int(prediction[0][9].item())  # AU17
        result[17] = int(prediction[0][10].item())  # AU18
        result[19] = int(prediction[0][11].item())  # AU20
        result[23] = int(prediction[0][12].item())  # AU24
        result[24] = int(prediction[0][13].item())  # AU25
        result[25] = int(prediction[0][14].item())  # AU26
        result[27] = int(prediction[0][15].item())  # AU28
        result[42] = int(prediction[0][16].item())  # AU43
        result[50] = int(prediction[0][17].item())  # AU51
        result[51] = int(prediction[0][18].item())  # AU52
        result[52] = int(prediction[0][19].item())  # AU53
        result[53] = int(prediction[0][20].item())  # AU54
        result[54] = int(prediction[0][21].item())  # AU55
        result[55] = int(prediction[0][22].item())  # AU56

        import pdb; pdb.set_trace()
        result_dict[tags[0]] = result
        print(i)


    np.save('final_test_result.npy', result_dict)


if __name__ == '__main__':
    main()