# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 08:59:48 2018
Output template for EmotioNet recognition

@author: Qianli
"""
# -------------------------------------------------------track 1 AU recognition
import torch
from torchvision import transforms
#from PIL import Image
from ResNet import resnet18, resnet50, resnet101
import collections
from collections import OrderedDict
import torch.nn.functional as F
import cv2
from PIL import Image
import random as rd
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152

def load_models():
    model = IR_50([112,112])
    
    pretrained_net_dict = torch.load('model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        
        name = k[7:] # remove `module.`
        #print(name)
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict = False)
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


        # center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
        #          (img.height/2 + crop_center_y_offset)*(1+trans_diff_y))

        
        # if center[0] < crop_width_aug/2:
        #     crop_width_aug = center[0]*2-0.5
        # if center[1] < crop_height_aug/2:
        #     crop_height_aug = center[1]*2-0.5
        # if (center[0]+crop_width_aug/2) >= img.width:
        #     crop_width_aug = (img.width-center[0])*2-0.5
        # if (center[1]+crop_height_aug/2) >= img.height:
        #     crop_height_aug = (img.height-center[1])*2-0.5

        # crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
        #             center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

        #mid_img = img.crop(crop_box)
        res_img = img.resize( (final_width, final_height) )
        #import pdb; pdb.set_trace()
        return res_img

model = load_models()
model.eval()
RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)



def image2AUvect(image):
    
  
    #img = Image.open(path).convert("RGB")
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    import pdb; pdb.set_trace()
    #img = cv2.resize(img,(112,112),interpolation=cv2.INTER_CUBIC)
    transform_kw = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([112, 112]), # smaller side resized
        #transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    #import pdb; pdb.set_trace()
    img_tensor = transform_kw(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    #import pdb; pdb.set_trace()

    import time

    # pause for 2 second for testing on latency
    time.sleep(0.01)
    #prediction = F.sigmoid(model(img_tensor))#1*23
    prediction = model(img_tensor)
    prediction[prediction<0.5] =0
    prediction[prediction>=0.5] =1
    #import pdb; pdb.set_trace()

    
    # generate result vector
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
   
    return result



### test & debug

hhh = image2AUvect('analysis_image.jpg')
print(hhh)
# image_result['image_result'] = image2AUvect('')
