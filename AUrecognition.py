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

def image2AUvect(image):
    
    transform_kw = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread(image)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)

    img_tensor = transform_kw(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    #import pdb; pdb.set_trace()
    model = resnet18(end2end=True)
    #model.load_state_dict(torch.load('model_best.pth.tar',map_location='cpu'))
    #model.eval()
    #import pdb; pdb.set_trace()
    # pretrained_net = Net_OLD()
    ######
    pretrained_net_dict = torch.load('model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict)
    model.eval()

    import time

    # pause for 2 second for testing on latency
    time.sleep(0.01)
    #prediction = F.sigmoid(model(img_tensor))#1*23
    prediction = model(img_tensor)
    prediction[prediction<=0.5] =0
    prediction[prediction>0.5] =1

    
    # generate result vector
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

# hhh = image2AUvect('analysis_image.jpg')
# print(hhh)
# image_result['image_result'] = image2AUvect('')
