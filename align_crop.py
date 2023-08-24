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

def load_imgs(image_list_file):
    #import pdb; pdb.set_trace()
    imgs = list()
    with open(image_list_file, 'r') as imf:
        name = []
        for line in imf:
            #pdb.set_trace()
            line = line.strip()
            line = line.split()
            img_name = line[0]
            label_arr = line[1:]
            img_path = '/home/jisijie/Code/img_dataset_align/' + img_name
            # path, label, dentity_level = line.split(' ',2)
            # label = int(label)
            imgs.append((img_path,label_arr))
            name.append(img_name)
    return imgs,name



def main():
    
    imgs,img_name = load_imgs('label_test_align.txt')
    total_cnt = len(imgs)
    for i in range(total_cnt):
        
        path = imgs[i][0]
        img = Image.open(path).convert("RGB")
        name = img_name[i]
        width, height = img.size
        box = img.copy()
        box1 = (0,0,0.75*height,0.75*width)
        box2 = (0,0.25*width,0.75*height,width)
        box3 = (0.25*height,0.125*width,height,0.875*width)
        box4 = (0.1*height,0.1*width,0.9*height,0.9*width)
        box5 = (0.15*height,0.15*width,0.85*height,0.85*width)
        
        upleft_img = box.crop(box1)
        upright_img = box.crop(box2)
        centerlow_img = box.crop(box3)
        centerbig_img = box.crop(box4)
        centersmall_img = box.crop(box5)
        
        import pdb; pdb.set_trace()

        savepath1 = '/home/jisijie/Code/crop_image_test/' + name
        savepath2 = 'image'+[i]
        upleft_img.save(savepath1)
        savename = os.path.join(savepath,) 
        



if __name__ == '__main__':
    main()