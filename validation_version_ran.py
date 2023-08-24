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
from sklearn.metrics import f1_score
import pickle
from model_irse_ran import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import torchvision.transforms as transforms
import torch
import collections
from collections import OrderedDict

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def load_ran_imgs(image_list_file):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    imgs_forth = list()
    imgs_fifth = list()
    imgs_sixth = list()
    labels = list()
    with open(image_list_file, 'r') as imf:
    	for line in imf:
            #pdb.set_trace()
            line = line.strip()
            line = line.split()
            img_name = line[0]
            label_arr = line[1:]
            # pdb.set_trace()
            img_path = '/home/jisijie/Code/crop_image_test/' + img_name.split('.')[0] +'/'
            img_list = os.listdir(img_path)
            img_list.sort()
            # import pdb; pdb.set_trace()
            imgs_first.append((img_path+img_list[0], label_arr))
            imgs_second.append((img_path+img_list[1], label_arr))
            imgs_third.append((img_path+img_list[2], label_arr))
            imgs_forth.append((img_path+img_list[3], label_arr))
            imgs_fifth.append((img_path+img_list[4], label_arr))
            imgs_sixth.append((img_path+img_list[5], label_arr))
            labels.append(label_arr)
    return imgs_first,imgs_second,imgs_third, imgs_forth, imgs_fifth, imgs_sixth,labels


def load_imgs_label(image_list_file):
    #import pdb; pdb.set_trace()
    imgs = list()
    with open(image_list_file, 'r') as imf:
        for line in imf:
            # pdb.set_trace()
            line = line.strip()
            line = line.split()
            img_name = line[0]
            label_arr = line[1:]
            img_path = '/home/jisijie/Code/img_dataset_align/' + img_name
            # path, label, dentity_level = line.split(' ',2)
            # label = int(label)
            imgs.append((img_path,label_arr))
            #print(imgs)  #imgs here actually is label of imgs
    return imgs


class ran_MsCelebDataset(data.Dataset):
    def __init__(self, image_list_file, transform=None):
        self.imgs_first, self.imgs_second, self.imgs_third, self.imgs_forth, self.imgs_fifth, self.imgs_sixth, self.labels = load_ran_imgs(image_list_file)
        self.transform = transform

    def _get_all_label(self):
        return self.labels

    def __getitem__(self, index):
        path_firt, target_first = self.imgs_first[index]
        img_first = Image.open(path_firt).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)
        
        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)

        path_forth, target_forth = self.imgs_forth[index]
        img_forth = Image.open(path_forth).convert("RGB")
        if self.transform is not None:
            img_forth = self.transform(img_forth)

        path_fifth, target_fifth = self.imgs_fifth[index]
        img_fifth = Image.open(path_fifth).convert("RGB")
        if self.transform is not None:
            img_fifth = self.transform(img_fifth)

        path_sixth, target_sixth = self.imgs_sixth[index]
        img_sixth = Image.open(path_sixth).convert("RGB")
        if self.transform is not None:
            img_sixth = self.transform(img_sixth)

        return img_first, target_first ,img_second,target_second,img_third,target_third,img_forth,target_forth, img_fifth, target_fifth, img_sixth, target_sixth
    
    def __len__(self):
        return len(self.imgs_first)

def sigmoid(x):
    return 1/(1+np.exp(-x))


def load_models():
    model = IR_152([224,224]).cuda()
    # model = torch.nn.DataParallel(model)
    #cudnn.benchmark = True
    pretrained_net_dict = torch.load('ran_ir152_imbalanced_sample_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict, strict = True)
    model.eval()
    return model


NPUT_SIZE = [224,224]
RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)

val_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([224, 224]), # smaller side resized
        # transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])


val_list_file = 'label_test_align.txt'# 换成你自己的
val_dataset =  ran_MsCelebDataset(val_list_file, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
model = load_models()

def main():
    #imgs= load_imgs_label('label_test_align.txt')
    #pdb.set_trace()
    Whole_pre = []
    Whole_pre_thres = []
    Whole_gt  = []
    Whole_acc = []
    Whole_F1 = []
    Whole_avg = []
    for i, (input_first, target_first, input_second,target_second, input_third, target_third, input_forth, target_forth, input_fifth, target_fifth, input_sixth, target_sixth) in enumerate(val_loader):
        # measure data loading time
        
        #target = imgs[i][1]
        input = torch.zeros([input_first.shape[0],input_first.shape[1],input_first.shape[2],input_first.shape[3],6])
        input[:,:,:,:,0] = input_first
        input[:,:,:,:,1] = input_second
        input[:,:,:,:,2] = input_third
        input[:,:,:,:,3] = input_forth
        input[:,:,:,:,4] = input_fifth
        input[:,:,:,:,5] = input_sixth
        
        #pdb.set_trace()

        target_arr = np.array(target_first,dtype='int32').T
        target_tensor = torch.tensor(target_arr)
        target = target_tensor
        # input_var = torch.autograd.Variable(input, volatile=True).cuda()
        #target_var = torch.autograd.Variable(target, volatile=True)
        
        
        # compute output
        # import pdb; pdb.set_trace()
        pred_score = model(input.cuda())
        prediction = torch.sigmoid(pred_score)
        
        #import pdb; pdb.set_trace()
        prediction[prediction>=0.5] = 1
        prediction[prediction<0.5] = 0
        prediction = prediction.cpu()
        prediction = prediction.detach().numpy()

        #target = target.cpu()
        target = target.detach().numpy()
        Whole_pre.append(prediction.flatten())
        Whole_gt.append(target.flatten().tolist())
        #import pdb; pdb.set_trace()
        print(i)
    
    #pdb.set_trace()
    labels = Whole_gt # the grountruth value
    #labels = labels.astype(np.int)
    #labels = labels.tolist()
    merged_preds = np.array(Whole_pre)

    for q in range(23):

        import pdb; pdb.set_trace()
        col = [x[q] for x in labels] # extract col 
        col = np.array(col).astype(np.int)
        idx = np.argwhere(col == 999).flatten()
        #col.index(999)
        col = [i for i in col if i != 999]
   

        col_pre = np.array(merged_preds[:,q]).flatten()
        col_pre = col_pre.tolist()
        
        cnt = 0
        for i in range(len(idx)):
            col_pre.pop(idx[i]-cnt)
        #import pdb; pdb.set_trace()
            cnt = cnt + 1

        F1 = f1_score(col,col_pre)
        mean_acc = accuracy_score(col, col_pre)
        AVG = (F1 + mean_acc) / 2
        Whole_acc.append(mean_acc)
        Whole_F1.append(F1)
        Whole_avg.append(AVG)
        #import pdb; pdb.set_trace()

    print('Class ACC',np.mean(Whole_acc))
    print('Class F1',np.mean(Whole_F1))
    print('Class AVG',np.mean(Whole_avg))

        



if __name__ == '__main__':
    main()
