

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


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# def load_imgs(image_list_file):
#     #import pdb; pdb.set_trace()
#     imgs = list()
#     with open(image_list_file, 'r') as imf:
#         for line in imf:
#             # pdb.set_trace()
#             line = line.strip()
#             line = line.split()
#             img_name = line[0]
#             label_arr = line[1:]
#             img_path = '/home/jisijie/Code/test_imgs_align/' + img_name
#             # path, label, dentity_level = line.split(' ',2)
#             # label = int(label)
#             imgs.append((img_path,label_arr))
#             #print(imgs)  #imgs here actually is label of imgs
#     return imgs


def load_models():
    model_224_all = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('ir50_all_face_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_224_all.load_state_dict(new_state_dict, strict = True)
    model_224_all.eval()
    return model_224_all


def load_affect_models():
    model_affect_all = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('affectnet_ir50_all_img_model_best.pth.tar')
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
    pretrained_net_dict = torch.load('imbalanced_sample_all_img_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_imbalance_all.load_state_dict(new_state_dict, strict = True)
    model_imbalance_all.eval()
    return model_imbalance_all

RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)


def load_imgs(image_list_file):
    imgs = list()
    with open(image_list_file, 'r') as imf:
        for line in imf:
            line = line.strip()
            #import pdb; pdb.set_trace()
            img_path = '/home/jisijie/Code/test_imgs_align/' + line #root_path换成你自己的path
            line_no_jpg = line.split('.')[0]
            imgs.append((img_path,line_no_jpg))
    return imgs


class MsCelebDataset(data.Dataset):
    def __init__(self, image_list_file, transform=None):
        self.imgs= load_imgs(image_list_file)
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
        # transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])


val_list_file = 'test_imgs_align.txt'# 换成你自己的
val_dataset =  MsCelebDataset(val_list_file, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)


# class MsCelebDataset(data.Dataset):
#     def __init__(self, image_list_file, transform=None):
#         self.imgs= load_imgs(image_list_file)
#         self.transform = transform

#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         img = Image.open(path).convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
    
#     def __len__(self):
#         return len(self.imgs)


def sigmoid(x):
    return 1/(1+np.exp(-x))



def main_test():
    imgs= load_imgs('label_test_align.txt')
    #imgs_train = load_imgs('label.txt')
    Whole_pre = []
    Whole_pre_thres = []
    Whole_gt  = []
    #Train_gt = []
    #import pdb; pdb.set_trace()
    total_cnt = len(imgs)
    #total_cnt_train = len(imgs_train)
    Whole_acc = []
    Whole_F1 = []
    Whole_avg = []
    Whole_acc_thres = []
    Whole_F1_thres = []
    Whole_avg_thres = []
    model_224 = load_models()
    model_affect = load_affect_models()
    model_imbalance = load_imbalance_models()
    model_224.eval()
    model_affect.eval()
    model_imbalance.eval()
    
    #print(model.eval())
    for i in range(1886):

        #import pdb; pdb.set_trace()
        path = imgs[i][0]
        
        target_arr = np.array(imgs[i][1],dtype='int32').T
        target_tensor = torch.tensor(target_arr)
        target = target_tensor
        
        #gt = np.array(imgs[i][1]).astype(np.int)
        #img = Image.open(path).convert("RGB")
        #import pdb; pdb.set_trace()
        pred_score,prediction_thres = image2AUvect(path,model)
         
        # = pred_score.detach().numpy()
        
        Whole_pre.append(pred_score)
        Whole_pre_thres.append(prediction_thres)
        target= target.gt(0)
        target= target.float()
        # correct_num = sum(pred_score == target).sum().item()
        # mean_acc = correct_num/23
        #import pdb; pdb.set_trace()
        mean_acc = accuracy_score(target.detach().numpy(), pred_score.flatten())
        mean_acc_thres = accuracy_score(target.detach().numpy(), prediction_thres.flatten())
        Whole_acc.append(mean_acc)
        Whole_acc_thres.append(mean_acc_thres)
        #import pdb; pdb.set_trace()
        Fx = pred_score
        Fx_thres = prediction_thres
        Fy = np.array(target)
        #compute Avg F score
        TP = ((Fx == 1) & (Fy == 1)).sum()
        TP_thres = ((Fx_thres == 1) & (Fy == 1)).sum()
            # TN    predict 和 label 同时为
        TN = ((Fx == 0) & (Fy == 0)).sum()
        TN_thres = ((Fx_thres == 0) & (Fy == 0)).sum()
        # FN    predict 0 label 1
        FN = ((Fx == 0) & (Fy == 1)).sum()
        FN_thres = ((Fx_thres == 0) & (Fy == 1)).sum()
        # FP    predict 1 label 0
        FP = ((Fx == 1) & (Fy == 0)).sum()
        FP_thres =((Fx_thres == 1) & (Fy == 0)).sum()

        Fpre  = TP/ (TP + FP)
        Frec = TP / (TP + FN)
        F1 = (1+1) * Fpre * Frec / (1*Fpre + Frec)
        F1_thres =  2*TP_thres / (2*TP_thres+FP_thres+FN_thres)
        #Avg = ( mean_acc + F1.item() ) / 2
        Whole_F1.append(F1.item())
        Whole_F1_thres.append(F1_thres.item())
        print(i)
        
        #import pdb; pdb.set_trace()
        #Whole_avg.append(Avg)
    #import pdb; pdb.set_trace()
    FFFFF1 = np.array(Whole_F1)
    where_are_nan = np.isnan(FFFFF1)
    FFFFF1[where_are_nan] = 0

    FFFFF1_thres = np.array(Whole_F1_thres)
    where_are_nan_thres = np.isnan(FFFFF1_thres)
    FFFFF1_thres[where_are_nan_thres] = 0
    #FFFFF1 = [x for x in FFFFF1 if str(x) != 'nan'] 
    FFF1 = np.mean(FFFFF1)
    ACCC = np.mean(Whole_acc)
    AVG = (FFF1 + ACCC) / 2
    print('ACC',ACCC)
    print('F1',FFF1)
    print('AVG',AVG)

    FFF1_thres = np.mean(FFFFF1_thres)
    ACCC_thres = np.mean(Whole_acc_thres)
    AVG_thres = (FFF1_thres + ACCC_thres) / 2
    print('ACC',ACCC_thres)
    print('F1',FFF1_thres)
    print('AVG',AVG_thres)


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
    prediction = model(img_tensor.cuda())
    prediction = torch.sigmoid(prediction)
    #prediction_thres = prediction
    #prediction = model(img_tensor)
    prediction[prediction<0.5] =0
    prediction[prediction>=0.5] =1
    #prediction= prediction.float()

    threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    
    # for i in range(23):
    #     if prediction_thres[0,i] < threshold[i]:
    #         prediction_thres[0,i] = 0
    #     else:
    #         prediction_thres[0,i] = 1
        #prediction= prediction.float()
    return prediction
    #return prediction,prediction_thres



def main():
    #imgs= load_imgs('test_imgs_align.txt')

    #total_cnt = len(imgs)
    model_224_all = load_models()
    model_affect_all = load_affect_models()
    model_imbalance_all = load_imbalance_models()
    model_224_all.eval()
    model_affect_all.eval()
    model_imbalance_all.eval()
    #Whole_pre = []

    # for i in range(total_cnt):

    #     #import pdb; pdb.set_trace()
    #     path = imgs[i][0]
    #     pred_score = image2AUvect(path,model)
    #     import pdb; pdb.set_trace()
    #     Whole_pre.append(pred_score)

    result_dict = {}
    threshold =[0.23372596502304077, 0.1990925371646881, 0.3834395110607147, 0.14819832146167755, 0.4715411067008972, 0.5796836614608765, 0.5434879064559937, 0.41309654712677, 0.3481535017490387, 0.2778008282184601, 0.3311555087566376, 0.7886293530464172, 0.6664986610412598, 0.36747920513153076, 0.22083085775375366, 0.2278563678264618, 0.4715306758880615, 0.23590746521949768, 0.2270904779434204, 0.1872551143169403, 0.16792409121990204, 0.10406456887722015, 0.10069383680820465]

    for i, (input, tags) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        prediction = model(input_var.cuda())
        
        prediction = torch.sigmoid(prediction)
        # prediction = prediction.cpu().detach().numpy() # for thres
        # for j in range(23):
        #     if prediction[0,j] < threshold[i]:
        #         prediction[0,j] = 0
        #     else:
        #         prediction[0,j] = 1
        #import pdb; pdb.set_trace()
        prediction[prediction>=0.5] = 1
        prediction[prediction<0.5] = 0


    #     prediction1 = model_224_all(input_var.cuda())
    #     prediction2 = model_affect_all(input_var.cuda())
    #     prediction3 = model_imbalance_all(input_var.cuda())
    # #prediction = prediction1 
    #     prediction = (prediction1 + prediction2 + prediction3) / 3
    #     prediction = torch.sigmoid(prediction)
    #     #import pdb; pdb.set_trace()
    #     prediction = prediction.cpu()
    #     prediction = prediction.detach().numpy()
    #     threshold =[0.23372596502304077, 0.1990925371646881, 0.3834395110607147, 0.14819832146167755, 0.4715411067008972, 0.5796836614608765, 0.5434879064559937, 0.41309654712677, 0.3481535017490387, 0.2778008282184601, 0.3311555087566376, 0.7886293530464172, 0.6664986610412598, 0.36747920513153076, 0.22083085775375366, 0.2278563678264618, 0.4715306758880615, 0.23590746521949768, 0.2270904779434204, 0.1872551143169403, 0.16792409121990204, 0.10406456887722015, 0.10069383680820465]
        
    #     for j in range(23):
    #         if prediction[0,j] < threshold[j]:
    #             prediction[0,j] = 0
    #         else:
    #             prediction[0,j] = 1

        # prediction[prediction<0.5] =0
        # prediction[prediction>=0.5] =1

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
        #import pdb; pdb.set_trace()
        result_dict[tags[0]] = result
        print(i)
    # root_path = '/home/jisijie/Code/test_imgs_align/'
    # with open("test_imgs_align.txt","r") as file:
    #     for line in file:
    #         line = line.strip()
    #         #import pdb; pdb.set_trace()
    #         img_path = root_path + line
    #         line_no_jpg = line.split('.')[0].split('_')[-1]
    #         result_dict[line_no_jpg] = image2AUvect(img_path,model)
    #         print("line",line)
    np.save('107k_result_dict_all_thres.npy', result_dict)



if __name__ == '__main__':
    main()
