

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
from sklearn.metrics import f1_score
import pickle
from model_irse_224 import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
            img_path = '/home/jisijie/Code/img_dataset_align/' + img_name
            # path, label, dentity_level = line.split(' ',2)
            # label = int(label)
            imgs.append((img_path,label_arr))
            #print(imgs)  #imgs here actually is label of imgs
    return imgs

# all data

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

RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)

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
    #print(model.eval())
    for i in range(total_cnt):

        #import pdb; pdb.set_trace()
        path = imgs[i][0]
        
        target = imgs[i][1]
        #target_arr = np.array(imgs[i][1],dtype='int32').T
        #target_tensor = torch.tensor(target_arr)
        #target = target_tensor
        
        #gt = np.array(imgs[i][1]).astype(np.int)
        #img = Image.open(path).convert("RGB")
        #import pdb; pdb.set_trace()
        #pred_score = image2AUvectfusion3(path,model_affect,model_224,model_imbalance)
        pred_score = image2AUvectfusion5(path,model_224,model_affect,model_imbalance,model_152imbalanced,model_152_weighted)
        #import pdb; pdb.set_trace()

        # 224_align
        #threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
        
        # affect
        #threshold = [0.31092214584350586, 0.23688571155071259, 0.39763566851615906, 0.20768088102340698, 0.46962788701057434, 0.614916205406189, 0.5190575122833252, 0.41517284512519836, 0.3829299807548523, 0.36871975660324097, 0.3304348587989807, 0.9005534052848816, 0.6617922186851501, 0.36772528290748596, 0.2704208195209503, 0.24678117036819458, 0.49305975437164307, 0.25313812494277954, 0.2387600839138031, 0.1992359161376953, 0.1803692877292633, 0.11917730420827866, 0.11412036418914795]

        #fusion
        #threshold =[0.23372596502304077, 0.1990925371646881, 0.3834395110607147, 0.14819832146167755, 0.4715411067008972, 0.5796836614608765, 0.5434879064559937, 0.41309654712677, 0.3481535017490387, 0.2778008282184601, 0.3311555087566376, 0.7886293530464172, 0.6664986610412598, 0.36747920513153076, 0.22083085775375366, 0.2278563678264618, 0.4715306758880615, 0.23590746521949768, 0.2270904779434204, 0.1872551143169403, 0.16792409121990204, 0.10406456887722015, 0.10069383680820465]

        #152 threshold
        #threshold = [0.1954658180475235, 0.14763331413269043, 0.5626868605613708, 0.17851510643959045, 0.4197644889354706, 0.4298638701438904, 0.6756932735443115, 0.3584326207637787, 0.4915931820869446, 0.39378973841667175, 0.40083447098731995, 0.5579985976219177, 0.5807003378868103, 0.3423861265182495, 0.32387253642082214, 0.33025434613227844, 0.49145999550819397, 0.2706698775291443, 0.2643400728702545, 0.17712070047855377, 0.2642246186733246, 0.11149978637695312, 0.17064517736434937]

        # avg threshold
        #threshold =[0.21459589, 0.17336293, 0.47306319, 0.16335671, 0.4456528 ,0.50477377, 0.60959059, 0.38576458, 0.41987334, 0.33579528,0.36599499, 0.67331398, 0.6235995 , 0.35493267, 0.2723517 ,0.27905536, 0.48149534, 0.25328867, 0.24571528, 0.18218791,0.21607435, 0.10778218, 0.13566951]
        # 152:old = 2: 1
        threshold =[0.2082192 , 0.16478639, 0.50293774, 0.16840951, 0.43702336,0.4798038 , 0.63162482, 0.37665393, 0.44377996, 0.35512677,0.37760815, 0.63487552, 0.60929978, 0.35075049, 0.28952531,0.29612169, 0.48481689, 0.25908241, 0.25192354, 0.18049884,0.23212444, 0.10902138, 0.14732806]
        
         # 5 model threshold
        #threshold = [0.5588093996047974, 0.5451753735542297, 0.5902812480926514, 0.5334650874137878, 0.5998384952545166, 0.6265903115272522, 0.6307288408279419, 0.5836422443389893, 0.5863364934921265, 0.5646711587905884, 0.5919898748397827, 0.6713384985923767, 0.676170825958252, 0.571273148059845, 0.5569359064102173, 0.5476313233375549, 0.6037800312042236, 0.5590639710426331, 0.5542662143707275, 0.5427563786506653, 0.5439549088478088, 0.5230002403259277, 0.5266437530517578]
    
        for j in range(23):
            if pred_score[0,j] < threshold[j]:
                pred_score[0,j] = 0
            else:
                pred_score[0,j] = 1

        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        #pred_score_affect = image2AUvect(path,model_affect)
         
        # = pred_score.detach().numpy()
        
        
        print(i)
        Whole_pre.append(pred_score.flatten())
        #Whole_pre_thres.append(prediction_thres)
        # target= target.gt(0)
        # target= target.float()
        Whole_gt.append(target)
        # correct_num = sum(pred_score == target).sum().item()
        # mean_acc = correct_num/23
    #import pdb; pdb.set_trace()
    labels = np.array(Whole_gt) # the grountruth value
    labels = labels.astype(np.int)
    labels = labels.tolist()
    merged_preds = np.array(Whole_pre)

    for q in range(23):

        #import pdb; pdb.set_trace()
        col = [x[q] for x in labels] # extract col 
        col = np.array(col).astype(np.int)
        idx = np.argwhere(col == 999).flatten()
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


def image2AUvectfusion5(image,model1,model2,model3,model4,model5):
    
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
    prediction1 = model1(img_tensor.cuda())
    prediction2 = model2(img_tensor.cuda())
    prediction3 = model3(img_tensor.cuda())
    prediction4 = model4(img_tensor.cuda())
    prediction5 = model5(img_tensor.cuda())
    import pdb; pdb.set_trace()
    #prediction = prediction1 
    prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
    prediction = torch.sigmoid(prediction)
    #prediction_thres = prediction
    #prediction = model(img_tensor)

    # prediction[prediction<0.5] =0
    # prediction[prediction>=0.5] =1


    #prediction= prediction.float()
    # # 5 model threshold
    # threshold = [0.5588093996047974, 0.5451753735542297, 0.5902812480926514, 0.5334650874137878, 0.5998384952545166, 0.6265903115272522, 0.6307288408279419, 0.5836422443389893, 0.5863364934921265, 0.5646711587905884, 0.5919898748397827, 0.6713384985923767, 0.676170825958252, 0.571273148059845, 0.5569359064102173, 0.5476313233375549, 0.6037800312042236, 0.5590639710426331, 0.5542662143707275, 0.5427563786506653, 0.5439549088478088, 0.5230002403259277, 0.5266437530517578]
    # #threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    
    # for i in range(23):
    #     if prediction_thres[0,i] < threshold[i]:
    #         prediction_thres[0,i] = 0
    #     else:
    #         prediction_thres[0,i] = 1
    #     #prediction= prediction.float()
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()



    return prediction


def image2AUvectfusion3(image,model1,model2,model3):
    
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
    prediction1 = model1(img_tensor.cuda())
    prediction2 = model2(img_tensor.cuda())
    prediction3 = model3(img_tensor.cuda())
    #prediction = prediction1 
    prediction = (prediction1 + prediction2 + prediction3) / 3
    prediction = torch.sigmoid(prediction)
    #prediction_thres = prediction
    #prediction = model(img_tensor)

    # prediction[prediction<0.5] =0
    # prediction[prediction>=0.5] =1


    #prediction= prediction.float()

    #threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    
    # for i in range(23):
    #     if prediction_thres[0,i] < threshold[i]:
    #         prediction_thres[0,i] = 0
    #     else:
    #         prediction_thres[0,i] = 1
        #prediction= prediction.float()
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()



    return prediction
    #return prediction,prediction_thres


def image2AUvectfusion(image,model1,model2):
    
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
    prediction1 = model1(img_tensor.cuda())
    prediction2 = model2(img_tensor.cuda())
    #prediction = prediction1 
    prediction = (prediction1 + prediction2) / 2
    prediction = torch.sigmoid(prediction)
    #prediction_thres = prediction
    #prediction = model(img_tensor)

    prediction[prediction<0.5] =0
    prediction[prediction>=0.5] =1


    #prediction= prediction.float()

    #threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    
    # for i in range(23):
    #     if prediction_thres[0,i] < threshold[i]:
    #         prediction_thres[0,i] = 0
    #     else:
    #         prediction_thres[0,i] = 1
        #prediction= prediction.float()
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()



    return prediction
    #return prediction,prediction_thres


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
    # prediction[prediction<0.5] =0
    # prediction[prediction>=0.5] =1


    #prediction= prediction.float()

    #threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    


    # for i in range(23):
    #     if prediction_thres[0,i] < threshold[i]:
    #         prediction_thres[0,i] = 0
    #     else:
    #         prediction_thres[0,i] = 1
        #prediction= prediction.float()
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    return prediction
    #return prediction,prediction_thres



def main():
    #imgs= load_imgs('test_imgs_align.txt')

    #total_cnt = len(imgs)
    model = load_models()
    model.eval()
    #Whole_pre = []

    # for i in range(total_cnt):

    #     #import pdb; pdb.set_trace()
    #     path = imgs[i][0]
    #     pred_score = image2AUvect(path,model)
    #     import pdb; pdb.set_trace()
    #     Whole_pre.append(pred_score)

    result_dict = {}
    threshold = [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]
    
    for i, (input, tags) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        prediction = model(input_var.cuda())
        
        prediction = torch.sigmoid(prediction)
        # prediction = prediction.cpu().detach().numpy() # for thres
        # for j in range(23):
        #     if prediction[0,j] < threshold[j]:
        #         prediction[0,j] = 0
        #     else:
        #         prediction[0,j] = 1

        #import pdb; pdb.set_trace()

        prediction[prediction>=0.5] = 1
        prediction[prediction<0.5] = 0

        mean_acc = accuracy_score(target.detach().numpy(), pred_score.flatten())
        mean_acc_thres = accuracy_score(target.detach().numpy(), prediction_thres.flatten())
        Whole_acc.append(mean_acc)
        Whole_acc_thres.append(mean_acc_thres)

        # result = [999] * 60
        # result[0] = int(prediction[0][0].item())   # AU01
        # result[1] = int(prediction[0][1].item())   # AU02      
        # result[3] = int(prediction[0][2].item())  # AU04
        # result[4] = int(prediction[0][3].item())   # AU05
        # result[5] = int(prediction[0][4].item())   # AU06
        # result[8] = int(prediction[0][5].item())   # AU09
        # result[9] = int(prediction[0][6].item())   # AU10
        # result[11] = int(prediction[0][7].item())  # AU12
        # result[14] = int(prediction[0][8].item())  # AU15
        # result[16] = int(prediction[0][9].item())  # AU17
        # result[17] = int(prediction[0][10].item())  # AU18
        # result[19] = int(prediction[0][11].item())  # AU20
        # result[23] = int(prediction[0][12].item())  # AU24
        # result[24] = int(prediction[0][13].item())  # AU25
        # result[25] = int(prediction[0][14].item())  # AU26
        # result[27] = int(prediction[0][15].item())  # AU28
        # result[42] = int(prediction[0][16].item())  # AU43
        # result[50] = int(prediction[0][17].item())  # AU51
        # result[51] = int(prediction[0][18].item())  # AU52
        # result[52] = int(prediction[0][19].item())  # AU53
        # result[53] = int(prediction[0][20].item())  # AU54
        # result[54] = int(prediction[0][21].item())  # AU55
        # result[55] = int(prediction[0][22].item())  # AU56
        # #import pdb; pdb.set_trace()
        # result_dict[tags[0]] = result
        print(i)
    
    #np.save('107k_result_dict_thres.npy', result_dict)



if __name__ == '__main__':
    main_test()
