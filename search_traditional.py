
import torch
from torchvision import transforms
import numpy as np
#from PIL import Image
#from ResNet import resnet18, resnet50, resnet101
import collections
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import os, sys, shutil
import random as rd
from model_irse_224 import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import time

#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import roc_curve
from sklearn.utils.extmath import stable_cumsum
import pickle

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

def load_models():
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

# def load_models():
#     model_224 = IR_152([224,224]).cuda()
#     pretrained_net_dict = torch.load('ir152_weighted_loss_model_best.pth.tar')
#     new_state_dict = OrderedDict()
    
#     for k, v in pretrained_net_dict['state_dict'].items():
#         # import pdb; pdb.set_trace()
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     #import pdb; pdb.set_trace()
#     model_224.load_state_dict(new_state_dict, strict = True)
#     model_224.eval()
#     return model_224


def load_affect_models():
    model_affect = IR_50([224,224]).cuda()
    pretrained_net_dict = torch.load('affectnet_model_best.pth.tar')
    new_state_dict = OrderedDict()
    
    for k, v in pretrained_net_dict['state_dict'].items():
        # import pdb; pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #import pdb; pdb.set_trace()
    model_affect.load_state_dict(new_state_dict, strict = True)
    model_affect.eval()
    return model_affect

RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
RGB_STD = (0.5, 0.5, 0.5)




def sigmoid(x):
    return 1/(1+np.exp(-x))

def main():
    
    f=open('/home/jisijie/Code/align_gt.txt','rb')
    Whole_gt=pickle.load(f)
    #print(Whole_gt)
    f.close()

    f=open('/home/jisijie/Code/align_preds.txt','rb')
    Whole_pre=pickle.load(f)
    #print(Whole_pre)
    f.close()
    #import pdb; pdb.set_trace()
    merged_preds = np.array(Whole_pre) # the prediction value
    merged_preds = sigmoid(merged_preds) 

    labels = np.array(Whole_gt) # the grountruth value
    labels = labels.astype(np.int)
    labels = labels.tolist()
    f1_optimal_thresholds = []
    acc_optimal_thresholds = []

    for q in range(23):

        import pdb; pdb.set_trace()
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
        #
        #import pdb; pdb.set_trace()
        threshold = precision_recall_curve(col,col_pre)
        #fpr, tpr, ac_thresholds = roc_curve(col, col_pre)
        #accuracy_score(target, pred_score)
        #import pdb; pdb.set_trace()

        f1_optimal_thresholds.append(threshold)
        # acc_optimal_thresholds.append(ac_thresholds[np.abs(precision-recall).argmin(0)])

    #precision, recall, thresholds = precision_recall_curve(Whole_gt, Whole_pre)
    
    print('F',f1_optimal_thresholds)
    
    #print('ACC',acc_optimal_thresholds)


    # for i in range(23):
    #     import pdb; pdb.set_trace()
    #     threshold= np.arange(0.0,1.0,0.1)





        


def precision_recall_curve(y_true, probas_pred, pos_label=None,
                           sample_weight=None):
    fps, tps, tns ,thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    tns = np.array(tns)
    
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    acc = (tps + tns) /len(tns)
    f1 = 2*precision*recall / (precision + recall)
    avg = ( acc + f1 )/ 2
    #import pdb; pdb.set_trace()
    avg = avg.tolist()
    idx = avg.index(max(avg))
    
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return thresholds[idx]



def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    #import pdb; pdb.set_trace()
    # ensure binary classification if pos_label is not specified
    #classes = np.unique(y_true)
    
    pos_label = 1.

    # make y_true a boolean vector
    #y_true = (y_true == pos_label)
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
 
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # We need to use isclose to avoid spurious repeated thresholds
    # stemming from floating point roundoff errors.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    #import pdb; pdb.set_trace()
    # accumulate the true positives with decreasing threshold
    tps = y_true.cumsum()[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    thresholds = y_score[threshold_idxs]
    tp = []
    fp = []
    tns = []
    for threshold in thresholds:
       
        #import pdb; pdb.set_trace()
        y_prob = [1 if i>=threshold else 0 for i in y_score]

        result = [i==j for i,j in zip(y_true, y_prob)]
    
        # positive = [i==1 for i in y_prob]
        negative = [i==0 for i in y_prob]

        # tp = [i and j for i,j in zip(result, positive)] # 预测为正类且预测正确
        # fp = [(not i) and j for i,j in zip(result, positive)] # 预测为正类且预测错误
        tn = [i and j for i,j in zip(result, negative)]
        nmb = tn.count(True)
        tns.append(nmb)

    #tn = np.array(tn).astype(int)
    # fp = np.array(fp).astype(int)
    #tns = tn.cumsum()
    # fps = fp.cumsum()
        #import pdb; pdb.set_trace()
        #tp.append(tp.count(True))
        #print(tp.count(True), fp.count(True),tn.count(True))
        
    #import pdb; pdb.set_trace()
    return fps, tps, tns, thresholds



if __name__ == '__main__':
    main()

#[0.5496821403503418, 0.5463277697563171, 0.5966190099716187, 0.5366142392158508, 0.6078689098358154, 0.6426506638526917, 0.6381572484970093, 0.6034342050552368, 0.5845600962638855, 0.566964328289032, 0.5860170722007751, 0.6481299996376038, 0.6580347418785095, 0.5925973653793335, 0.5516956448554993, 0.5585710406303406, 0.604377031326294, 0.5559682250022888, 0.5546668767929077, 0.5457150340080261, 0.5418009757995605, 0.5239280462265015, 0.5237650871276855]

#[0.5494877099990845, 0.5463277697563171, 0.5953701138496399, 0.5364421606063843, 0.6063692569732666, 0.6356817483901978, 0.6360287070274353, 0.5978166460990906, 0.5840008854866028, 0.566964328289032, 0.5851378440856934, 0.6481299996376038, 0.654565155506134, 0.5881308317184448, 0.5509662628173828, 0.5585710406303406, 0.604377031326294, 0.5569213628768921, 0.5546668767929077, 0.5461963415145874, 0.5418009757995605, 0.5239280462265015, 0.5237650871276855]


# threshold for best_align_model
# [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]  
# 
# threshold for   align_224_27 model
# [0.19938641786575317, 0.18584410846233368, 0.3913974165916443, 0.14671951532363892, 0.4383629560470581, 0.5868874788284302, 0.567375123500824, 0.4197950065135956, 0.34152165055274963, 0.26947635412216187, 0.34752434492111206, 0.6108295321464539, 0.6545486450195312, 0.3747131824493408, 0.2075241655111313, 0.23536483943462372, 0.423736572265625, 0.22481508553028107, 0.21954506635665894, 0.18337230384349823, 0.16759498417377472, 0.09578543901443481, 0.09513197839260101]