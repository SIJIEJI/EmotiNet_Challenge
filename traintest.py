import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
#Model
from ResNet import resnet18, resnet50, resnet101
from selfDefine import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb

##############################################################
def main():
    global args, best_prec1
    args = parser.parse_args()
    print('end2end?:', args.end2end)
    train_list_file = 'label.txt'
    caffe_crop = CaffeCrop('train')
    train_dataset =  MsCelebDataset(train_list_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
   
    caffe_crop = CaffeCrop('test')
    val_list_file = 'label_test.txt'
    val_dataset =  MsCelebDataset(val_list_file, 
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # prepare model
    model = None
    assert(args.arch in ['resnet18','resnet50','resnet101'])
    if args.arch == 'resnet18':
        print ('we use resnet18')
        model = resnet18(end2end=args.end2end)
   #     model = resnet18(pretrained=False, nverts=nverts_var,faces=faces_var,shapeMU=shapeMU_var,shapePC=shapePC_var,num_classes=class_num, end2end=args.end2end)
    if args.arch == 'resnet50':
        model = resnet50(end2end=args.end2end)
    if args.arch == 'resnet101':
        pass
    model = torch.nn.DataParallel(model).cuda()

    criterion = BCE_sigmoid().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    
    if args.pretrained:
        print ('load model from pretrained ijba_res18_naive ')
        # checkpoint = torch.load('/home/victor_kai/AAAI_2020/CurriculumNet/curriculum_clustering/tests/RAFDB_DATA/extract_raf_db_code/model_best.pth.tar')
        checkpoint = torch.load('ijba_res18_naive.pth.tar')
        # pdb.set_trace()
        # checkpoint = torch.load('/data1/AAAI_Baseline_Train/baseline_models/web_emotion_model_best.pth.tar')
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        #pdb.set_trace()
        
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')|(key=='module.feature.weight')|(key=='module.feature.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)
        # pdb.set_trace()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

   
    print ('args.evaluate',args.evaluate)
    print (model)
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        #pdb.set_trace()
        prec1 = validate(val_loader, model, criterion)
        train(train_loader, model, criterion, optimizer, epoch)
    
        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)




### execute
if __name__ == '__main__':
    main()
