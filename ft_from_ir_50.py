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
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from selfDefine import MsCelebDataset
import scipy.io as sio  
import numpy as np
import pdb
#from loss.loss_BCE import BCE_sigmoid_negtive_bias_all;
import random
# AU_num = 23;
# AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22];


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='IR_50', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='../../model/model/cfp_res50_naive.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='./ir_50_models', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print('end2end?:', args.end2end)
    train_list_file = 'label.txt'
    INPUT_SIZE = [112,112]
    RGB_MEAN = (0.5, 0.5, 0.5) # for normalize inputs to [-1, 1]
    RGB_STD = (0.5, 0.5, 0.5)
    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),
    ])
    val_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([112, 112]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    train_dataset =  MsCelebDataset(train_list_file, train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
   
    val_list_file = 'label_test.txt'
    val_dataset =  MsCelebDataset(val_list_file, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # prepare model
    model = None
    assert(args.arch in ['IR_50'])
    if args.arch == 'IR_50':
        print ('we use IR_50')
        model = IR_50([112,112])
    model = torch.nn.DataParallel(model).cuda()
    criterion = BCE_sigmoid().cuda()
    #criterion = BCE_sigmoid_negtive_bias_all().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    
    if args.pretrained:
        print ('load model from pretrained ijba_res18_naive ')
        # checkpoint = torch.load('/home/victor_kai/AAAI_2020/CurriculumNet/curriculum_clustering/tests/RAFDB_DATA/extract_raf_db_code/model_best.pth.tar')
        checkpoint = torch.load('backbone_ir50_ms1m_epoch63.pth')
        # pdb.set_trace()
        # checkpoint = torch.load('/data1/AAAI_Baseline_Train/baseline_models/web_emotion_model_best.pth.tar')
        pretrained_state_dict = checkpoint
        model_state_dict = model.state_dict()
        pdb.set_trace()
        
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict['module.' + key] = pretrained_state_dict[key]

        # pdb.set_trace()
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

def train(train_loader, model, criterion, optimizer, epoch):
    import pdb; pdb.set_trace()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    torch.autograd.set_detect_anomaly(True)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # pdb.set_trace()
        
        target_arr = np.array(target,dtype='int32').T
        target_tensor = torch.tensor(target_arr)
        target = target_tensor.cuda()
 

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        
        # compute output
        # try:
        pdb.set_trace()
        pred_score = model(input_var)
        # except:
        	# pdb.set_trace()
        #
        
        loss = criterion(pred_score, target_var)

        # measure accuracy and record loss
        pred_score[pred_score<=0.5] =0
        pred_score[pred_score>0.5] =1
        pred_score= pred_score.float()
        #loss = criterion(pred_score, target_var)
        target= target.gt(0)
        target= target.float()
        correct_num = sum(pred_score == target).sum()
        mean_acc = correct_num.float()/(target.size()[0]*23)

        # prec1, prec5 = accuracy(pred_score.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(mean_acc.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Data {data_time.val} ({data_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                                                              .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # pdb.set_trace()
        target_arr = np.array(target,dtype='int32').T
        target_tensor = torch.tensor(target_arr)
        target = target_tensor.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output

        pred_score = model(input_var)
        # pdb.set_trace()

        
        # pdb.set_trace()
        loss = criterion(pred_score, target_var)
        # measure accuracy and record loss
        pred_score[pred_score<=0.5] =0
        pred_score[pred_score>0.5] =1
        pred_score= pred_score.float()
        
        target = target.gt(0)
        target= target.float()
        correct_num = sum(pred_score == target).sum()
        mean_acc = correct_num.float()/(target.size()[0]*23)

        # prec1, prec5 = accuracy(pred_score.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(mean_acc.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Loss {loss.val} ({loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, 
                   top1=top1))

    print(' * Prec@1 {top1.avg}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class BCE_sigmoid(nn.Module):
    def __init__(self, size_average = False):
        super(BCE_sigmoid, self).__init__()
        self.size_average = size_average;

    def forward(self, x, labels):
        N = x.size(0);
        # pdb.set_trace()
        # mask1 = labels.eq(0);

        # mask = 1 - mask1.float();
        mask2 = labels.eq(999)
        weights = 1- mask2.float()

        target = labels.gt(0);
        target = target.float();
        # pdb.set_trace()
        #import pdb; pdb.set_trace()
        #loss = F.binary_cross_entropy_with_logits(F.sigmoid(x), target, weights)
        loss = F.binary_cross_entropy_with_logits(x, target, weights)
        if self.size_average:
            loss = loss/N;

        return loss

class BCE_sigmoid_negtive_bias_all(nn.Module):
    def __init__(self, size_average = False, AU_num = 23, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]):
        super(BCE_sigmoid_negtive_bias_all, self).__init__()
        self.size_average = size_average;
        self.AU_num = AU_num;
        #import pdb; pdb.set_trace()
        self.AU_idx = AU_idx;
        self.boundary = 1;

        ##### balance weights for different databases
        #self.weight = [0.2, 0.3, 0.2, 0.2, 0.5, 0.2, 0.5, 0.2, 0.1, 0.5, 0.2, 0.3,0.2, 0.3, 0.2, 0.2, 0.5, 0.2, 0.5, 0.2, 0.1, 0.5, 0.2];
        #self.weight = [0.0012597430655963838, 0.0004919313290455535, 0.0021106513104319356, 0.0007678117365508301, 0.004719881670572202, 0.000372272357115554, 0.029090425620315438, 0.010056339432617042, 0.0034817436971298467, 0.0003057951504877765, 0.003995280118329428, 8.808229878180519e-05, 0.012070598793438699, 0.016788818533845208, 0.0017832510677901316, 0.0008758371973209686, 0.0005933090691529143, 0.0031992155689617922, 0.003212511010287348, 0.0016685778863572154, 0.0009356666832859684, 0.0010985358395240233, 0.00103372056306194]
        self.weight = [0.0012597430655963838, 0.0004919313290455535, 0.0021106513104319356, 0.0007678117365508301, 0.004719881670572202, 0.000372272357115554, 0.029090425620315438, 0.010056339432617042, 0.0034817436971298467, 0.0003057951504877765, 0.003995280118329428, 8.808229878180519e-05, 0.012070598793438699, 0.016788818533845208, 0.0017832510677901316, 0.0008758371973209686, 0.0005933090691529143, 0.0031992155689617922, 0.003212511010287348, 0.0016685778863572154, 0.0009356666832859684, 0.0010985358395240233, 0.00103372056306194]
        #self.weight = [0.012597430655963837, 0.004919313290455535, 0.021106513104319356, 0.007678117365508301, 0.04719881670572202, 0.00372272357115554, 0.29090425620315435, 0.10056339432617041, 0.034817436971298465, 0.0030579515048777648, 0.03995280118329428, 0.0008808229878180518, 0.12070598793438699, 0.16788818533845207, 0.017832510677901314, 0.008758371973209686, 0.005933090691529142, 0.03199215568961792, 0.03212511010287348, 0.016685778863572153, 0.009356666832859684, 0.010985358395240234, 0.0103372056306194]
        #self.weight_0 = 1/ (np.array(self.weightw) + 1)
        #self.weight_1 = 1 - self.weight_0

        ##### balance weights for different databases
        # if database == 0:
        #     self.weight = self.weight_0
        # elif database == 1:
        #     self.weight = self.weight_1


        self.balance_a = [];
        for i in range(0,self.AU_num):
            self.balance_a.append(self.weight[self.AU_idx[i]]);

    def forward(self, x, labels):
        N = x.size(0);
        #import pdb; pdb.set_trace()

        mask1 = labels.eq(999);
        mask = 1 - mask1.float();

        ## selective learning balance
        ################################################################
        for i in range(0,self.AU_num):
            temp = labels[:,i];  # 对列来的
            zero_num = torch.sum(temp.eq(999));
            pos_num = torch.sum(temp.eq(1));
            neg_num = torch.sum(temp.eq(0));
            zero_num = zero_num.float();
            pos_num = pos_num.float();
            neg_num = neg_num.float();
            half_num = (N - zero_num)*self.balance_a[i];  # weights of zero number

            if (pos_num.item() <  half_num.item()):
                idx = torch.nonzero(temp.eq(0));

                sample_num = int(neg_num.item() - math.ceil(half_num.item()));

                if sample_num < 1:
                    continue;
                # import pdb; pdb.set_trace()

                # zero_idx = random.sample(idx, sample_num);
                zero_idx = random.sample(list(idx), sample_num)
                for j in range(0, len(zero_idx)):
                    mask[int(zero_idx[j]), i] = 0;

                ### postive under-representation
                if pos_num.item() != 0:
                    ratio = half_num/pos_num;
                    if ratio.item() > self.boundary:
                        ratio = self.boundary;

                    idx = torch.nonzero(temp.eq(1));

                    for j in range(0, len(idx)):
                        mask[int(idx[j].data), i] = ratio;
        ################################################################
        # import pdb; pdb.set_trace()

        target = labels.gt(0);
        target = target.float();
        #import pdb; pdb.set_trace()

        loss = F.binary_cross_entropy_with_logits(x, target, mask, size_average = False);

        if self.size_average:
            loss = loss/N;

        return loss


if __name__ == '__main__':
    main()
