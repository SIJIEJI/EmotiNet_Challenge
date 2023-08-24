import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from torch.autograd import Variable
from torch.autograd.function import Function

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