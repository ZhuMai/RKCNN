import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init

import math
from .utils import transition, global_pool, norm

class RK_block(nn.Module):
    def __init__(self, input_channels, growth_rate, s, update_num, keep_prob, if_b, wk, replace):
        super(RK_block, self).__init__()

        self.update_num = update_num
        self.keep_prob = keep_prob
        self.if_b = if_b
        self.s = s
        self.replace = replace

        if if_b:
            self.init_bn_1x1 = nn.ModuleList([norm(input_channels + i * growth_rate) for i in range(self.s)])
            self.init_bn_3x3 = nn.ModuleList([norm(wk) for i in range(self.s)])
            self.conv_1x1 = nn.ModuleList([nn.Conv2d(input_channels + i * growth_rate, wk, kernel_size = 1, stride = 1, padding = 0, bias = False)
                                            for i in range(self.s)])
            self.conv_3x3 = nn.ModuleList([nn.Conv2d(wk, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False)
                                            for i in range(self.s)])

            if update_num > 0:
                self.update_bn_1x1 = nn.ModuleList([norm(input_channels * s)
                                        for i in range(s)])
                self.update_bn_3x3 = nn.ModuleList([norm(wk) for i in range(s)])
                self.conv_1x1_update = nn.ModuleList([nn.Conv2d(input_channels * s, wk, kernel_size = 1, stride = 1, padding = 0, bias = False)
                                                    for i in range(s)])
                self.conv_3x3_update = nn.ModuleList([nn.Conv2d(wk, input_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
                                                    for i in range(s)])
        else:
            print("Only bottelneck mode is supportted now.")


    def forward(self, x):
        # key: 1, 2, 3, 4, 5, update every update
        self.blob_final = []

        # init
        bottom_blob = x
        count = 0
        for j in range(self.s):
            if self.if_b:
                next_layer = self.init_bn_1x1[j](bottom_blob)
                next_layer = F.relu(next_layer, inplace=True)
                # conv 1 x 1
                next_layer = self.conv_1x1[j](next_layer)
                if self.keep_prob < 1:
                    next_layer = F.dropout(next_layer, 1 - self.keep_prob, self.training)
                # conv 3 x 3
                next_layer = self.init_bn_3x3[j](next_layer)
                next_layer = F.relu(next_layer, inplace=True)
                next_layer = self.conv_3x3[j](next_layer)
                if self.keep_prob < 1:
                    next_layer = F.dropout(next_layer, 1 - self.keep_prob, self.training)
            else:
                print("Only bottelneck mode is supportted now.")

            bottom_blob = torch.cat((bottom_blob, next_layer), 1)

            self.blob_final.append(next_layer)

        # update
        if self.replace:
            for update_id in range(self.update_num):
                for stage_id in range(self.s):
                    bottom_blobs = x
                    if self.if_b:
                        for bottom_id in range(self.s):
                            if bottom_id != stage_id:
                                bottom_blobs = torch.cat((bottom_blobs, self.blob_final[bottom_id]), 1)
                        bottom_blobs = self.update_bn_1x1[stage_id](bottom_blobs)
                        bottom_blobs = F.relu(bottom_blobs, inplace=True)
                        # conv 1 x 1
                        mid_blobs = self.conv_1x1_update[stage_id](bottom_blobs)
                        if self.keep_prob < 1:
                            mid_blobs = F.dropout(mid_blobs, 1 - self.keep_prob, self.training)
                        # conv 3 x 3
                        top_blob = self.update_bn_3x3[stage_id](mid_blobs)
                        top_blob = F.relu(top_blob, inplace=True)
                        top_blob = self.conv_3x3_update[stage_id](top_blob)
                        if self.keep_prob < 1:
                            top_blob = F.dropout(top_blob, 1 - self.keep_prob, self.training)
                    else:
                        print("Only bottelneck mode is supportted now.")

                    self.blob_final[stage_id]=top_blob
        else:
            for update_id in range(self.update_num):
                self.blob_middle = self.blob_final
                self.blob_final = []

                for stage_id in range(self.s):
                    bottom_blobs = x
                    if self.if_b:
                        for bottom_id in range(self.s):
                            if bottom_id != stage_id:
                                bottom_blobs = torch.cat((bottom_blobs, self.blob_middle[bottom_id]), 1)
                        bottom_blobs = self.update_bn_1x1[stage_id](bottom_blobs)
                        bottom_blobs = F.relu(bottom_blobs, inplace=True)
                        # conv 1 x 1
                        mid_blobs = self.conv_1x1_update[stage_id](bottom_blobs)
                        if self.keep_prob < 1:
                            mid_blobs = F.dropout(mid_blobs, 1 - self.keep_prob, self.training)
                        # conv 3 x 3
                        top_blob = self.update_bn_3x3[stage_id](mid_blobs)
                        top_blob = F.relu(top_blob, inplace=True)
                        top_blob = self.conv_3x3_update[stage_id](top_blob)
                        if self.keep_prob < 1:
                            top_blob = F.dropout(top_blob, 1 - self.keep_prob, self.training)
                    else:
                        print("Only bottelneck mode is supportted now.")

                    self.blob_final.append(top_blob)

        # output
        # X0 + increment
        output_state = x
        #torch.utils.backcompat.broadcast_warning.enabled=True
        for i in range(self.s):
            output_state = torch.add(input=output_state, other=self.blob_final[i])

        return output_state

class build_RKCNN(nn.Module):
    def __init__(self, growth_rates, stages, if_att, update_nums, keep_prob, out_features, if_b, steps, neck, multiscale, replace=False):
        super(build_RKCNN, self).__init__()
        self.period_num = len(growth_rates)

        self.fir_trans = nn.Conv2d(3, growth_rates[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.if_att = if_att
        self.list_block = nn.ModuleList()
        self.list_trans = nn.ModuleList()
        self.list_gb = nn.ModuleList()
        gb_channel = 0

        self.steps = steps
        self.multiscale = multiscale

        for i in range(self.period_num):
            for j in range(self.steps[i]):
                self.list_block.append(RK_block(input_channels=growth_rates[i], growth_rate=growth_rates[i], s=stages[i], update_num=update_nums[i], keep_prob=keep_prob, if_b=if_b, wk=neck*growth_rates[i], replace=replace))
            if self.multiscale or (self.period_num-1 == i):
                gb_channel = gb_channel + growth_rates[i]

            if i < self.period_num - 1:
                self.list_trans.append(transition(self.if_att, current_size=None, input_channels=growth_rates[i], out_channels=growth_rates[i+1], keep_prob=keep_prob))

            if self.multiscale or (self.period_num-1 == i):
                self.list_gb.append(global_pool(input_size=None, input_channels=growth_rates[i]))

        self.fc = nn.Linear(in_features=gb_channel, out_features=out_features)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                mod.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)

        feature_I_list = []
        step_num = 0

        for i in range(self.period_num):
            for j in range(self.steps[i]):
                output = self.list_block[step_num](output)
                step_num = step_num + 1
            block_feature_I = output
            block_feature_II = output

            if self.multiscale:
                feature_I_list.append(self.list_gb[i](block_feature_I))
            elif self.period_num-1 == i:
                feature_I_list.append(self.list_gb[0](block_feature_I))
            if i < self.period_num - 1:
                output = self.list_trans[i](block_feature_II)


        final_feature = feature_I_list[0]
        for block_id in range(1, len(feature_I_list)):
            final_feature=torch.cat((final_feature, feature_I_list[block_id]), 1)

        final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        output = self.fc(final_feature)
        return output
