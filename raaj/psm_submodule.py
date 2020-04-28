from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, bn_running_avg = False):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad,
                                   dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes,
                                        track_running_stats = bn_running_avg))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes,
                                   kernel_size=kernel_size, padding=pad,
                                   stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

def convbn_3d_bias(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes,
                                   kernel_size=kernel_size, padding=pad,
                                   stride=stride,bias=True),
                         nn.BatchNorm3d(out_planes))

class ResConvBlock(torch.nn.Module):
    def __init__(self, In_D, Out_D, BN=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResConvBlock, self).__init__()

        if BN:
            self.a = nn.Sequential(
                convbn(In_D, Out_D, 3, 1, 1, 1, False), nn.ReLU(inplace=True)
            )
            self.b = nn.Sequential(
                convbn(Out_D, Out_D, 3, 1, 1, 1, False), nn.ReLU(inplace=True)
            )
            self.c = nn.Sequential(
                convbn(Out_D, Out_D, 3, 1, 1, 1, False), nn.ReLU(inplace=True)
            )
        else:
            self.a = nn.Sequential(
                nn.Conv2d(In_D, Out_D, 3, 1, 1),
                nn.PReLU(Out_D)
            )
            self.b = nn.Sequential(
                nn.Conv2d(Out_D, Out_D, 3, 1, 1),
                nn.PReLU(Out_D)
            )
            self.c = nn.Sequential(
                nn.Conv2d(Out_D, Out_D, 3, 1, 1),
                nn.PReLU(Out_D)
            )

    def forward(self, x):
        a_output = self.a(x)
        b_output = self.b(a_output)
        c_output = self.c(b_output)
        output = torch.cat([a_output, b_output, c_output], 1)
        return output

class ABlock3x3(torch.nn.Module):
    def __init__(self, In_D, Out_D, Depth=64, SubDepth=256, C=7, BN=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ABlock3x3, self).__init__()

        modules = [ResConvBlock(In_D, Depth, BN)]
        for i in range(0, C):
            modules.append(ResConvBlock(Depth * 3, Depth, BN))
        modules.append(nn.Conv2d(Depth * 3, SubDepth, 1, 1, 0))
        modules.append(nn.PReLU(SubDepth))
        modules.append(nn.Conv2d(SubDepth, Out_D, 1, 1, 0))

        self.net = nn.Sequential(
            *modules
        )
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class SimpleBlockA(torch.nn.Module):
    def __init__(self, In_D, Out_D, Depth=64, C=3, mode="default"):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleBlockA, self).__init__()
        import m_submodule

        modules = []
        if mode == "default":
            for i in range(0, C):
                modules.append(m_submodule.conv2d_leakyRelu(
                        ch_in = In_D, ch_out = Depth, kernel_size=3, stride=1, pad=1, use_bias=True) )
            modules.append(nn.Conv2d(Depth, Out_D, 3, 1, 1))

        self.net = nn.Sequential(
            *modules
        )
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, bn_running_avg = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation, bn_running_avg ),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, bn_running_avg)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  =\
                F.pad(torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right =\
                F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(\
                np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(),
                requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class feature_extraction(nn.Module):
    def __init__(self, feature_dim =32, bn_running_avg = False, multi_scale= False):
        ''' 
        inputs:
        multi_scale - if output multi-sclae features: 
        [1/4 scale of input image, 1/2 scale of input image] 
        '''

        super(feature_extraction, self).__init__()

        MUL = feature_dim/64.
        S0 = int(16*MUL)
        S1 = int(32*MUL)
        S2 = int(64*MUL)
        S3 = int(128*MUL)

        self.inplanes = S1
        self.multi_scale = multi_scale

        bn_ravg = bn_running_avg
        self.firstconv = nn.Sequential(convbn(3, S1, 3, 2, 1, 1, bn_ravg), nn.ReLU(inplace=True),
                                       convbn(S1, S1, 3, 1, 1, 1, bn_ravg), nn.ReLU(inplace=True),
                                       convbn(S1, S1, 3, 1, 1, 1, bn_ravg), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, S1, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, S2, S0, 2,1,1)

        #Chao: this is different from the dilation in the paper (2)
        self.layer3 = self._make_layer(BasicBlock, S3, 3, 1,1,1)

        #Chao: in the paper, this is 4
        self.layer4 = self._make_layer(BasicBlock, S3, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(S3, S1, 1, 1, 0, 1, bn_ravg),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(S3, S1, 1, 1, 0, 1, bn_ravg),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(S3, S1, 1, 1, 0, 1, bn_ravg),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(S3, S1, 1, 1, 0, 1, bn_ravg),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(S1*4 + S2 + S3, S3, 3, 1, 1, 1, bn_ravg),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(S3,
                                          feature_dim, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x) #
        output_layer1 = self.layer1(output)
        output_raw  = self.layer2(output_layer1)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, 
                (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]), mode='bilinear', align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1) 
        output_feature = self.lastconv(output_feature) 

        if self.multi_scale:
            return output_layer1, output_feature
        else:
            return output_feature 

