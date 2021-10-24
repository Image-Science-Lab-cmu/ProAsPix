import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from DenseResUnet import *
from pytorch_prototyping import Unet
from deblurring import *


# ==============================================
# The following are for network-2: Deconvolution
# ==============================================
class ConvNet1d(nn.Module):
    """ 1d convolutional neural network for Network-2. Deconvolution
    
    input: num_in_channels x 256
    output: 1 x num_outputs
    """
    def __init__(self, num_in_channels, num_outputs, num_feature_maps=64):
        super(ConvNet1d, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_outputs = num_outputs
        self.num_feature_maps = num_feature_maps

        # input channels x points:  1 x 256
        self.conv1 = nn.Conv1d(self.num_in_channels, num_feature_maps, kernel_size=11, stride=1, padding=5) # f x 256
        self.conv2 = nn.Conv1d(num_feature_maps, num_feature_maps, kernel_size=11, stride=1, padding=5) # f x 256
        self.maxpool2 = nn.MaxPool1d(2) # f x 128
        self.conv3 = nn.Conv1d(num_feature_maps, num_feature_maps, kernel_size=11, stride=1, padding=5) # f x 128
        self.maxpool3 = nn.MaxPool1d(2) # f x 64
        self.conv4 = nn.Conv1d(num_feature_maps, num_feature_maps * 2, kernel_size=11, stride=1, padding=5) # 2f x 64
        self.maxpool4 = nn.MaxPool1d(2) # 2f x 32
        self.conv5 = nn.Conv1d(num_feature_maps * 2, num_feature_maps * 2, kernel_size=11, stride=1, padding=5) # 2f x 32
        self.maxpool5 = nn.MaxPool1d(2) # 2f x 16
        self.conv6 = nn.Conv1d(num_feature_maps * 2, num_feature_maps * 2, kernel_size=11, stride=1, padding=5) # 2f x 16
        self.maxpool6 = nn.MaxPool1d(2) # 2f x 8
        self.fc1 = nn.Linear((num_feature_maps * 2) * 8, num_feature_maps * 4) # 4f
        self.fc2 = nn.Linear(num_feature_maps * 4, self.num_outputs) # num_outputs

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv5(x))
        x = self.maxpool5(x)
        x = F.relu(self.conv6(x))
        x = self.maxpool6(x)
        x = x.view(-1, (self.num_feature_maps * 2) * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
# ==============================================
# The following are for network-1: Demosaicing
# ==============================================
class DemosaicConvTrans(nn.Module):
    """ Upsampling network using ConvTranspose2d for Network-1. Demosaicing
    input: 16x16x256
    output: 256x256x256
    """
    def __init__(self):
        super(DemosaicConvTrans, self).__init__()
        layers = OrderedDict()
        # 2-4-1 for x2 and 4-4-0 for x4 upsampling
        layers[f'conv1'] = nn.ConvTranspose2d(256, 512, stride=4, kernel_size=4, padding=0) 
        layers[f'relu1'] = nn.ReLU()
        layers[f'conv2'] = nn.ConvTranspose2d(512, 512, stride=4, kernel_size=4, padding=0)
        layers[f'relu2'] = nn.ReLU()
        layers[f'conv3'] = nn.ConvTranspose2d(512, 512, stride=4, kernel_size=4, padding=0)
        layers[f'relu3'] = nn.ReLU()
        layers[f'conv4'] = nn.ConvTranspose2d(512, 256, stride=4, kernel_size=4, padding=0)
        self.demosaic = nn.Sequential(layers)

    def forward(self, x):
        return self.demosaic(x)


class DemosaicInterp(nn.Module):
    """ 2d convolutional network for Network-1. Demosaicing
    input: 4xBxB
    output: 1xBxB
    """
    def __init__(self):
        super(DemosaicInterp, self).__init__()
        layers = OrderedDict()
        layers[f'conv1'] = nn.Conv2d(4, 64, stride=1, kernel_size=3, padding=1) 
        layers[f'relu1'] = nn.ReLU()
        layers[f'conv2'] = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        layers[f'relu2'] = nn.ReLU()
        layers[f'conv3'] = nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1)
        layers[f'relu3'] = nn.ReLU()
        layers[f'conv4'] = nn.Conv2d(128, 256, stride=1, kernel_size=3, padding=1)
        layers[f'relu4'] = nn.ReLU()
        layers[f'conv5'] = nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1)
        layers[f'relu5'] = nn.ReLU()
        layers[f'conv6'] = nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1)
        layers[f'relu6'] = nn.ReLU()
        layers[f'conv7'] = nn.Conv2d(256, 256, stride=1, kernel_size=3, padding=1)
        layers[f'relu7'] = nn.ReLU()
        layers[f'conv8'] = nn.Conv2d(256, 128, stride=1, kernel_size=3, padding=1)
        layers[f'relu8'] = nn.ReLU()
        layers[f'conv9'] = nn.Conv2d(128, 1, stride=1, kernel_size=3, padding=1)
        self.demosaic = nn.Sequential(layers)

    def forward(self, x):
#         return x[:,0].unsqueeze(1) + self.demosaic(x)
        return self.demosaic(x)
    

class DemosaicDenseResUnet(nn.Module):
    def __init__(self, inchan=4, outchan=1, nchan=64):
        super(DemosaicDenseResUnet,self).__init__()

        self.in1 = InBlock(1, nchan//2)
        self.in2 = InBlock(3, nchan//2)

        self.down1 = DownBlock(nchan, 2*nchan, nresblocks=2, nreslayers=3)
        self.down2 = DownBlock(2*nchan, 4*nchan, nresblocks=2, nreslayers=3)
        self.down3 = DownBlock(4*nchan, 8*nchan, nresblocks=2, nreslayers=3)
        self.down4 = DownBlock(8*nchan, 16*nchan, nresblocks=2, nreslayers=3)

        self.mid = DenseResBlock(16*nchan,nreslayers=3)

        self.up4 = UpBlock(16*nchan, 8*nchan, nresblocks=2, nreslayers=3)
        self.up3 = UpBlock(8*nchan, 4*nchan, nresblocks=2, nreslayers=3)
        self.up2 = UpBlock(4*nchan, 2*nchan, nresblocks=2, nreslayers=3)
        self.up1 = UpBlock(2*nchan, nchan, nresblocks=2, nreslayers=3)

        self.out1 = OutBlock(nchan, 1)

    def forward(self, x):
#         print('x.shape', x.shape)
        y = torch.cat( (self.in1(x[:,0:1]), self.in2(x[:,1:4])), dim=1 )
        yadd1,y = self.down1(y)
        yadd2,y = self.down2(y)
        yadd3,y = self.down3(y)
        yadd4,y = self.down4(y)

        y = self.mid(y)

        y = self.up4(y,yadd4)
        y = self.up3(y,yadd3)
        y = self.up2(y,yadd2)
        y = self.up1(y,yadd1)

        y = self.out1(y)
#         print('y.shape', y.shape)
        return y
    #     return (torch.tanh(y)  + 1.0 ) / 2.0 #F.tanh is deprecated


class UnetAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetAtt, self).__init__()
        self.unet = Unet(in_channels=in_channels, out_channels=64, nf0=64,
                         num_down=3, max_channels=512, use_dropout=False,
                         outermost_linear=False)
        self.sa1 = Self_Attn(in_dim=64, activation='relu')
        self.conv1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = self.unet(x)
        x, _ = self.sa1(x)
        x = self.conv1(x)
        return x


class ParallelUnetAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelUnetAtt, self).__init__()
        self.unet1 = Unet(in_channels=1, out_channels=64, nf0=64,
                          num_down=4, max_channels=512, use_dropout=False,
                          outermost_linear=False)
        self.unet2 = Unet(in_channels=3, out_channels=64, nf0=64,
                          num_down=4, max_channels=512, use_dropout=False,
                          outermost_linear=False)
        self.conv1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        batch_size, C, width, height = x.shape
        x1 = self.unet1(x[:, 0:1])
        x2 = self.unet2(x[:, 1:])

        x = F.relu(x1 * x2)
        x = self.conv1(x)
        return x
#         for i in range(4):
#             x = torch.bmm(x1[:, i*4:i*4+16].view(batch_size, 16, width*height).permute(0, 2, 1),
#                           x2[:, i*4:i*4+16].view(batch_size, 16, width*height))
#             print('inter x', i, x.shape)
#             x = x.reshape(batch_size, 1, width, height)
#             print('inter x reshape', i, x.shape)
#             if i == 0:
#                 xcat = x
#             else:
#                 xcat = torch.cat((xcat, x), dim=1)
#         xcat = self.conv1(xcat)
#         print('xcat', xcat.shape)
#         return xcat


class ConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=512):
        super(ConvResNet, self).__init__()
        self.conv1  = nn.Conv2d(in_channels, num_channels, (5, 5), (1, 1), (2, 2))#, bias=False)
        
        self.res1 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 3)
        self.res2 = self.make_layer2(num_channels, num_channels)
        self.res3 = self.make_layer3(num_channels, num_channels)
        self.res4 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 3)

        self.fuse1 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse4 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        
        self.conv2  = nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1))#, bias=False)
        self.res5 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 1)      
        
        self.norm3 = nn.InstanceNorm2d(num_channels+3, affine=True)
#         self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d(num_channels+3, out_channels, (1, 1), (1, 1), (0, 0))#, bias=False)
        
        self.norm4 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv4  = nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x):
        # x (B, 132 or 4, 32, 32)
        # C = 512
        rgb = x[:, 1:4]  # (B, 3, 32, 32)
        
        x1 = self.conv1(x)  # (B, C, 32, 32)
        x1 = self.res1(x1)  # (B, C, 32, 32)
        x2 = self.res2(x1)  # (B, C, 32, 32)
        x3 = self.res3(x2)  # (B, C, 32, 32)
        x4 = self.res4(x3)  # (B, C, 32, 32)        

        x1 = self.fuse1(x1) # (B, C/4, 32, 32)
        x2 = self.fuse2(x2) # (B, C/4, 32, 32)
        x3 = self.fuse3(x3) # (B, C/4, 32, 32)
        x4 = self.fuse4(x4) # (B, C/4, 32, 32)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        
        out = self.conv2(x5)
        out = self.res5(out)
        
        out = torch.cat((out, rgb), dim=1)
        
        out = self.norm3(out)
        out = F.relu(out)        
        out = self.conv3(out)
        
        out = self.norm4(out)
        out = F.relu(out)        
        out = self.conv4(out)
        
        return out

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        return nn.Sequential(*layers)
    def make_layer3(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)


class ShuffleConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=144):
        super(ShuffleConvResNet, self).__init__()
        self.shuffle_factor = 4
        
        self.conv1  = nn.Conv2d(in_channels * self.shuffle_factor, num_channels, (5, 5), (1, 1), (2, 2))#, bias=False)

        self.pixel_shuffle = nn.PixelShuffle(self.shuffle_factor)

        self.res1 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 3)
        self.res2 = self.make_layer2(num_channels, num_channels)
        self.res3 = self.make_layer3(num_channels, num_channels)
        self.res4 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 3)

        self.fuse1 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)
        self.fuse4 = self.make_layer(TwoConvBlockIN, num_channels, num_channels//4, 1)

        self.conv2  = nn.Conv2d(num_channels,num_channels, (3, 3), (1, 1), (1, 1))#, bias=False)
        self.res5 = self.make_layer(FourConvResBlockIN, num_channels, num_channels, 1)
        
        self.norm3 = nn.InstanceNorm2d(num_channels, affine=True)
#         self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d(num_channels, out_channels, (1, 1), (1, 1), (0, 0))#, bias=False)
    
    def forward(self, Blurry):
        x = pixel_reshuffle(x, 4)

        x1 = self.conv1(x)
        x1 = self.res1(x1)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)

        x1 = self.fuse1(x1)
        x2 = self.fuse2(x2)
        x3 = self.fuse3(x3)
        x4 = self.fuse4(x4)

        x5 = self.pixel_shuffle(x1 + x2 + x3 + x4 + x)  

        out = x5
        out = self.conv2(out)
        out = self.res5(out)

        out = self.norm3(out)
        out = F.relu(out)
        out = self.conv3(out) + x

        return out

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        return nn.Sequential(*layers)
    def make_layer3(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)

