import math
import numpy as np
import random
import scipy.io as sio
import torch

from scipy.interpolate import interp1d
from skimage import color

import torch
import torch.nn as nn
import torch.utils.data as data
from datasets import RestoreDataTestAssortedSingleScene, RestoreDataTestAssortedSingleSceneSinglePattern
from pytorch_prototyping import Unet, UnetSiamese, ResUnet
from bilateral import interp_bilateral
from preprocess import ScaleY
from ResDenseNet import ResDenseNet


def restore_net(data_pthfile, restore_saved_model, pattern_idx, patch_size, batch_size, pos_enc, use_guide_image, device=torch.device('cuda:0')):
    image_dataset = RestoreDataTestAssortedSingleScene(pthfile=data_pthfile,
                                           patch_size=patch_size,
                                           pos_enc=pos_enc,
                                           use_guide_image=use_guide_image,
                                           transform=ScaleY(1.0/65535.0),
                                           pattern_idx=pattern_idx)

    image_dataloader = data.DataLoader(image_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)

    num_patterns = image_dataset.num_patterns
    num_rows = image_dataset.num_rows
    num_cols = image_dataset.num_cols

    # Define and load model if only model name is given
    if isinstance(restore_saved_model, str):
        # Define model
        in_channels = 1
        if pos_enc['enabled']:
            in_channels += pos_enc['len'] * (1 + 2 * int(pos_enc['nd']))
        if use_guide_image:
            in_channels += 3
        if 'ResUnet' in restore_saved_model:
            restore_model = ResUnet(in_channels=in_channels,
                            out_channels=1,
                            nf0=192,
                            num_down=4,
                            max_channels=768,
                            use_dropout=False,
                            outermost_linear=True)
        elif 'ResDenseNet20' in restore_saved_model:
            restore_model = ResDenseNet(num_rdbs=20, num_convs=6, in_channels=in_channels, rdb_channels=64, inter_channels=256, out_channels=1)
        elif 'ResDenseNet' in restore_saved_model:
            restore_model = ResDenseNet(num_rdbs=6, num_convs=6, in_channels=in_channels, rdb_channels=64, inter_channels=256, out_channels=1)
        elif 'Unet192' in restore_saved_model:
            restore_model = Unet(in_channels=in_channels,
                            out_channels=1,
                            nf0=192,
                            num_down=4,
                            max_channels=768,
                            use_dropout=False,
                            outermost_linear=True)
        else:
            restore_model = Unet(in_channels=in_channels,
                                out_channels=1,
                                nf0=64,
                                num_down=4,
                                max_channels=512,
                                use_dropout=False,
                                outermost_linear=True)
        device = torch.device('cuda:0')
        restore_model = restore_model.to(device)

        checkpoint = torch.load(restore_saved_model)
        restore_model.load_state_dict(checkpoint['state_dict'])

        if torch.cuda.device_count() > 1:
            restore_model = nn.DataParallel(restore_model)
    else:
        restore_model = restore_saved_model

    # Prediction
    assorted_meas = torch.zeros(num_patterns, num_rows, num_cols)
    assorted_sim = torch.zeros(num_patterns, num_rows, num_cols)
    target_image = torch.zeros(num_patterns, num_rows, num_cols)

    restore_model.eval()
    with torch.no_grad():
        for i, sample in enumerate(image_dataloader):
            coord, input, target = sample

            input = input.to(device)
            target = target.to(device)

            output = restore_model(input)

            # coord is a list of 1d tensors of the following form:
            # [tensor(pattern indices for the batch),
            #  tensor(rows for the batch), tensor(cols for the batch),
            #  tensor(heights for the batch), tensor(widths for the batch)]

            # Go through each batch item
            for j in range(coord[0].shape[0]):
                pattern_idx = coord[0][j].item()
                ii = coord[1][j].item()
                jj = coord[2][j].item()
                hh = coord[3][j].item()
                ww = coord[4][j].item()

                assorted_meas[pattern_idx, ii:ii+hh, jj:jj+ww] = input[j][0].clone().detach().cpu()
                assorted_sim[pattern_idx, ii:ii+hh, jj:jj+ww] = output[j].clone().detach().cpu() 
                target_image[pattern_idx, ii:ii+hh, jj:jj+ww] = target[j].clone().detach().cpu()

    # batch_size=1 takes 1min for all 13 patterns
    return image_dataset, assorted_meas, assorted_sim, target_image



def restore_net_full_size(data_pthfile, restore_saved_model, pattern_idx, patch_size, batch_size, pos_enc, use_guide_image, device=torch.device('cuda:0')):
    image_dataset = RestoreDataTestAssortedSingleScene(pthfile=data_pthfile,
                                           patch_size=patch_size,
                                           pos_enc=pos_enc,
                                           use_guide_image=use_guide_image,
                                           transform=ScaleY(1.0/65535.0),
                                           pattern_idx=pattern_idx)

    image_dataloader = data.DataLoader(image_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)

    num_patterns = image_dataset.num_patterns
    num_rows = image_dataset.num_rows
    num_cols = image_dataset.num_cols

    # Define and load model if only model name is given
    if isinstance(restore_saved_model, str):
        # Define model
        in_channels = 1
        if pos_enc['enabled']:
            in_channels += pos_enc['len'] * (1 + 2 * int(pos_enc['nd']))
        if use_guide_image:
            in_channels += 3
        if 'Unet192' in restore_saved_model:
            restore_model = Unet(in_channels=in_channels,
                            out_channels=1,
                            nf0=192,
                            num_down=4,
                            max_channels=768,
                            use_dropout=False,
                            outermost_linear=True)
        else:
            restore_model = Unet(in_channels=in_channels,
                                out_channels=1,
                                nf0=64,
                                num_down=4,
                                max_channels=512,
                                use_dropout=False,
                                outermost_linear=True)
        device = torch.device('cuda:0')
        restore_model = restore_model.to(device)

        checkpoint = torch.load(restore_saved_model)
        restore_model.load_state_dict(checkpoint['state_dict'])

        if torch.cuda.device_count() > 1:
            restore_model = nn.DataParallel(restore_model)
    else:
        restore_model = restore_saved_model

    # Prediction
    num_patches = 2
    assorted_meas = torch.zeros(num_patches, num_patterns, num_rows, num_cols)
    assorted_sim = torch.zeros(num_patches, num_patterns, num_rows, num_cols)
    target_image = torch.zeros(num_patches, num_patterns, num_rows, num_cols)
    print('len(image_dataloader): ', len(image_dataloader))
    restore_model.eval()
    with torch.no_grad():
        for i, sample in enumerate(image_dataloader):
            coord, input, target = sample

            input = input.to(device)
            target = target.to(device)

            output = restore_model(input)

            # coord is a list of 1d tensors of the following form:
            # [tensor(pattern indices for the batch),
            #  tensor(rows for the batch), tensor(cols for the batch),
            #  tensor(heights for the batch), tensor(widths for the batch)]
            print(coord)

            # Go through each batch item
            for j in range(coord[0].shape[0]):
                pattern_idx = coord[0][j].item()
                ii = coord[1][j].item()
                jj = coord[2][j].item()
                hh = coord[3][j].item()
                ww = coord[4][j].item()
                print('i, j, pattern_idx, ii, jj, hh, ww', i, j, pattern_idx, ii, jj, hh, ww)

                # assorted_meas[pattern_idx, ii:ii+hh, jj:jj+ww] = input[j][0].clone().detach().cpu()
                # assorted_sim[pattern_idx, ii:ii+hh, jj:jj+ww] = output[j].clone().detach().cpu() 
                # target_image[pattern_idx, ii:ii+hh, jj:jj+ww] = target[j].clone().detach().cpu()

    # batch_size=1 takes 1min for all 13 patterns
    return image_dataset, assorted_meas, assorted_sim, target_image


def restore_net_singlepattern(data_pthfile, restore_saved_model, pattern_idx, patch_size, pattern_size, batch_size, pos_enc, use_guide_image):
    image_dataset = RestoreDataTestAssortedSingleSceneSinglePattern(pthfile=data_pthfile,
                                           patch_size=patch_size,
                                           pattern_size=pattern_size,
                                           pattern_idx=pattern_idx,
                                           pos_enc=pos_enc,
                                           use_guide_image=use_guide_image,
                                           transform=ScaleY(1.0/65535.0)
                                           )

    image_dataloader = data.DataLoader(image_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=False)

    num_patterns = image_dataset.num_patterns
    num_rows = image_dataset.num_rows
    num_cols = image_dataset.num_cols

    # Define model
    in_channels = 1
    if pos_enc['enabled']:
        in_channels += pos_enc['len'] * (0 + 2 * int(pos_enc['nd']))
    if use_guide_image:
        in_channels += 3
#     restore_model = Unet(in_channels=in_channels,
#                           out_channels=1,
#                           nf0=64,
#                           num_down=2,
#                           max_channels=512,
#                           use_dropout=False,
#                           outermost_linear=True)
    # for 16x16
    if patch_size[0] == 16:
        restore_model = Unet(in_channels=in_channels,
                              out_channels=1,
                              nf0=256,
                              num_down=2,
                              max_channels=512,
                              use_dropout=False,
                              outermost_linear=True)
    # for 32x32 and 64x64
    if patch_size[0] > 16:
        restore_model = Unet(in_channels=in_channels,
                              out_channels=1,
                              nf0=64,
                              num_down=4,
                              max_channels=512,
                              use_dropout=False,
                              outermost_linear=True)

    device = torch.device('cuda:0')
    restore_model = restore_model.to(device)

    checkpoint = torch.load(restore_saved_model)
    restore_model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.device_count() > 1:
        restore_model = nn.DataParallel(restore_model)

    # Prediction
    assorted_meas = torch.zeros(num_patterns, num_rows, num_cols)
    assorted_sim = torch.zeros(num_patterns, num_rows, num_cols)
    valid_mask = torch.zeros(num_patterns, num_rows, num_cols)
    sum_mask = torch.zeros(num_patterns, num_rows, num_cols)
    target_image = torch.zeros(num_patterns, num_rows, num_cols)

    restore_model.eval()
    with torch.no_grad():
        for i, sample in enumerate(image_dataloader):
            coord, input, target = sample

            input = input.to(device)
            target = target.to(device)

            output = restore_model(input)

            # coord is a list of 1d tensors of the following form:
            # [tensor(pattern indices for the batch),
            #  tensor(rows for the batch), tensor(cols for the batch),
            #  tensor(heights for the batch), tensor(widths for the batch)]

            # Go through each batch item
            for j in range(coord[0].shape[0]):
                pattern_idx = coord[0][j].item()
                ii = coord[1][j].item()
                jj = coord[2][j].item()
                hh = coord[3][j].item()
                ww = coord[4][j].item()

                assorted_meas[pattern_idx, ii:ii+hh, jj:jj+ww] += input[j][0].clone().detach().cpu()
                assorted_sim[pattern_idx, ii:ii+hh, jj:jj+ww] += output[j][0].clone().detach().cpu() 
                target_image[pattern_idx, ii:ii+hh, jj:jj+ww] += target[j][0].clone().detach().cpu()
                sum_mask[pattern_idx, ii:ii+hh, jj:jj+ww] += 1.0
                valid_mask[pattern_idx, ii:ii+hh, jj:jj+ww] = 1.0
    
    assorted_meas = assorted_meas / sum_mask
    assorted_sim = assorted_sim / sum_mask
    target_image = target_image / sum_mask
    assorted_meas[~torch.isfinite(assorted_meas)] = 0.0
    assorted_sim[~torch.isfinite(assorted_sim)] = 0.0
    target_image[~torch.isfinite(target_image)] = 0.0

    return image_dataset, assorted_meas, assorted_sim, target_image, valid_mask

