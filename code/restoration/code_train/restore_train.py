""" restore_train.py

Author: Vijay Rengarajan
"""

import math
import os
import sys
import time

from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../functions/')
from datasets import RestoreAssortedChannel
from pytorch_prototyping import Unet
from helpers import reset_random, print_gpu_usage
from preprocess import *
from disc import Discriminator

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

GPU_DEBUG = False

#--------------------
# Random seeds for training and testing
#--------------------
random_seed = 2022
reset_random(random_seed)

#--------------------
# Training parameters
#--------------------
save_name = 'restore_0625_data4b_r1_pm1_64x64_Assort_PosEncND_Unet192_1024x1024_NoGuide_rand2022'
train_pthfile = '../data/restore/0625_data4b_train_1024x1024.pth'
val_pthfile = '../data/restore/0625_data4b_val_1024x1024.pth'
patch_size = (64, 64)


# train
num_patches_train = 24_000
rand_patches_flag_train = True
scene_idx_train = -1
pattern_idx_train = -1
filter_idx_train = -1
# >=0 is specific index, -1 is random, -2 is random but fixed during init

# val
num_patches_val = 6_000
rand_patches_flag_val = False
scene_idx_val = -2
pattern_idx_val = -2
filter_idx_val = -2
# >=0 is specific index, -1 is random, -2 is random but fixed during init

pos_enc = {'enabled': True, 'len': 64, 'min_freq': 1e-4, 'nd': True} # this gives 3*64 channels
use_guide_image = False
in_channels = 1 + 0 + 3*64 # assort_meas, guide_image, pos_enc_nd
out_channels = 1

batch_size = 150
num_workers = 8
lr = 1e-3
lr_disc = 1e-4
adv_wt = 1e-4 if 'GAN' in save_name else 0
num_epochs = 1000
print_every_nth_epoch = 1
save_every_epoch = False
load_model_path = None

savemodel_basedir = '../saved_models/'
tensorboard_basedir = '../runs/'

#--------------------
# Define dataset and dataloader
#--------------------
train_transform = transforms.Compose([
                                      ScaleY(1.0/65535)
                                     ])
val_transform = transforms.Compose([
                                   ScaleY(1.0/65535)
                                   ])

train_dataset = RestoreAssortedChannel(train_pthfile, 
                                        patch_size=patch_size, 
                                        num_patches=num_patches_train, 
                                        rand_patches_flag=rand_patches_flag_train, 
                                        scene_idx=scene_idx_train,
                                        pattern_idx=pattern_idx_train, 
                                        pos_enc=pos_enc, 
                                        use_guide_image=use_guide_image, 
                                        transform=train_transform)
val_dataset = RestoreAssortedChannel(val_pthfile, 
                                      patch_size=patch_size, 
                                      num_patches=num_patches_val, 
                                      rand_patches_flag=rand_patches_flag_val, 
                                      scene_idx=scene_idx_val,
                                      pattern_idx=pattern_idx_val, 
                                      pos_enc=pos_enc, 
                                      use_guide_image=use_guide_image, 
                                      transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True, drop_last=True)

#--------------------
# Save model information
#--------------------
save_dir = os.path.join(savemodel_basedir, save_name)
Path(save_dir).mkdir(parents=True, exist_ok=True)

#--------------------
# Define device
#--------------------
device = torch.device('cuda:0')

#--------------------
# Define model
#--------------------
model = Unet(in_channels=in_channels, out_channels=out_channels, nf0=192, num_down=4, max_channels=768, use_dropout=False, outermost_linear=True)

# Load model if given
if (not load_model_path) or (load_model_path.lower() == 'none'):
    start_epoch = 1
else:
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
model = model.to(device)

# For multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

#--------------------
# Define criterion
#--------------------
criterion = nn.MSELoss()
criterion = criterion.to(device)

#--------------------
# Define optimizer
#--------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

# Load model if given
if (not load_model_path) or (load_model_path.lower() == 'none'):
    start_epoch = 1
else:
    checkpoint = torch.load(load_model_path)
    checkpoint['optimizer']['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1

#--------------------
# Discriminator
#--------------------
if adv_wt > 0:
    netD = Discriminator(inchan=1)
    netD = netD.to(device)

    # For multi-GPU training
    if torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_disc)
    adv_criterion = nn.BCELoss()
    adv_criterion = adv_criterion.to(device)

#--------------------
# Tensorboard
#--------------------
tensorboard_dir = os.path.join(tensorboard_basedir, save_name)
writer = SummaryWriter(log_dir=tensorboard_dir)

#--------------------
# Min and max of dataset
#--------------------
if False:
    print_min_max_dataset(train_dataloader)

#--------------------
# Save checkpoint
#--------------------
def save_checkpoint(epoch, save_pthfile):
    '''
    Save model checkpoint
    '''
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler
    }
    torch.save(state, save_pthfile)
    del state

#--------------------
# Training loop
#--------------------
min_loss = float('inf') # Best validation loss tracker
start_time = perf_counter()
print(f'Training started at {time.asctime(time.localtime())}')

# Go through each epoch
num_iter = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    epoch_time = perf_counter()
    onlyG = True if epoch == 1 else False

    # Training minibatch loop
    model.train()
    train_loss_epoch = 0.0
    train_disc_loss_epoch = 0.0
    for i, sample in enumerate(train_dataloader):
        input, target = sample[0], sample[1]

        input = input.to(device)
        target = target.to(device)

        if onlyG == False and adv_wt > 0:
            # ====================
            # Train discriminator
            # ====================
            output1 = model(input.clone())
            
            target_real = torch.rand((batch_size,1), requires_grad=False, device=target.device)*0.5 + 0.7
            target_fake = torch.rand((batch_size,1), requires_grad=False, device=output1.device)*0.2

            target_real = target_real.to(device)
            target_fake = target_fake.to(device)

            netD.zero_grad()
            optimizerD.zero_grad()

            netD_loss = adv_criterion(netD(target), target_real) + adv_criterion(netD(output1), target_fake)
            netD_loss.backward(retain_graph=True)
            optimizerD.step()

            train_disc_loss_epoch += netD_loss.item() / len(train_dataloader)
        else:
            netD_loss = 0.0

        output = model(input)

        train_loss = criterion(output, target)
        train_loss_epoch += train_loss.item() / len(train_dataloader)
        ones_const = torch.ones((batch_size, 1), requires_grad=False, device=output.device)

        if adv_wt > 0 and onlyG == False:
            adv_loss = adv_wt * adv_criterion(netD(output), ones_const)
        else:
            adv_loss = 0.0

        gen_loss = train_loss + adv_loss
        
        writer.add_scalar('IterLoss/train', train_loss, num_iter)

        # Backprop for each bin
        model.zero_grad()
        gen_loss.backward()
        optimizer.step()
        num_iter = num_iter + 1

    if use_guide_image:
        writer.add_image('Image/train', torch.cat( (
                                                torch.cat((input[0,0:1], input[0,1:2], output[0,0:1], target[0,0:1]), dim=2),
                                                torch.cat((input[1,0:1], input[1,1:2], output[1,0:1], target[1,0:1]), dim=2),
                                                torch.cat((input[2,0:1], input[2,1:2], output[2,0:1], target[2,0:1]), dim=2),
                                                torch.cat((input[3,0:1], input[3,1:2], output[3,0:1], target[3,0:1]), dim=2),
                                                torch.cat((input[4,0:1], input[4,1:2], output[4,0:1], target[4,0:1]), dim=2)
                                                ),
                                              dim=1), epoch)
    else:
        writer.add_image('Image/train', torch.cat( (
                                                torch.cat((input[0,0:1], output[0,0:1], target[0,0:1]), dim=2),
                                                torch.cat((input[1,0:1], output[1,0:1], target[1,0:1]), dim=2),
                                                torch.cat((input[2,0:1], output[2,0:1], target[2,0:1]), dim=2),
                                                torch.cat((input[3,0:1], output[3,0:1], target[3,0:1]), dim=2),
                                                torch.cat((input[4,0:1], output[4,0:1], target[4,0:1]), dim=2)
                                                ),
                                              dim=1), epoch)
    print_gpu_usage(f'After training epoch {epoch}', device, GPU_DEBUG)

    # Scheduler for epoch, not bin
    if scheduler is not None:
        scheduler.step()

    # Validation minibatch loop
    model.eval()   
    with torch.no_grad():
        val_loss_epoch = 0.0
        for i, sample in enumerate(val_dataloader):
            input, target = sample[0], sample[1]
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            val_loss = criterion(output, target)
            val_loss_epoch += val_loss.item() / len(val_dataloader)

    if use_guide_image:
        writer.add_image('Image/val', torch.cat( (
                                                torch.cat((input[0,0:1], input[0,1:2], output[0,0:1], target[0,0:1]), dim=2),
                                                torch.cat((input[1,0:1], input[1,1:2], output[1,0:1], target[1,0:1]), dim=2),
                                                torch.cat((input[2,0:1], input[2,1:2], output[2,0:1], target[2,0:1]), dim=2),
                                                torch.cat((input[3,0:1], input[3,1:2], output[3,0:1], target[3,0:1]), dim=2),
                                                torch.cat((input[4,0:1], input[4,1:2], output[4,0:1], target[4,0:1]), dim=2)
                                                ),
                                              dim=1), epoch)
    else:
        writer.add_image('Image/val', torch.cat( (
                                                torch.cat((input[0,0:1], output[0,0:1], target[0,0:1]), dim=2),
                                                torch.cat((input[1,0:1], output[1,0:1], target[1,0:1]), dim=2),
                                                torch.cat((input[2,0:1], output[2,0:1], target[2,0:1]), dim=2),
                                                torch.cat((input[3,0:1], output[3,0:1], target[3,0:1]), dim=2),
                                                torch.cat((input[4,0:1], output[4,0:1], target[4,0:1]), dim=2)
                                                ),
                                              dim=1), epoch)

    print_gpu_usage(f'After validation epoch {epoch}', device, GPU_DEBUG)

    
    if epoch == 1 or epoch % print_every_nth_epoch == 0 or epoch == num_epochs:
        rmse_train = math.sqrt(train_loss_epoch)
        rmse_val = math.sqrt(val_loss_epoch)
        print("Epoch: %d, Train: %.8f, Val: %.8f, Time: %d seconds" % 
              (epoch, rmse_train, 
               rmse_val, (perf_counter() - epoch_time) ))

    writer.add_scalar('Loss/train', train_loss_epoch, epoch)
    writer.add_scalar('Loss/validation', val_loss_epoch, epoch)

    if val_loss_epoch < min_loss:
        min_loss = val_loss_epoch
        epoch_best_loss = epoch
        save_checkpoint(epoch, os.path.join(save_dir, 'best_loss.pth'))

    if save_every_epoch:
        save_checkpoint(epoch, os.path.join(save_dir, "epoch-%d.pth" % epoch))

end_time = perf_counter()
print()
print(f'Training ended at {time.asctime(time.localtime())}')
print(f'Time taken is {(end_time-start_time):.2f} seconds')
print("Lowest validation loss in epoch %d" % epoch_best_loss)
print()
print()
# train and val ends
