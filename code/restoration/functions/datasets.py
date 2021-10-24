import torch
import torch.utils.data as data
import torchvision.transforms as transforms


torch.set_default_tensor_type(torch.FloatTensor)


def is_invalid_image(img):
    # the actual pixel value is normalized in 0 to 1 from 0 to 65535.0
    thresh = 8e-2 
    if torch.min(img).item() < thresh and torch.max(img).item() < thresh:
        return True
    if torch.mean(img).item() < thresh and torch.max(img).item() < thresh:
        return True


def positional_encoding(mat, pos_enc_len, min_freq):
    num_rows, num_cols = mat.shape
    freqs = min_freq ** (2 * (torch.arange(pos_enc_len)//2) / pos_enc_len)
    pos_enc_vec = mat.reshape(-1, 1) * freqs.reshape(1, -1)
    pos_enc_vec[:, ::2] = torch.cos(pos_enc_vec[:, ::2])
    pos_enc_vec[:, 1::2] = torch.sin(pos_enc_vec[:, 1::2])
    pos_enc_mat = pos_enc_vec.reshape(num_rows, num_cols, pos_enc_len)
    pos_enc_mat = pos_enc_mat.permute(2, 0, 1)
    return pos_enc_mat


def get_fixed_coords(pattern_idx, crop_size, patch_size, pattern_size):
    if pattern_idx == 8 or pattern_idx == 5 or pattern_idx == 2: # pattern_size = 32 
        fixed_coords = torch.empty((0, 2))
        for i in range(0, crop_size-patch_size[0], pattern_size):
            for j in range(0, crop_size-patch_size[1], pattern_size):
                start_r = i 
                start_c = j + 6
                coords_this = torch.tensor([[start_r, start_c]])
                fixed_coords = torch.cat((fixed_coords, coords_this), dim=0)
    return fixed_coords


class RestoreAssortedChannel(data.Dataset):
    """
    Restores mosaiced data of assorted filters.
    Uses the file that contains multiple scenes e.g. 0625_data4b_train_1024x1024.pth or 0625_data4b_val_1024x1024.pth

    rand_patches_flag: True pick a random patch in each getitem call
                       False generate fixed indices in init and use the same for each epoch

    pattern_idx: >= 0 use the index
                   -1 use random index in each getitem call
                   -2 generate fixed indices in init and use the same for each epoch

    scene_idx: >= 0 use the index
                 -1 use random index in each getitem call
                 -2 generate fixed indices in init and use the same for each epoch
                   
    assort_meas||guide_image||pos_enc -> assort_sim
    """
    def __init__(self, pthfile, patch_size, num_patches, rand_patches_flag, scene_idx, pattern_idx, pos_enc, use_guide_image, transform):
        from PIL import Image
        data = torch.load(pthfile)

        self.assort_meas = data['assort_meas']
        self.assort_index = data['assort_index']
        self.assort_sim = data['assort_sim']

        self.num_scenes = self.assort_meas.shape[0]
        self.num_patterns = self.assort_meas.shape[1]
        self.num_rows = self.assort_meas.shape[2]
        self.num_cols = self.assort_meas.shape[3]

        self.patch_size = patch_size

        self.use_guide_image = use_guide_image

        if self.use_guide_image:
            self.guide_image = data['guide_image'] # num_scenes x 3 x H x W

        self.transform = transform
        self.num_patches = num_patches
                
        self.rand_patches_flag = rand_patches_flag
        if not self.rand_patches_flag:
            # We DO NOT make sure all patches have at least one filter_idx present
            self.fixed_coords = torch.rand((num_patches,2)) * torch.tensor([self.num_rows-self.patch_size[0], self.num_cols-self.patch_size[1]])
            self.fixed_coords = torch.floor(self.fixed_coords)
        
        self.pattern_idx = pattern_idx
        if pattern_idx == -2:
            self.fixed_patterns = torch.rand(num_patches) * self.num_patterns
            self.fixed_patterns = torch.floor(self.fixed_patterns)
        
        self.scene_idx = scene_idx
        if scene_idx == -2:
            self.fixed_scenes = torch.rand(num_patches) * self.num_scenes
            self.fixed_scenes = torch.floor(self.fixed_scenes)
        
        self.pos_enc = pos_enc
            
        self.mesh_row, self.mesh_col = torch.meshgrid(torch.arange(0, self.num_rows, dtype=torch.float), 
                                                              torch.arange(0, self.num_cols, dtype=torch.float))
        if self.pos_enc['enabled']:
            self.pos_enc_assort_index = torch.zeros((self.num_patterns, self.pos_enc['len'], self.num_rows, self.num_cols))
            for pattern_idx in range(self.num_patterns):
                    self.pos_enc_assort_index[pattern_idx] = positional_encoding(self.assort_index[0][pattern_idx], 
                                                                self.pos_enc['len'], self.pos_enc['min_freq'])
            if self.pos_enc['nd']:
                self.pos_enc_mesh_row = positional_encoding(self.mesh_row, self.pos_enc['len'], self.pos_enc['min_freq'])
                self.pos_enc_mesh_col = positional_encoding(self.mesh_col, self.pos_enc['len'], self.pos_enc['min_freq'])

    def __getitem__(self, index):
        # pick one scene
        if self.scene_idx == -1: # random
            scene_idx = torch.randint(self.num_scenes, (1,)).item()
        elif self.scene_idx == -2: # fixed during init
            scene_idx = int(self.fixed_scenes[index])
        else: # fixed by user
            scene_idx = self.scene_idx

        assort_sim = self.assort_sim[scene_idx]
        assort_index = self.assort_index[scene_idx]
        assort_meas = self.assort_meas[scene_idx]
        if self.use_guide_image:
            guide_image = self.guide_image[scene_idx]

        # pick one pattern
        if self.pattern_idx == -1:
            pattern_idx = torch.randint(self.num_patterns, (1,)).item()
        elif self.pattern_idx == -2: # fixed during init
            pattern_idx = int(self.fixed_patterns[index])
        else:
            pattern_idx = self.pattern_idx
        
        # pick a crop
        while True:
            if self.rand_patches_flag: # random
                i, j, h, w = transforms.RandomCrop.get_params(transforms.ToPILImage()(assort_index[0]), output_size=self.patch_size)
            else: # fixed during init
                i = int(self.fixed_coords[index, 0].item())
                j = int(self.fixed_coords[index, 1].item())
                h = self.patch_size[0]
                w = self.patch_size[1]

            assort_sim = transforms.functional.crop(assort_sim[pattern_idx], i, j, h, w) # BxB
            if not is_invalid_image(assort_sim):
                break

        if self.use_guide_image:
            guide_image = transforms.functional.crop(guide_image, i, j, h, w) # 3xBxB

        assort_index = transforms.functional.crop(assort_index[pattern_idx], i, j, h, w) # BxB
        assort_meas = transforms.functional.crop(assort_meas[pattern_idx], i, j, h, w) # BxB

        if self.pos_enc['enabled']:
            pos_enc_assort_index = transforms.functional.crop(self.pos_enc_assort_index[pattern_idx], i, j, h, w)
            if self.pos_enc['nd']:
                pos_enc_mesh_row = transforms.functional.crop(self.pos_enc_mesh_row, i, j, h, w)
                pos_enc_mesh_col = transforms.functional.crop(self.pos_enc_mesh_col, i, j, h, w)
                pos_enc_chan = torch.cat((pos_enc_mesh_row, pos_enc_mesh_col, pos_enc_assort_index), dim=0)
            else:
                pos_enc_chan = pos_enc_assort_index

        if self.pos_enc['enabled']:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image, pos_enc_chan), dim=0).type(torch.float) # (1+3+3*pos_enc[len]) x B x B
            else:
                input = torch.cat((assort_meas.unsqueeze(0), pos_enc_chan), dim=0).type(torch.float) # (1+3*pos_enc[len]) x B x B
        else:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image), dim=0).type(torch.float) # (1+3) x B x B
            else:
                input = assort_meas.unsqueeze(0).type(torch.float) # 1 x B x B

        target = assort_sim.unsqueeze(0).type(torch.float) # 1xBxB
        
        input[0] = self.transform(input[0])
        target = self.transform(target)
        
        return (input, target) 
    
    def __len__(self):
        return self.num_patches



class RestoreDataTestAssortedSingleScene(data.Dataset):
    """
    Loads mosaiced data of assorted filters from all patterns for restoring a single scene.
    Uses the file that contains single scene e.g. data/restore/0422.SleepingDead_data_1024x1024.pth
    """
    def __init__(self, pthfile, patch_size, pos_enc, use_guide_image, transform, pattern_idx=-1):
        import sys
        sys.path.append('../functions/')
        from helpers import unravel_index 

        data = torch.load(pthfile)
        self.assort_sim = data['assort_sim']
        self.assort_meas = data['assort_meas']
        self.assort_index = data['assort_index']

        self.num_rows = self.assort_meas.shape[1]
        self.num_cols = self.assort_meas.shape[2]

        self.num_patterns = self.assort_meas.shape[0] if pattern_idx < 0 else 1
        self.pattern_idx = pattern_idx

        self.patch_size = patch_size
        self.use_guide_image = use_guide_image
        if use_guide_image:
            self.guide_image = data['guide_image'] # 3 x H x W
        self.transform = transform

        self.unraveled_indices = unravel_index(torch.tensor(range(self.__len__())), 
                                (self.num_patterns, self.num_rows // self.patch_size[0], self.num_cols // self.patch_size[1])
                               )
        self.pos_enc = pos_enc
            
        self.mesh_row, self.mesh_col = torch.meshgrid(torch.arange(0, self.num_rows, dtype=torch.float), 
                                                              torch.arange(0, self.num_cols, dtype=torch.float))
        if self.pos_enc['enabled']:
            # create zero pos enc for all patterns irrespective of which pattern is asked for.
            self.pos_enc_assort_index = torch.zeros((self.assort_meas.shape[0], self.pos_enc['len'], self.num_rows, self.num_cols))
            if self.pattern_idx < 0: # find pos enc for all patterns
                for pattern_idx in range(self.num_patterns):
                    self.pos_enc_assort_index[pattern_idx] = positional_encoding(self.assort_index[pattern_idx], 
                                                                self.pos_enc['len'], self.pos_enc['min_freq'])
            else: # only do find pos enc for the desired pattern
                self.pos_enc_assort_index[self.pattern_idx] = positional_encoding(self.assort_index[self.pattern_idx], 
                                                                self.pos_enc['len'], self.pos_enc['min_freq'])

            if self.pos_enc['nd']:
                self.pos_enc_mesh_row = positional_encoding(self.mesh_row, self.pos_enc['len'], self.pos_enc['min_freq'])
                self.pos_enc_mesh_col = positional_encoding(self.mesh_col, self.pos_enc['len'], self.pos_enc['min_freq'])


    def __getitem__(self, index):
        pattern_idx = self.unraveled_indices[index][0].item() if self.pattern_idx < 0 else self.pattern_idx
        i = self.unraveled_indices[index][1].item() * self.patch_size[0]
        j = self.unraveled_indices[index][2].item() * self.patch_size[1]
        h = self.patch_size[0]
        w = self.patch_size[1]

        assort_sim = transforms.functional.crop(self.assort_sim[pattern_idx], i, j, h, w) # BxB
        assort_index = transforms.functional.crop(self.assort_index[pattern_idx], i, j, h, w) # BxB
        assort_meas = transforms.functional.crop(self.assort_meas[pattern_idx], i, j, h, w) # BxB
        if self.use_guide_image:
            guide_image = transforms.functional.crop(self.guide_image, i, j, h, w) # 3xBxB

        if self.pos_enc['enabled']:
            pos_enc_assort_index = transforms.functional.crop(self.pos_enc_assort_index[pattern_idx], i, j, h, w)
            if self.pos_enc['nd']:
                pos_enc_mesh_row = transforms.functional.crop(self.pos_enc_mesh_row, i, j, h, w)
                pos_enc_mesh_col = transforms.functional.crop(self.pos_enc_mesh_col, i, j, h, w)
                pos_enc_chan = torch.cat((pos_enc_mesh_row, pos_enc_mesh_col, pos_enc_assort_index), dim=0)
            else:
                pos_enc_chan = pos_enc_assort_index

        if self.pos_enc['enabled']:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image, pos_enc_chan), dim=0).type(torch.float) # (1+3+3*pos_enc[len]) x B x B
            else:
                input = torch.cat((assort_meas.unsqueeze(0), pos_enc_chan), dim=0).type(torch.float) # (1+3*pos_enc[len]) x B x B
        else:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image), dim=0).type(torch.float) # (1+3*pos_enc[len]) x B x B
            else:
                input = assort_meas.unsqueeze(0).type(torch.float) # 1 x B x B
                
        target = assort_sim.unsqueeze(0).type(torch.float)

        input[0] = self.transform(input[0])
        target = self.transform(target)

        # pattern_idx = -1 => any of the 13
        coords = [pattern_idx, i, j, h, w]
        if self.pattern_idx >= 0: # single pattern_idx
            coords[0] = 0

        return coords, input, target
    
    def __len__(self):
        return self.num_patterns * (self.num_rows // self.patch_size[0]) * (self.num_cols // self.patch_size[1])


class RestoreDataTestAssortedSingleSceneSinglePattern(data.Dataset):
    """
    Loads mosaiced data of assorted filters from a single pattern for restoring a single scene.
    Uses the file that contains single scene e.g. data/restore/0422.SleepingDead_data_1024x1024.pth

    Works only for certain pattern_idx values. See get_fixed_coords().
    """
    def __init__(self, pthfile, patch_size, pattern_size, pattern_idx, pos_enc, use_guide_image, transform):
        import sys
        sys.path.append('../functions/')
        from helpers import unravel_index 

        data = torch.load(pthfile)
        self.assort_sim = data['assort_sim']
        self.assort_meas = data['assort_meas']
        self.assort_index = data['assort_index']

        self.num_rows = self.assort_meas.shape[1]
        self.num_cols = self.assort_meas.shape[2]
        self.num_patterns = self.assort_meas.shape[0] if pattern_idx < 0 else 1
        self.patch_size = patch_size
        self.pattern_size = pattern_size
        self.pattern_idx = pattern_idx

        self.use_guide_image = use_guide_image

        if self.use_guide_image:
            self.guide_image = data['guide_image'] # num_scenes x 3 x H x W

        self.transform = transform
        self.fixed_coords = get_fixed_coords(pattern_idx, self.num_rows, self.patch_size, self.pattern_size)

        self.pos_enc = pos_enc

        self.mesh_row, self.mesh_col = torch.meshgrid(torch.arange(0, self.num_rows, dtype=torch.float), 
                                                              torch.arange(0, self.num_cols, dtype=torch.float))
        if self.pos_enc['enabled']:
            self.pos_enc_mesh_row = positional_encoding(self.mesh_row, self.pos_enc['len'], self.pos_enc['min_freq'])
            self.pos_enc_mesh_col = positional_encoding(self.mesh_col, self.pos_enc['len'], self.pos_enc['min_freq'])

    def __getitem__(self, index):
        pattern_idx = self.pattern_idx
        i = int(self.fixed_coords[index, 0].item())
        j = int(self.fixed_coords[index, 1].item())
        h = self.patch_size[0]
        w = self.patch_size[1]

        assort_sim = transforms.functional.crop(self.assort_sim[pattern_idx], i, j, h, w) # BxB
        assort_index = transforms.functional.crop(self.assort_index[pattern_idx], i, j, h, w) # BxB
        assort_meas = transforms.functional.crop(self.assort_meas[pattern_idx], i, j, h, w) # BxB

        if self.use_guide_image:
            guide_image = transforms.functional.crop(self.guide_image, i, j, h, w) # 3xBxB
        
        if self.pos_enc['enabled']:
                pos_enc_mesh_row = transforms.functional.crop(self.pos_enc_mesh_row, i, j, h, w)
                pos_enc_mesh_col = transforms.functional.crop(self.pos_enc_mesh_col, i, j, h, w)
                pos_enc_chan = torch.cat((pos_enc_mesh_row, pos_enc_mesh_col), dim=0)

        if self.pos_enc['enabled']:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image, pos_enc_chan), dim=0).type(torch.float) # (1+3+3*pos_enc[len]) x B x B
            else:
                input = torch.cat((assort_meas.unsqueeze(0), pos_enc_chan), dim=0).type(torch.float) # (1+3*pos_enc[len]) x B x B
        else:
            if self.use_guide_image:
                input = torch.cat((assort_meas.unsqueeze(0), guide_image), dim=0).type(torch.float) # (1+3*pos_enc[len]) x B x B
            else:
                input = assort_meas.unsqueeze(0).type(torch.float) # 1 x B x B

        target = assort_sim.unsqueeze(0).type(torch.float)

        input[0] = self.transform(input[0])
        target = self.transform(target)

        # pattern_idx = -1 => any of the 13
        coords = [pattern_idx, i, j, h, w]
        if self.pattern_idx >= 0: # single pattern_idx
            coords[0] = 0

        return coords, input, target
    
    def __len__(self):
        return self.fixed_coords.shape[0]
