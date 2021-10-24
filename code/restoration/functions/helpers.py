import math
import numpy as np
import random
import scipy.io as sio
import torch

from scipy.interpolate import interp1d
from skimage import color



def reset_random(seed):
    """
    Sets pytorch, numpy and python random seeds to given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_gpu_usage(help_str, device, gpu_debug):
    """
    Print current GPU usage.
    """
    if gpu_debug:
        print(help_str)
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**2, 1), 'MB')
        print('Cached:   ', round(torch.cuda.memory_cached(device)/1024**2, 1), 'MB')
        print()


def print_min_max_dataset(dataloader):
    min0 = float('inf')
    max0 = float('-inf')
    min1 = float('inf')
    max1 = float('-inf')
    for i, sample in enumerate(dataloader):
        if torch.min(sample[0]) < min0:
            min0 = torch.min(sample[0])
        if torch.max(sample[0]) > max0:
            max0 = torch.max(sample[0])
        if torch.min(sample[1]) < min1:
            min1 = torch.min(sample[1])
        if torch.max(sample[1]) > max1:
            max1 = torch.max(sample[1])

    print('Input min max', min0, max0)
    print('Target min max', min1, max1)


def crop_center_array(arr, crop_rows, crop_cols):
    """
    Center-crop a numpy array image.

    Args:
        arr: Numpy array of size W x H x () to be cropped.
        crop_nrows: Number of rows in crop.
        crop_ncols: Number of cols in crop.

    Returns:
        Numpy array of size crop_nrows x crop_ncols x ()
    """
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    start_row = nrows // 2 - crop_rows // 2
    start_col = ncols // 2 - crop_cols // 2
    return arr[start_row:start_row+crop_rows, start_col:start_col+crop_cols]


def unravel_index(indices, shape):
    """
    Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).

    Copied from https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def hsi2rgb_t_vijay(hsi, hsi_wvl, cam_spectra, cam_wvl, channel_last=False, 
            channelwise_norm=False, spectral_resp_norm=False):
    """
    Hyperspectral image (cube) to RGB image as pytorch tensors.

    Args:
        hsi: hyperspectral image of size (C, H, W)
        hsi_wvl: hyperspectral wavelengths of size (C,)
        cam_spectra: camera RGB spectral response of size (P, 3)
        cam_wvl: camera wavelengths of size (P,)

    Returns:
        rgb: RGB image of size (H, W, 3) if channel_last else (3, H, W)
    """
    cam_wvl = cam_wvl.numpy()
    cam_spectra = cam_spectra.numpy()
    hsi_wvl = hsi_wvl.numpy()

    C, H, W = hsi.shape
    rgb = torch.zeros((3, H, W))

    for k in range(3):
        resamp_wvl = interp1d(cam_wvl, cam_spectra[:, k], fill_value=(0, 0), bounds_error=False)(hsi_wvl)
        resamp_wvl = torch.from_numpy(resamp_wvl)
        rgb[k] = torch.sum(hsi * resamp_wvl.reshape(-1, 1, 1), dim=0)
        if channelwise_norm:
            rgb[k] = rgb[k] / torch.max(rgb[k])
        if spectral_resp_norm:
            rgb[k] = rgb[k] / torch.sum(resamp_wvl)

    return rgb if not channel_last else rgb.permute(1, 2, 0)


def hsi2rgb_vijay(hsi, hsi_wvl, channel_last=False,
               channelwise_norm=False, spectral_resp_norm=False):
    """
    Hyperspectral image (cube) to RGB image as numpy arrays.

    Usage:
        rgb = hsi2rgb_vijay(hsi, wvl, channel_last=True)

    Args:
        hsi: hyperspectral image of size (C, H, W)
        hsi_wvl: hyperspectral wavelengths of size (C,)
        cam_spectra: camera RGB spectral response of size (P, 3)
        cam_wvl: camera wavelengths of size (P,)

    Returns:
        rgb: RGB image of size (H, W, 3) if channel_last else (3, H, W)
    """
    cam_resp_rgb_matfile = '/home/datasets/ProAsPix/resources/cam_resp.mat'
    cam_resp = sio.loadmat(cam_resp_rgb_matfile)
    cam_spectra = cam_resp['resp']
    cam_wvl = cam_resp['wvl'].flatten().astype(float)

    C, H, W = hsi.shape
    rgb = np.zeros((3, H, W))
    resamp_wvl = np.zeros((C, 3))
    for k in range(3):
        resamp_wvl[:, k] = interp1d(cam_wvl, cam_spectra[:, k], fill_value=(0, 0), bounds_error=False)(hsi_wvl)
#         rgb[k] = np.sum(hsi * resamp_wvl.reshape(-1, 1, 1), axis=0)
#         if channelwise_norm:
#             rgb[k] = rgb[k] / np.max(rgb[k])
#         if spectral_resp_norm:
#             rgb[k] = rgb[k] / np.sum(resamp_wvl)

    rgb = np.matmul(hsi.transpose(1, 2, 0).reshape(-1, C), resamp_wvl).reshape(H, W, 3)
    rgb = rgb.transpose(2, 0, 1)
    return rgb if not channel_last else rgb.transpose(1, 2, 0)


def print_min_max(a):
    """
    Handy function to print min and max of a tensor.
    """
    print(torch.min(a), torch.max(a))


def rescale(x, type='max'):
    if type.lower() == 'max':
        return x / np.max(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def hsi2xyz(imhyper, wavelengths):
    '''
        Function to convert a hyperspectral image to XYZ image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imxyz: XYZ image.
    '''
    cmf_data = np.genfromtxt('../functions/lin2012xyz2e_1_7sf.csv', delimiter=',')

    # Interpolate the wavelengths and x, y, z values
    cmf_data_new = np.zeros((len(wavelengths), 3))
    for idx in range(3):
        interp_func = interp1d(cmf_data[:, 0],
                                           cmf_data[:, idx+1],
                                           kind='linear',
                                           fill_value='extrapolate')
        cmf_data_new[:, idx] = interp_func(wavelengths)

    # Find valid indices for converting to RGB image.
    #valid_idx = np.where((wavelengths > min(l)) & (wavelengths < max(l)))

    [H, W, T] = imhyper.shape
    hypermat = imhyper.reshape(H*W, T)

    # Compute XYZ image
    imxyz = np.dot(hypermat, cmf_data_new);
    imxyz = imxyz.reshape(H, W, 3)
    
    return imxyz
    
    
def hsi2rgb(imhyper, wavelengths, gamma=1, normalize=True):
    '''
        Function to convert a hyperspectral image to RGB image.

        Inputs:
            imhyper: 3D Hyperspectral image.
            wavelengths: Wavelengths corresponding to each slice.
            gamma: Gamma correction constant. Default is 1.

        Outputs:
            imrgb: RGB image.
    '''
    imxyz = hsi2xyz(imhyper, wavelengths)

    # Before you convert to rgb, normalize
    if normalize:
#         imxyz /= imxyz.max()
        imxyz = rescale(imxyz, 'max')

    # Compute RGB image from xyz
    imrgb = pow(color.xyz2rgb(imxyz), 1.0/gamma)

    return imrgb


def get_metrics(reconst, target):
    '''
        Function to calculate MSE, PSNR, and SNR.

        Inputs:
            reconst and target
        
        Outputs:
            dict with keys mse, psnr and snr
    '''
    max_val = 1.0
    err = torch.abs(reconst - target)
    mse = torch.mean(err ** 2)
    snr = 10 * torch.log10(torch.mean(target ** 2) / torch.mean(err ** 2))
    psnr = 10 * torch.log10(max_val ** 2 / torch.mean(err ** 2))
    return {'mse': mse, 'psnr': psnr, 'snr': snr}