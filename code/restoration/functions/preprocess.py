import torch

class PoissonNoise(object):
    """Add photon noise.
    """
    def __call__(self, z):
        y, x = z
        return torch.poisson(y), x

class ReadNoise(object):
    """Add read (Gaussian) noise for a given SNR in dB.
    
    snr dB = 20 log10 [mean(s) / std(n)]
    std = 10^ -(snr db / 20)  * mean(s)
    """
    def __init__(self, snr):
        self.snr = snr
        
    def __call__(self, z):
        y, x  = z
        std = 10.0 ** (-self.snr / 20.0) * torch.mean(y)
        return y + torch.randn(y.shape) * std, x
    
class Normalize(object):
    """Normalize the sample tensor by its maximum.
    """
    def __init__(self, only_y=False):
        self.only_y = only_y
        
    def __call__(self, z):
        y, x = z
        if self.only_y:
            return y / torch.max(y), x
        return y / torch.max(y), x / torch.max(y)

class UnitNorm(object):
    """Normalize the sample tensor by its maximum.
    """
    def __init__(self, only_y=False):
        self.only_y = only_y
        
    def __call__(self, z):
        y, x = z
        if self.only_y:
            return y / torch.norm(y), x
        return y / torch.norm(y), x / torch.norm(y)

class Scale(object):
    """Scale by the given amount.
    """
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, z):
        y, x = z
        return y * self.scale, x * self.scale
    
class ScaleRand(object):
    """Scale by a random amount in the given range.
    """
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, z):
        y, x = z
        scale = torch.randint(int(self.scale[0]), int(self.scale[1]), (1,)).item()
        return y * scale, x * scale

class NormalizeY(object):
    """Normalize the sample tensor by its maximum.
    """
        
    def __call__(self, y):
        return y / torch.max(y)
    
class ScaleY(object):
    """Scale by the given amount.
    """
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, y):
        return y * self.scale

class UnitNormY(object):
    """Normalize the sample tensor by its maximum.
    """
    def __init__(self, only_y=False):
        self.only_y = only_y
        
    def __call__(self, y):
        return y / torch.norm(y)

class RandPatternMaskChannel(object):
    """ INCOMPLETE
    Mask channel other than the initialized.
    """
    def __init__(self, pattern_size=64, num_filts=256):
        """
        pattern is a PxP tensor containing values in [0,p]
        mask_channel is an integer in [0,p]
        """
        self.num_filts = num_filts

    def __call__(self, x):
        mask_channel = torch.randint(self.num_filts, (1,)).item()
        return x * (self.pattern == mask_channel)


class PickRandChannel(object):
    def __call__(self, x):
        pick_channel = torch.randint(x.shape[0], (1,)).item()
        return x[pick_channel]