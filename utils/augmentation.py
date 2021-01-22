import imgaug as ia
from torchvision import transforms
from imgaug import augmenters as iaa
import numpy as np
import torch

def generate_augmentation(aug_pad,
                          aug_affine,
                          aug_ch_suffle,
                          aug_dropout,
                          aug_AGN,
                          aug_fliplr,
                          aug_flipud,
                          aug_percent):
    '''
    This function creates an augment for dataset transform to use. Args
    determine which augments are used. The augments are predetermined and
    appliend in same order as args.
    Args:
        aug_pad (bool): Boolean value if pad filter will be used.
        aug_affine (bool): Boolean value if affine rotation filter will be used.
        aug_ch_suffle (bool): Boolean value if channel suffle filter will be used.
        aug_dropout (bool): Boolean value if dropout filter will be used.
        aug_AGN (bool): Boolean value if Additive Gaussian Noice filter will be used.
        aug_fliplr (bool): Boolean value if left-right flip filter will be used.
        aug_flipud (bool): Boolean value if up-down flip filter will be used.
        aug_percent (float): Float between 0 and 1. The determined augments are 
                             applied randomly based on aug_percent.
    Return:
        augment (imgaug augmenter): Imgaug augmenter that can be 
                                    used to perform transformations on images.
    '''
    
    #Create all augments
    pad = iaa.Pad(px=(0, 4))
    affine = iaa.Affine(rotate=(-10, 10))
    ch_suffle = iaa.ChannelShuffle(0.35)
    dropout = iaa.Dropout(p=(0, 0.2))
    AGN = iaa.AdditiveGaussianNoise(loc=0, scale=(0, 15))
    flip_lr = iaa.Fliplr(0.5)
    flip_ud = iaa.Flipud(0.5)
    
    #Put augments to list and choose only if aug_ parameter is true
    aug_list = [pad, affine, ch_suffle, dropout, AGN, flip_lr, flip_ud]
    use_aug_list = [aug_pad, aug_affine, aug_ch_suffle, aug_dropout, aug_AGN, aug_fliplr, aug_flipud]
    aug_list = [aug_list[i] for i in np.where(use_aug_list)[0]]
    
    #Create the augment and use aug_percent to determine how oftern augments are used
    augment = iaa.Sequential([iaa.Sometimes(aug_percent, 
                            iaa.Sequential(aug_list))])
    return augment

class Transfrom_using_aug(object):
    """
    Transformer class for training dataset. Applies augment on images.
    """

    def __init__(self, augment):
        self.augment = augment

    def __call__(self, image):
        image = np.array(image)
        image = self.augment(image=image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]).forward(image)
        return image
    