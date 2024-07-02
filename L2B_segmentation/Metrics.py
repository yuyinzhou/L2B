import numpy as np

def dice_coeff(im1, im2, empty_score=1.0, need_ori=False):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    #im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)
    if need_ori:
        return (2. * intersection.sum() / im_sum).item(), im2.numpy()
    else:
        return 2. * intersection.sum() / im_sum
