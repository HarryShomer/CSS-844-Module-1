import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 
from plantcv import plantcv as pcv

import skimage
from skimage.segmentation import slic
from skimage.measure import regionprops


IMG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "all_imgs") 


def dist_from_img_center(segment, img):
    """
    Return euclidean distance from center of image for segement

    Parameters:
    -----------
        segment: regionprops
            Region properties for segment
        img: np.array
            Original image
    """
    segment_center = np.array(segment.centroid)
    img_center = np.array([img.shape[0] / 2, img.shape[1] / 2]) 

    return np.linalg.norm(img_center - segment_center, 2)



def get_mask(img):
    """
    Perform K-Means segmentation and detect the centermost segment in the image

    Parameters:
    -----------
        img: np.array
            Hopefully just the corn

    Returns:
    --------
    tuple
        masked_img, mask
    """
    img_with_segments = slic(img, n_segments=5, convert2lab=True, compactness=100) 

    segment_dists = [dist_from_img_center(prop, img) for prop in regionprops(img_with_segments)]
    center_segment = np.argmin(segment_dists) + 1

    img_with_segments_3d = np.repeat(img_with_segments[:, :, np.newaxis], 3, axis=2)

    # Set everything besides center to black
    center_mask = np.where(img_with_segments_3d == center_segment, 255, 0)

    # Apply mask to image
    masked_img = pcv.apply_mask(img, center_mask, 'black')

    # Convert from un-normalized
    if np.max(masked_img) != 255:
        masked_img = (masked_img * 255).astype(np.uint8)

    return masked_img, center_mask


def img_color_analysis(img_name, img_lab, img_mask):
    """
    Analyze the color properties of the LAB image.

    Done by extracting:
        1. Summary statistics of B channel (mean, median, ...)
        2. Grading the image based on the most populous bin of pixels
    """
    # All color metrics for img are stored here
    img_by_metrics = {"img": os.path.basename(img_name)}

    masked_img_lab = pcv.apply_mask(img_lab, img_mask, 'black')

    # Mask is 3D of same matrix...just extract one slice
    img_mask_2d = img_mask[:, :, 0].astype(np.uint8)

    # Only extract corn portion of Blue-Yellow channel
    # center_mask_2d = 0 for black and > 0 otherwise
    blue_yellow_channel = masked_img_lab[:, :, 2]
    blue_yellow_just_corn = blue_yellow_channel[img_mask_2d > 0]

    for m in ['mean', 'median', 'std', 'min', 'max']:
        img_by_metrics[m] = getattr(np, m)(blue_yellow_just_corn)

    # Sort list to make counting values in each bin more memory efficient and change to series for easier manipulation 
    blue_yellow_just_corn.sort()
    bval_series = pd.Series(blue_yellow_just_corn)

    # Number of pixels split between 8 ranges specified based on value
    bins = list(range(0, 144, 16))
    bin_val = bval_series.value_counts(bins=bins, sort=False)
    bin_perc = (bin_val / len(blue_yellow_just_corn)) * 100

    for b, b_perc in zip(range(len(bins) - 1), bin_perc):
        img_by_metrics[f'Bin {b}: {bins[b]}-{bins[b+1]}%'] = b_perc

    # Index of the bin with the largest amount of pixels 
    # Creating a grade from 1 - 8 for an image. +1 since starts as index 0
    img_by_metrics['grade'] = bin_val.argmax() + 1

    return img_by_metrics



def main():
    all_img_metrics = []


    for file in tqdm(os.listdir(IMG_DIR), desc="Processing Images"):
        # print(f"Processing Image {file}")
        img = skimage.io.imread(os.path.join(IMG_DIR, file))

        # Two variants: Lab version and RGB normalized (for segmentation)
        #  - Normalized RGB - For segmentation
        #  - LAB: For color analysis 
        img_norm = skimage.util.img_as_float(img) # Normalize btwn [0, 1]
        img_lab = skimage.color.rgb2lab(img)

        # Mask retrieved form segmentation
        _, center_mask = get_mask(img_norm)

        img_by_metrics = img_color_analysis(file, img_lab, center_mask)
        all_img_metrics.append(img_by_metrics)


    pd.DataFrame(all_img_metrics).to_csv("imgs_summary.csv", index=False)



if __name__ == "__main__":
    main()
