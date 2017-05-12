import os
import sys
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from feature import *

def build_filenames(dirs):
    base_dir = './data/'
    print('Search files...')

    filenames = []
    for subdir in dirs:
        subdirpath = os.path.join(base_dir, subdir)
        for root, dirs, files in os.walk(subdirpath):
            print(root)
            for ext_filter in ['*.png', '*.jpeg']:
                path = os.path.join(root, ext_filter)
                print(path)
                _filenames = glob.glob(path)
                filenames.extend(_filenames)
    return filenames

def main():
    # Divide up into cars and notcars
    notcar_dirs = ['non-vehicles', 'non-vehicles_smallset']
    car_dirs = ['vehicles', 'vehicles_smallset']

    
    cars = build_filenames(car_dirs)
    notcars = build_filenames(notcar_dirs)
    print('numbers of car files: {}'.format(len(cars)))
    print('numbers of notcar files: {}'.format(len(notcars)))

    ### TODO: Tweak these parameters and see how the results change.
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hist_bins=32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

    print('1. load one car image nad one non-car image')
    car_img = mpimg.imread(cars[0])
    notcar_img = mpimg.imread(notcars[0])

    _, subplots = plt.subplots(1, 2)
    #subplots = plt.subplots(1, 2, figsize=(20,10))
    subplots[0].imshow(car_img)
    subplots[0].set_title('Car}')
    subplots[1].imshow(notcar_img)
    subplots[1].set_title('NotCar')

    plt.show()

    print('2. convert to YCrCb')
    car_img = convert_color(car_img, conv='RGB2YCrCb')
    notcar_img = convert_color(notcar_img, conv='RGB2YCrCb')

    _, subplots = plt.subplots(3, 2)
    #_, subplots = plt.subplots(1, 2, figsize=(20,10))
    
    for i in range(3):
        subplots[i][0].imshow(car_img[:,:,i], cmap='gray')
        subplots[i][0].set_title('Car Ch{}'.format(i))
        subplots[i][1].imshow(notcar_img[:,:,i], cmap='gray')
        subplots[i][1].set_title('NotCar Ch{}'.format(i))

    plt.show()

    print('3-1. extract bin_spatial feature')
    car_spatial = bin_spatial(car_img)
    notcar_spatial = bin_spatial(notcar_img)

    _, subplots = plt.subplots(1, 2)
    print(car_spatial.shape)
    subplots[0].plot(car_spatial)
    subplots[0].set_title('Car bin_spatial')
    subplots[1].plot(notcar_spatial)
    subplots[1].set_title('NotCar bin_spatial')
    plt.show()

    print('3-2. extract color_hist feature')
    car_color_hist = color_hist(car_img, nbins=hist_bins)
    notcar_color_hist = color_hist(notcar_img, nbins=hist_bins)

    ch_offset = car_color_hist.shape[0]/3
    _, subplots = plt.subplots(3, 2)
    for i in range(3):
        subplots[i][0].plot(car_color_hist[i*ch_offset:(i + 1)*ch_offset])
        subplots[i][0].set_title('Car Color Hist Ch{}'.format(i))
        subplots[i][1].plot(notcar_color_hist[i*ch_offset:(i + 1)*ch_offset])
        subplots[i][1].set_title('NotCar Color Hist Ch{}'.format(i))

    plt.show()

    print('3-3. extract hog feature')
    _, subplots = plt.subplots(3, 2)

    for i in range(3):
        _, car_hog_img = get_hog_features(car_img[:,:,i], orient,
                    pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        _, notcar_hog_img = get_hog_features(notcar_img[:,:,i], orient,
                    pix_per_cell, cell_per_block, vis=True, feature_vec=False)

        subplots[i][0].imshow(car_hog_img, cmap='gray')
        subplots[i][0].set_title('Car HOG Ch{}'.format(i))
        subplots[i][1].imshow(notcar_hog_img, cmap='gray')
        subplots[i][1].set_title('NotCar HOG Ch{}'.format(i))
    
    plt.show()

    print('4. show sliding window search')
    
    filelist = glob.glob(os.path.join('./test_images', '*.jpg'))

    print('1. basic car detection')
    for i, filename in enumerate(filelist):
        img = mpimg.imread(filename)

        out_img, _ = find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        plt.imshow(out_img)
        plt.title(filename)
        plt.savefig(os.path.join('./output_images/', os.path.basename(filename)))
        plt.show()

if __name__ == '__main__':
    main()
