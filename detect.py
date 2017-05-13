import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from lesson_functions import *
from feature import *
from scipy.ndimage.measurements import label

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bbox_list = []
    for xb in range(nxsteps + 1):
        for yb in range(nysteps + 1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 

                bbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                bbox_list.append(bbox)
    return draw_img, bbox_list

def draw_sliding_window(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]

    imshape = img_tosearch.shape
    w, h = (np.int(imshape[1]/scale), np.int(imshape[0]/scale))
    # Define blocks and steps as above
    nxblocks = (w // pix_per_cell) - cell_per_block + 1
    nyblocks = (h // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bbox_list = []
    for xb in range(nxsteps + 1):
        for yb in range(nysteps + 1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 

    return draw_img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_bboxes(img, bbox_list):
    draw_img = img.copy()

    for bbox in bbox_list:
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)
    return draw_img

# label[0] is segmented and labeled images
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    ystart = 400
    ystop = 656
    scale = 1.5
    spatial_size=(32, 32)
    bbox_list = []

    for scale in scale_list:
        _, sub = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        bbox_list.extend(sub)

    return bbox_list

def detect_cars(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    bbox_list = find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img = draw_bboxes(img, bbox_list)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    cv2.imshow('heatmap', heatmap.astype(np.uint8)*16)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)


    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)

    return bbox_list

def main():
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    print(dist_pickle)
    ystart = 400
    ystop = 656
    scale = 1.5
    scale_list = [1.33, 1.66, 2.00, 2.33]

    filelist = glob.glob(os.path.join('./test_images', '*.jpg'))

    print('0. sliding window')
    img = mpimg.imread(filelist[0])
    for i, scale in enumerate(scale_list):

        out_img = draw_sliding_window(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        plt.imshow(out_img)
        plt.title('sliding window scale:' + str(scale))
        plt.savefig(os.path.join('./output_images/', 'sliding_window_' + str(scale) + '.png'))
        plt.show()

    print('1. basic car detection')
    for i, filename in enumerate(filelist):
        img = mpimg.imread(filename)

        bbox_list = find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        out_img = draw_bboxes(img, bbox_list)

        plt.imshow(out_img)
        plt.title(filename)
        plt.savefig(os.path.join('./output_images/', os.path.basename(filename)))
        plt.show()

    print('2. generate heatmaps')
    for i, filename in enumerate(filelist):
        img = mpimg.imread(filename)

        bbox_list = find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img = draw_bboxes(img, bbox_list)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat,bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        figs, subplots = plt.subplots(1, 2, figsize=(15, 5))
        subplots[0].imshow(out_img)
        subplots[0].set_title(filename)

        subplots[1].imshow(heatmap, cmap='hot')
        subplots[1].set_title('heatmap ' + filename)

        figs.savefig(os.path.join('./output_images/', 'heatmap_' + os.path.basename(filename)))
        plt.show()

    print('3. label on heatmaps')
    for i, filename in enumerate(filelist):
        img = mpimg.imread(filename)

        bbox_list = find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img = draw_bboxes(img, bbox_list)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat,bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)

        figs, subplots = plt.subplots(1, 2, figsize=(15, 5))
        subplots[0].imshow(out_img)
        subplots[0].set_title(filename)

        subplots[1].imshow(labels[0], cmap='gray')
        subplots[1].set_title('label ' + filename)

        figs.savefig(os.path.join('./output_images/', 'label_' + os.path.basename(filename)))
        plt.show()

    print('4. refine bounding boxes using heatmap')
    for i, filename in enumerate(filelist):
        img = mpimg.imread(filename)

        bbox_list = find_cars_multiscale(img, scale_list, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img = draw_bboxes(img, bbox_list)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat,bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        figs, subplots = plt.subplots(1, 2, figsize=(15, 5))
        subplots[0].imshow(out_img)
        subplots[0].set_title(filename)

        subplots[1].imshow(draw_img)
        subplots[1].set_title('final ' + filename)

        figs.savefig(os.path.join('./output_images/', 'final_' + os.path.basename(filename)))
        plt.show()


if __name__ == '__main__':
    main()

