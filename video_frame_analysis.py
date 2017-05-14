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
from detect import *
import sys

def draw_bboxes(img, bbox_list):
    draw_img = img.copy()

    for bbox in bbox_list:
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)
    return draw_img

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
    scale_list = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
    start = 1100

    print('0. capture video frames')
    cap = cv2.VideoCapture(sys.argv[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    count = 0

    video_frame_list = []
    while(cap.isOpened()):
        
        ret, frame = cap.read()

        if ret:
            video_frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if count > 3:
                break
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    print('1. generate heatmaps')
    for i, frame in enumerate(video_frame_list):
        img = frame

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
        subplots[0].set_title('frame {}'.format(i))

        subplots[1].imshow(heatmap, cmap='hot')
        subplots[1].set_title('heatmap {}'.format(i))

        figs.savefig(os.path.join('./output_images/', 'heatmap_video_frame{}'.format(i) + '.png'))
        plt.show()

    print('2. label on heatmaps')
    for i, frame in enumerate(video_frame_list):
        img = frame

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
        subplots[0].set_title('frame {}'.format(i))

        subplots[1].imshow(labels[0], cmap='gray')
        subplots[1].set_title('label {}'.format(i))

        figs.savefig(os.path.join('./output_images/', 'label_video_frame{}'.format(i) + '.png'))
        plt.show()

    print('4. refine bounding boxes using heatmap')
    for i, frame in enumerate(video_frame_list):
        img = frame

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
        subplots[0].set_title('frame {}'.format(i))

        subplots[1].imshow(draw_img)
        subplots[1].set_title('final {}'.format(i))

        figs.savefig(os.path.join('./output_images/', 'final_video_frame{}'.format(i) + '.png'))
        plt.show()

if __name__ == '__main__':
    main()
