**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_notcar]: ./output_images/car_notcar.png
[color_conversion]: ./output_images/color_conversion.png
[hog]: ./output_images/hog.png
[bin_spatial]: ./output_images/bin_spatial.png
[color_histogram]: ./output_images/color_histogram.png

[sliding_window0]: ./output_images/sliding_window_1.1.png
[sliding_window1]: ./output_images/sliding_window_1.3.png
[sliding_window2]: ./output_images/sliding_window_1.5.png
[sliding_window3]: ./output_images/sliding_window_1.7.png

[base_detection0]: ./output_images/test1.jpg
[base_detection1]: ./output_images/test3.jpg
[base_detection2]: ./output_images/test4.jpg

[heatmap0]: ./output_images/heatmap_test1.jpg
[heatmap1]: ./output_images/heatmap_test3.jpg
[heatmap2]: ./output_images/heatmap_test4.jpg

[label0]: ./output_images/label_test1.jpg
[label1]: ./output_images/label_test3.jpg
[label2]: ./output_images/label_test4.jpg

[final0]: ./output_images/final_test1.jpg
[final1]: ./output_images/final_test3.jpg
[final2]: ./output_images/final_test4.jpg

[video_heatmap0]: ./output_images/heatmap_video_frame0.png
[video_heatmap1]: ./output_images/heatmap_video_frame1.png
[video_heatmap2]: ./output_images/heatmap_video_frame2.png

[video_label0]: ./output_images/label_video_frame0.png
[video_label1]: ./output_images/label_video_frame1.png
[video_label2]: ./output_images/label_video_frame2.png

[video_final0]: ./output_images/final_video_frame0.png
[video_final1]: ./output_images/final_video_frame1.png
[video_final2]: ./output_images/final_video_frame2.png

[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted Bin Spatial, Color Histogram and HOG features from the training images.

The code for this step is contained in lines #48 through #115 of the file called `visualize.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_notcar]

Here is a color converted example using the `YCrCb` color space:

![alt text][color_conversion]

Here is an example using bin spatial parameters of `size=(32, 32)`:

![alt text][bin_spatial]

Here is an example using color historam parameters of `bin_size=32`:

![alt text][color_histogram]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I fixed up the parameters as described above, based on the parameters of suggested in the lecture.

#### 3. Describe how you trained a classifier using your selected Bin Spatial, Color Histogram and HOG features.

Since it takes just less than 5 miniutes to extract our features and learn SVM classifier, I decided to use all features proposed in the lecture to use as much information as possible.
I trained a linear SVM using sklearn.svm.LinearSVC(). LinearSVC() is very simple, but is very appropriate for this project since sample data size is not large enough for avoiding overfitting.
This is written down in lines #108 through #128 of the file `train.py`

I use 9988 car files and 10093 notcar files.
The elapsed time is: 114.58 secs for extracting feature vectors and 26.97 secs for training.
I got accuracy of  0.995 on testset, which is 20% of all dataset.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use Hog Sub-sampling Window Search proposed in the lecture.
I used multiple scales: 1.33, 1.66, 2.0, 2.33 for detecting multi-size cars
All implementations are in the file `detect.py`

![alt text][sliding_window0]
![alt text][sliding_window1]
![alt text][sliding_window2]
![alt text][sliding_window3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][base_detection0]
![alt text][base_detection1]
![alt text][base_detection2]

Here are heatmaps for integration multiple detected bounding boxes.

![alt text][heatmap0]
![alt text][heatmap0]
![alt text][heatmap0]

Here are label maps for distinguishing individual vehicles after threshodling 1. It means that detection process approve only multiple-overlapped regions as a detected vehicle.

![alt text][label0]
![alt text][label1]
![alt text][label2]

Here are final results after confirming largest bounding boxes.

![alt text][final0]
![alt text][final1]
![alt text][final2]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented vehicle-detection code for video frames in the file `video.py`.

And I wrote another file `video_frame_analysis.py` for visualization and documentation. I captured 3 frames from project_video.mp4 by using Open CV video-related function and applied the detection pipeline on these frames. At last saved result images for `writeup.md`.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 3 frames and their corresponding heatmaps:

![alt text][video_heatmap0]
![alt text][video_heatmap1]
![alt text][video_heatmap2]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 3 frames:
![alt text][video_label0]
![alt text][video_label1]
![alt text][video_label2]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][video_final0]
![alt text][video_final1]
![alt text][video_final2]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For robust detection I did the following things.
* I applied lowpass filter (or smoother) detected bounding boxes. By calculating overlapped IOU(intersection over union) I added bounding box tracking mechanism. By using this lowpass filter, I got fine result.
* I finetuned parameters: scale factor for sliding-window

Future work
* For more robust detection, more fine-grain bounding box tracking system is needed. Applying tracking filters such like Kalman filter can make more successful result.
