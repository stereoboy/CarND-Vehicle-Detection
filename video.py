import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import sys
import os

with open('camera_cal.npz','rb') as f:
    camera_cal = np.load(f)
    mtx = camera_cal['mtx']
    dist = camera_cal['dist']

W, H = 2500, 2280
trim_w, trim_h = 1500, 800
offset = 400
src = np.array([[594, 450], [684, 450], [1056, 690], [250,690]], np.float32)
dst = np.array([[offset + 250, 0], [offset + 1056, 0], [offset + 1056, H], [offset + 250, H]], np.float32)
Center = offset + (250 + 1056)//2

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1]) & (l_channel >= 100)] = 1
    return binary_output

def combined_threshold(image):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=11, thresh=(0.7, 1.3))

    hls_binary = hls_select(image, thresh=(90, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1

    return combined

def hls_select_v2(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    kernel = np.ones((11,11),np.uint8)

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1]) | (l_channel >= 200)] = 1
    binary_output = cv2.dilate(binary_output,kernel,iterations = 1)
    return binary_output

def combined_threshold_v2(image):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=5, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=5, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=5, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    hls_binary = hls_select_v2(image, thresh=(100, 255))

    combined = np.zeros_like(dir_binary)

    # visualization
#    gradxy = np.zeros_like(dir_binary)
#    gradxy[(gradx == 1) & (grady == 1) ] = 1
#
#    gradient = np.zeros_like(dir_binary)
#    gradient[ ((mag_binary == 1) & (dir_binary == 1)) ] = 1
#
#    color_binary = np.dstack((gradxy, gradient, hls_binary))
#    color_binary = 255*color_binary.astype(np.uint8)

    combined[(hls_binary == 1)&(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) ] = 1

    return combined

def unwarp_trim(img):
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    #delete the next two lines
    return warped[H-trim_h:]

def recover(img):
    h, w = img.shape[:2]
    new = np.zeros((H, W, 3), np.uint8)
    new[H-h:] = img

    return new

def inv_warp(img, line_img):
    h, w = img.shape[:2]

    line_img = recover(line_img)

    inv_warped = cv2.warpPerspective(line_img, Minv, (w, h), flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1, inv_warped, 0.3, 0)
    return result

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30.0/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, name):

        self.name = name
        # numbers of interations
        self.n = 10
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # the last n fits of the line
        self.recent_fits = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # threshold for difference
        self.threshold_fit = 700
        self.threshold_fitx = 200
        self.filter = ''

    def check_outlier(self, fit, fitx):

        self.diffs = np.abs(self.current_fit - fit)
        self.diffx = np.mean(np.abs(self.bestx - fitx))
        if self.name == self.filter:
            print(self.name)
            print(self.diffs, np.linalg.norm(self.diffs))
            print(self.diffx)

        if np.linalg.norm(self.diffs) > self.threshold_fit:
            return False
        if self.diffx > self.threshold_fitx:
            return False

        return True

    def set(self, fit, fitx):
        self.detected = True
        self.current_fit = fit
        self.recent_fits.append(fit)
        if len(self.recent_fits) > self.n:
            self.recent_fits.pop(0)
        self.best_fit = np.mean(np.vstack(self.recent_fits), axis=0)

        self.recent_xfitted.append(fitx)
        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.pop(0)

        self.bestx = np.mean(np.array(self.recent_xfitted), axis=0)

    def update(self, fit, ploty):
        y_eval = np.max(ploty)
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        if self.current_fit is not None:

            if self.check_outlier(fit, fitx):
                self.set(fit, fitx)
            else:
                if self.name == self.filter:
                    print("skip this frame")
                self.detected = False
                if len(self.recent_fits) > self.n:
                    self.recent_fits.pop(0)
        else:
            self.current_fit = fit
            self.set(fit, fitx)

        return self.bestx

class Lane():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.MARGIN = 50
        self.size_threshold = 200

        self.left_line = Line("LEFT")
        self.right_line = Line("RIGHT")

    def find_lane_lines_first(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.MARGIN
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None
        if len(leftx) > self.size_threshold and len(lefty) > self.size_threshold:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > self.size_threshold and len(righty) > self.size_threshold:
            right_fit = np.polyfit(righty, rightx, 2)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        left_warp = np.zeros_like(binary_warped).astype(np.uint8)
        right_warp = np.zeros_like(binary_warped).astype(np.uint8)

        left_warp[(lefty, leftx)] = 255
        right_warp[(righty, rightx)] = 255
        color_warp = np.dstack((right_warp, warp_zero, left_warp))

        return left_fit, right_fit, color_warp

    def find_lane_lines_after(self, binary_warped, left_fit, right_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.MARGIN
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None
        if len(leftx) > self.size_threshold and len(lefty) > self.size_threshold:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > self.size_threshold and len(righty) > self.size_threshold:
            right_fit = np.polyfit(righty, rightx, 2)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        left_warp = np.zeros_like(binary_warped).astype(np.uint8)
        right_warp = np.zeros_like(binary_warped).astype(np.uint8)

        left_warp[(lefty, leftx)] = 255
        right_warp[(righty, rightx)] = 255
        color_warp = np.dstack((right_warp, warp_zero, left_warp))
        return left_fit, right_fit, color_warp

    def find_lane_lines(self, binary_warped):
        if (self.left_fit is None) and (self.right_fit is None):
            left_fit, right_fit, color_warp = self.find_lane_lines_first(binary_warped)
            if left_fit is not None:
                self.left_fit = left_fit
            if right_fit is not None:
                self.right_fit = right_fit
        else:
            #left_fit, right_fit, color_warp = self.find_lane_lines_after(binary_warped, self.left_fit, self.right_fit)
            left_fit, right_fit, color_warp = self.find_lane_lines_first(binary_warped)
            if left_fit is not None:
                self.left_fit = left_fit
            if right_fit is not None:
                self.right_fit = right_fit

        h, w = binary_warped.shape[:2]

        ploty = np.linspace(0, h-1, num=h) # to cover same y-range as image

#        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
#        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        left_fitx = self.left_line.update(self.left_fit, ploty)
        right_fitx = self.right_line.update(self.right_fit, ploty)

        lane_warp = np.zeros_like(binary_warped).astype(np.uint8)

        if left_fitx is not None and right_fitx is not None:

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(lane_warp, np.int_([pts]), (255))
            color_warp[:,:,1] = lane_warp

        avg_fit = self.left_line.best_fit + self.right_line.best_fit
        #curvature = self.left_line.radius_of_curvature + self.right_line.radius_of_curvature
        curvature = ((1 + (2*avg_fit[0]*np.max(ploty)*ym_per_pix + avg_fit[1])**2)**1.5) / np.absolute(2*avg_fit[0])

        center = xm_per_pix*(- Center + (left_fitx[-1] + right_fitx[-1])/2)

        cv2.imshow('color_warp', color_warp)

        return color_warp, curvature, center

lane = Lane()

#def pipeline(img):
#    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
#    #combined = combined_threshold(undistorted)
#    combined = combined_threshold_v2(undistorted)
#    binary_warped = unwarp_trim(combined)
#    cv2.imshow('warped', cv2.resize(binary_warped*255, (binary_warped.shape[1]//2, binary_warped.shape[0]//2)))
#    line_img = lane.find_lane_lines(binary_warped)
#    result = inv_warp(img, line_img)
#
#    return result

def pipeline(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    combined = combined_threshold_v2(undistorted)
    binary_warped = unwarp_trim(combined)
    line_img, curvature, center = lane.find_lane_lines(binary_warped)
    result = inv_warp(img, line_img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text0 = 'curvature:%.4fm'%(curvature)
    text1 = 'center:%.4fm'%(center)
    cv2.putText(result, text0, (10,100), font, 1.5, (255,255,255), 3)
    cv2.putText(result, text1, (10,200), font, 1.5, (255,255,255), 3)

    return result

def main():
    cap = cv2.VideoCapture(sys.argv[1])
    #
    # sudo apt-get install ffmpeg x264 libx264-dev
    #
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    #out = cv2.VideoWriter('result_' + sys.argv[1],fourcc, 25.0, (1280,720))
    out = cv2.VideoWriter('result_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + ".mp4", fourcc, 25.0, (1280,720))

    while(cap.isOpened()):
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret:
            new = pipeline(frame)
            out.write(new)
            cv2.imshow('frame', cv2.resize(new, (new.shape[1]//2, new.shape[0]//2)))
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
