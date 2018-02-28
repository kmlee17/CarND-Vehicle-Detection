## Vehicle Detection - Kevin Lee

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image4]: ./writeup_imgs/windows_rf.png
[image3]: ./writeup_imgs/heatmap.png
[image21]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./writeup_imgs/pipeline_out.png
[video1]: ./project_video.mp4

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters through mostly trial and error looking for a high accuracy on the test set.  A chart of various combinations of parameters is shown below along with accuracy scores and feature extraction times.  Adding in 'ALL' HOG channels does increase accuracy, but doubles the extraction time of just a single channel.  This could come into play if efficiency becomes an issue.  The bolded row is the parameters that I chose.

| Color Space| Orient| Pix/Cell | Cell/Block | HOG Channel | Accuracy | Extraction Time (sec) |
|:-------------:|:-------------:|:-------------:| :-------------:| :-------------:| :-------------:| :-------------:|
| LUV   	| 9      | 8		| 2			| 0			| 96.9% | 96.2 |
| RGB   	| 9      | 8		| 2			| 0			| 95.5% | 65.1 |
| YCrCb   	| 9      | 8		| 2			| 0			| 95.1% | 70.9 |
| HLS   	| 9      | 8		| 2			| 0			| 95.9% | 62.2 |
| YUV  		| 9      | 8		| 2			| ALL		| 98.3% | 124.4 |
| HLS  		| 9      | 8		| 2			| ALL		| 97.5% | 125.2 |
| YUV  		| 12      | 8		| 2			| ALL		| 98.6% | 210.3 |
| YUV  		| 11      | 16		| 2			| ALL		| 98.2% | 154.3 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I first tried using a Linear SVM with default parameters.  The test set accuracy was quite good at 98+% for the parameters chosen, but after running the fitted model on a few of the test images, there were more false positives than I had hoped for as well as a few cases of not detecting vehicles very well.  I decided to use another type of model, RandomForestClassifier to see how it performed on the same features.  I used grid search to test out combinations of parameters (n_estimators, min_samples_split, min_samples_leaf, and max_leaf_nodes) and the highest accuracy (98.7%) was achieved by the following parameters.

{'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 100}

However, the train time was over 2 minutes for this configuration, so I decided to go with these parameters which had an accuracy of 98.6% and a train time of 30 seconds.

{'max_leaf_nodes': None, 'min_samples_leaf': 5, 'min_samples_split': 7, 'n_estimators': 50}

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the `fit_cars` function from the Udacity lectures and specified a couple of ranges and scales to fit the sliding windows.  I started the window detection at y-axis 400, because that is where the horizon intersects with the road and vehicles should not appear above that line.  I implemented three seperate sliding window searches with the following parameters.  I decided on the Y axis ranges and scales based on relative sizes of vehicles that will appear in those ranges (smaller vehicles higher up in the image, larger closer to the bottom).

| Y Start | Y Stop | Scale | Cells per Step
|:-------------:|:-------------:|:-------------:|
| 400	| 500    | 1.0		| 2 |
| 400	| 525    | 1.2		| 2 |
| 400  	| 550     | 1.5		| 2 |
| 450  	| 600     | 1.5		| 2 |
| 450   	| 650      | 2.0		| 2 |
|500 	| 650	| 2.5	| 2 |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are the results across the test images.  In general, the model does a pretty good job of detecting vehicles with few false positives.  One potential problem of note is that in test image 3, only two boxes are identified as cars which could be thresholded out so that no detection occurs.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

My approach was to

In general, the pipeline performs quite well in car detection.  Initially, the detected labels were quite jittery, but I used a similar smoothing technique from the last project to store the detected boxes in a queue from the previous 15 frames (if found at all) and used those to create the heatmap.  There is a big problem with performance though, as processing the full video took a very long time.  For project purposes and using pre-recorded video, this might be acceptable, but for actual usage in real-time, this will not be functional at all.

There are a couple places right off the bat that performance could be improved.  First is by tuning the feature extraction parameters to be more efficient.  Perhaps using all the HOG channels isn't neccesary.  Second, I could try augmenting the data set with additional vehicle/non-vehicle images (using the Udacity data sets for example), which would make the training of the model of more robust.  Because we are working with images here, perhaps using a CNN will be better suited than some of the more traditional models we explored in this project (SVMs, decision trees, etc.).  An ensemble method of using multiple models is also another possibility, which could use multiple models to reduce the variance and provide more consistent results.

Potential problems with this pipeline include vehicles that were not featured in the training set, so trucks, motorcycles, fire engines, etc. will not be detected well because there is little/no data.  Other important random objects will also fail at detection, like a pedestrian, animal, traffic cone, etc.  Shadows and lighting could come into play as different environments could make detection difficult in certain color spaces.
