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
[image2]: ./writeup_imgs/hog.png
[image4]: ./writeup_imgs/windows_rf.png
[image3]: ./writeup_imgs/heatmap.png
[image21]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./writeup_imgs/labels.png
[image7]: ./writeup_imgs/detection.png
[video1]: ./project_video_out.mp4

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook `vehicle_detection.ipynb` in the `get_hog_features` function.  This function is generally called by the `extract_features` function which also calculates spatial and histogram features.

I started by reading in all the `vehicle` and `non-vehicle` images.  There are 8792 vehicles and 8968 non-vehicles in the dataset, so it is balanced.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` just to get a visual idea of what HOG is computing:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters through mostly trial and error looking for a high accuracy on the test set (after a 80/20 train/test split).  A chart of various combinations of parameters is shown below along with accuracy scores and feature extraction times.  Adding in 'ALL' HOG channels does increase accuracy, but doubles the extraction time of just a single channel.  This could come into play if efficiency becomes an issue.  The bolded row is the parameters that I chose.

| Color Space| Orient| Pix/Cell | Cell/Block | HOG Channel | Accuracy | Extraction Time (sec) |
|:-------------:|:-------------:|:-------------:| :-------------:| :-------------:| :-------------:| :-------------:|
| LUV   	| 9      | 8		| 2			| 0			| 96.2% | 96.2 |
| YUV   	| 9      | 8		| 2			| 0			| 96.4% | 91.7 |
| RGB   	| 9      | 8		| 2			| 0			| 95.5% | 65.1 |
| YCrCb   	| 9      | 8		| 2			| 0			| 95.1% | 70.9 |
| HLS   	| 9      | 8		| 2			| 0			| 95.9% | 62.2 |
| **YUV**  		| **9**      | **8**		| **2**			| **ALL**		| **98.3%** | **124.4** |
| HLS  		| 9      | 8		| 2			| ALL		| 97.5% | 125.2 |
| YUV  		| 12      | 8		| 2			| ALL		| 98.6% | 210.3 |
| YUV  		| 11      | 16		| 2			| ALL		| 98.2% | 154.3 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After choosing the HOG parameters, I used the entire dataset, used a StandardScalar to normalize, and fed it into a grid search for a LinearSVC model using a 3 fold cross validation.  I tried different values of C as well as the loss function, and found the best performer was C=0.01 and loss='hinge'.  However, after running the fitted model with these parameters on a few of the test images, there were more false positives than I had hoped for as well as a few cases of not detecting vehicles very well.

I decided to try another type of model, RandomForestClassifier to see how it performed on the same dataset.  I used grid search to test out combinations of parameters (n_estimators, min_samples_split, min_samples_leaf) and the highest accuracy (98.7%) was achieved by the following parameters.  Results were much more consistent on the test images.

{'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 100}

However, the train time was over 2 minutes for this configuration, so I decided to go with these parameters which had an accuracy of 98.6% and a train time of 75 seconds.

**{'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 50}**

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the `fit_cars` function from the Udacity lectures and specified a couple of ranges and scales to fit the sliding windows.  I started the window detection at y-axis 400, because that is where the horizon intersects with the road and vehicles should not appear above that line.  I implemented three seperate sliding window searches with the following parameters.  I decided on the Y axis ranges and scales based on relative sizes of vehicles that will appear in those ranges (smaller vehicles higher up in the image, larger closer to the bottom).

| Y Start | Y Stop | Scale | Cells per Step
|:-------------:|:-------------:|:-------------:|
| 400	| 500    | 0.8		| 2 |
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

I aggregated all the results from the `find_cars` function running multiple times with different scales and y start/stops into an array of boxes (positive detections).

I used a similar smoothing technique from the last project to store the detected boxes in a queue from the previous 15 frames (if found at all) and used those to create the heatmap.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and weed out some of the false positives.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a test image with its corresponding heatmap:

![alt text][image3]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the test image:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the test image:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to initially adapt the code from the lectures into a general vehicle detection pipeline just as a sanity check.  Once I achieved this, I began tweaking parameters and exploring different classifier models.

I found that parameter tweaking was quite difficult and cumbersome, as it was more or less trial and error and feature extraction took a decent amount of time.  I seemed to run into issues where the LinearSVC model performed well on the test set (98+% accuracy), but this success did not translate on the test images or the test video where detection was spotty and there were quite a few false positives.  Moving to Random Forests helped this significantly, but it seems like the prediction time jumps quite a bit which isn't good for performance.

In general, the pipeline performs decently in car detection.  There are some issues with detection in the opposing lane causing erroroneous boxes, but the main vehicles in the lane of the camera car perform well.  Initially, the detected labels were quite jittery, but I used a similar smoothing technique from the last project to store the detected boxes in a queue from the previous 15 frames (if found at all) and used those to create the heatmap.  There is a big problem with performance though, as processing the full video took a very long time.  For project purposes and using pre-recorded video, this might be acceptable, but for actual usage in real-time, this will not be functional at all.

There are a couple places right off the bat that performance could be improved.
1. First is by tuning the feature extraction parameters to be more efficient.  Perhaps using all the HOG channels isn't neccesary or I could try a higher pixel per cell.  I could also try alternative methods of HOG extraction, such as using OpenCV's HogDescriptor which is supposedly much faster.
2. I could try augmenting the data set with additional vehicle/non-vehicle images (using the Udacity data sets for example), which would make the model itself more robust.  Random forests works well, but it is a major slowdown for prediction compared to a LinearSVC, so if the data was more robust, perhaps a lighter model could be used (or use different model parameters).
3. Because we are working with images here, perhaps using a CNN will be better suited than some of the more traditional classification models we explored in this project (SVMs, decision trees, etc.).
4. An ensemble method of using multiple models is also another possibility, which could use multiple models to reduce the variance and provide more consistent results.

Other than the big performance issue, potential problems with this pipeline include vehicles that were not featured in the training set, so trucks, motorcycles, fire engines, etc. will not be detected well because there is little/no training relevant data.  Other important random objects will also fail at detection, like a pedestrian, animal, traffic cone, etc.  Shadows and lighting could come into play as different environments could make detection difficult in certain color spaces.
