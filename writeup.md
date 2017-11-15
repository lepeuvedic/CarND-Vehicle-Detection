

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
[scales1]: ./output_images/car_scales.png
[scales2]: ./output_images/car_scales2.png
[vehicle]: ./examples/310.png
[nonvehicle]: ./output_images/false_pos_test6.jpg
[classb]:  ./examples/Class_B.png
[occlud1]: ./examples/381.png
[occlud2]: ./examples/562.png
[hog]: ./examples/hog.png
[scale15]: ./output_images/scale1.5.jpg
[scale03]: ./output_images/scale0.3.jpg
[test1]: ./output_images/test1.jpg
[test1hm]: ./output_images/test1_heatmap.jpg
[test5]: ./output_images/test5.jpg
[test2]: ./output_images/test2.jpg
[negative]: ./non-vehicles/neg-mining/test5-008.jpg
[video]: ./output_images/project_video_v4.mp4
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Overall impression

Reusing the strong infrastructure from the Advanced Lane Lines project, the overall impression was that the level of difficulty of this project was lower. Going through the project walkthrough video where Ryan tries to ensure that we start in the good direction certainly helped to save a lot of time, but the final code bears little resemblance to his, except in functions directly copied from the course, like `get_hog_features()`. I fairly quickly reached the stage where boxes are added to the test video.

The starting point however had the following issues:
* It uses a **huge** amount of RAM to train the classifier: it was not practically possible to train on the 200,000 image strong Udacity data set;
* It trains on a data set containing pictures of cars, then considers normal to match on parts of cars in the video, without any consideration of this strange, but convenient behavior;
* The sliding window function fails to reach the last position in the scanned area;
* Heatmaps can be scaled down a lot, for memory efficiency and many optimizations are possible;
* All the data sets have time-series issues.

When the black car passes ours, the correct scale for matching it is 4 (a 256x256 pixel patch). This implementation of the project matches at scales 4, 2, 1.5 and 1, scale 1 being nominal 64x64 pixel windows. Smaller scales 0.5 and 0.3 are defined but tend to generate a lot of false positives if they are used indiscriminately. They are not currently in use. Even though there is no specific reason for that, the classifier matches the distant car in the video too rarely for that to be of any use. In addition the number of windows to test increases quickly when smaller windows sizes are introduced, even with a restricted search area.

Conversely adding large scales adds few windows, and does not significantly impact performance.

The images below illustrate the most useful scales:

![scales 4 and 1.5][scales2]

![scales 4 and 2][scales1]

The code base is split in two parts:
* The experimentation in the IPython notebook `Experimentation.ipynb`;
* The operational software in `RoadImage.find_cars()` and all its sub-functions.

The last cell of the IPython notebook calls `RoadImage` to process the videos. All the cells before that one contribute to examining the data set, training the classifier or even prototyping the algorithms.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third and fourth code cells of the IPython notebook. The 3rd cell contains functions which extract the features from a single image. `get_hot_features()` (line 14), `bin_spatial()` (line 33) and `color_hist()` (line 42) are directly copied from the course. As suggested in the walkthrough, the color histogram bins are determined automatically based on the range of color value.

At line 54 the function `feature_length()` calculates the length of feature vectors. This function is called from the next code cell to preallocate storage for the feature vectors. Finally at line 73, the function `€xtract_features()` is only slightly modified. In my version, it is protected against incorrect image files, which could be non-existing, unreadable or in some cases which are completely black and white (no color information). The need to guard against incorrect image arose when I started working with the second Udacity data set.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![vehicle][vehicle]
![non-vehicle][nonvehicle]

The non-vehicle image is a false positive that I added to the non-vehicles data set myself. More later.

To try out different settings for the basic setting of `skimage.feature.hog()`, I used the parameter `vis` of the function `get_hog_features()` in order to generate the rendering of the histogram of oriented gradient for selected images. A selection of images can be displayed in the third code cell of the notebook.

I actually played with all the parameters of the feature extraction together using the classifier as a trend indicator. At line 7 in the fourth code cell the function `read_image_list_from_file()` has an argument `step` which can be set to values greater than 1 to skip images. It becomes possible to work quickly on subsets of the training data set.

Lines 24 to 30 show the not very original finally selected parameters. At the end of the cell, the trained cclassifier is saved, and I have saved a small number of well performing classifiers. If the corresponding settings are programmed in the state variable of the `RoadImage.find_cars()` method, all those classifiers can be used.


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `hog_channel=1` and `cells_per_block=(2, 2)`:


![From car image to HOG rendering][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored the influence of each parameter in turn, increasing, then decreasing its value until I found an optimum value. The poorest results from the classifier were around 80%, similar to results achieved without using the HOG feature. The optimal number of orientations was 15, but the number of orientations multiplied by 588 gives the number of features from the HOG, and going from 9 to 15 increased the length of the features vector by 40%. Given the tiny performance difference, and the need to use as many images as possible from the data set to prevent overfitting, I chose to stick with 9, which is the compromise value for which the original authors of this approach noted tapering performance improvements.

The method is not entirely new for me, since my master's thesis (back in 1993) used histograms of orientations to match handwritten uppercase characters. I used 8 orientations based on a biomimetic approach of the human eye.

After consideration of its definition, the `cell_per_block` values which made sense were 1, 2 or 3. A value of 1 led to a visible degradation, effectively loosing gradient normalization across neighbouring cells. A value of 3 led to a large increase in the feature vector size without any visible improvement (the HOG feature vector length is proportional to the square of that parameter, so going for 2 to 3 more than doubles its lenth). At the end, 2 is the only reasonable value for this parameter.

There is more flexibility for `pix_per_cell`, but the choice is a tradeoff between the number of points in the histogram, so the accuracy of the histogram, and the resolution of the hog map itself. The hog image must have a sufficient resolution to capture gradient orientation changes in and around the image of a car.

I tested the classifier in RGB, HSV and YCrCb colorspace. I wasn't expecting good results with RGB, because a red car and a green car might exhibit similar patterns, but in distinct regions of their feature vectors. In RGB it would probably be necessary to train on a very wide variety of car colors. 

I must admit that time limits on term 1 limited my investigations to what was needed to get a sense of the influence of each parameter, and to make sure that I worked with a sensible set of parameters. Re-training the classifier takes some time (almost equivalent to processing the video).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the fourth code cell (Step 4) while I explored the possible parameter settings. I have a collection of pre-trained classifiers and scalers, which are all variants of each other:
* Spatial binning 15x15, 16x16 or 32x32;
* Use of Udacity2 data set for vehicles, or the default project data (or both, with subsampling);
* Use of HSV or YCC colorspaces.

After having issue with detection I switched back to the project default data set, and ceased to use the Udacity vehicles. It is not just the GTI data set, that has time series issues (more or less the same view of the same vehicle over and over).

I have noticed an abnormally large representation of mountain gray class B Mercedes in the KITTI images. As the car is always shot from behind, I suspect that the car with the camera was following it. About one image out of twenty is a picture of that same car over hundreds of images. As examples, between images 550 and image 700, there are 9 images of that car. I used to own one, and it's a nice car, but it's not good for the classifier to see it too much.

![Lots of Mercedes shots][classb]

The dataset also contains occluded cars, as in the two examples below. The second is actually occluded by a sign or a stand which masks it almost entirely from view.

![Car hidding behind a car][occlud1]
![Car masked by something][occlud2]

I used the color spatial binning and histogram in addition to HOG features. There was a visible difference going from small spatially binned features 15x15 or 16x16 to 32x32 (using 4 times the RAM but starting from just 256 features it's OK).

The classifier is trained at line 95 in that code cell (Step 4), and both the trained classifier and the scaler are saved in a pickle file. It is much longer to extract all the feature vectors than to train the linear SVM. It is interesting to note that when the feature vector is not scaled (zero mean, unit variance), not only is the performance worse, but the Linear SVM also takes a much longer time to run.

In order to reduce the number of false positives, an early version of the `draw_boxes()` function saved each detected 64x64 pixel patch to disk (line 38 in the seventh code cell). Since I had many false positives, I manually examined the results, sorted the false positives out and added them to the 'non-vehicles' data under a new folder named `neg-mining` (which is included in the Github repository). The number of images does not seem very high, but running the re-trained classifier of the same images after adding those training examples, I reduced the number of false positives from 25 down to 20. Many false-positives re-occurred anyway.

This technique is called negative hard mining. It is probably even more effective when practiced on a larger scale, the limit being the manual classification of matches as correct or false positives.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window code, like all the final image processing code, is in the `RoadImage.find_cars()` method at line 3362.

![RoadImage.py](classes/RoadImage.py#3362)

The sub-function `find_at_scale()` is defined inside the `find_cars()` method, because it is not part of the public interface of `class RoadImage`. It takes as input the current video frame, the search area, the scale which corresponds to the search area, the blockmap (more later), the classifier and the scaler, and finally the parameters for feature extraction.

The search area is a rectangle, which is always scanned with a 75% overlap between successive windows.

I just kept the overlap the same as in the project walkthrough. It is clear that an overlap is useful, because otherwise we might lose track of objects which temporarily appear on the boundary of two positions. A large overlap also gives a better granularity to successive detections.

The search positions are illustrated for scale 1.5 (96x96 windows on the camera image) on the following figure, which was constructed by temporarily forcing the test `if test_prediction == 1:` to `if True:`. In this way, all the tested positions were matches and the windows were drawn.

![The sliding window algorithm searches those positions][scale15]

For the largest scales, only two lines of window positions crossing the whole image are searched. For smaller scales, typically 3 lines are searched below a pseudo horizon at line 400. The global search zone is therefore 1.5 times the search window height (considering overlap of 75% of each line). This choice was made because we are looking for smaller car images closer to the horizon, so it made sense to only apply the smaller scales to the areas of the image where distant sections of the highway are visible. The smallest defined (but not used) 0.3 scale or 20x20 windows, has a search area extending over half of the image, in its central portion.

![The smallest defined, but not used, search area][scale03]

For the long video, I took into account the information that the car which is filming, is on the leftmost lane of the highway. To increase the speed, only the right half of one of the search areas is used. The search areas used for the video are defined in the very last cell of the notebook. All the parameters of the algorithm are tunables, which can be modified without changing the code in `RoadImage`. The final parameters are the tunable setting of the notebook, and may not be reflected 100% in the default settings found in `RoadImage.py` at lines 3506 to 3537.

It would have been an interesting test, to try to move the whole search pattern slightly at every frame, because it would help define a tight and stable boundary around object. This approach is not compatible with another choice, which is implemented instead: scaled heatmaps.

On one hand, it is very visible on the illustrations above, that the search defines a grid-like pattern. On the other hand, the principle of the heatmap is the add 1 (typically) to the interior of each window in which a car is detected. At large scales, those areas to manipulate are large : 65536 pixels at scale 4. But the information will always be the same inside each area defined by the grid pattern. Whatever the scale, the heatmap inside a window with 75% overlap, can always be encoded using a 4x4 pixel image. When they cover the declared search area, those images are the scaled heatmaps. They are initialized the first time a scale is used at line 3582.

Heatmaps are part of the state information which is preserved from frame to frame, using a global variable attached to the `find_cars()` method. If possible, the method will use an attribute instead, allowing multiple video streams to be processed at the same time with distinct state data. In this case, it was not possible because the method is called as a callback from `clip.write_videofile()`, and gets a fresh `RoadImage` instance at each frame. There is not way to pass additional date in this (poorly designed) `VideoFileClip` interface, although it would be possible to try thread local storage as well.

In my implementation, the detected windows are stored in a deque operated as a FIFO queue. The detection boxes are added to the heatmap as they enter the queue, and substracted from the heatmap when they are removed from the queue and discarded. The queue depth is a tunable parameter. The video was processed with a queue depth of 10 frames.

It did not make much sense to add detections at multiple scales to the same heatmap. There are two reasons for this opinion: first, the smaller scales have also denser grids (smaller step in pixel), so in a given area containing a car, they will automatically have more opportunities to match part of that car than windows of a larger scale. The smaller scales therefore have an implicit higher weight. Secondly, when there are multiple cars, some can match at one scale which is large enough to allow only one match per car, and others may match at multiple scales, vastly increasing their temperature on the heatmap. When the heatmap is thresholded to eliminate false positives, the strong contrast caused by a minor difference in apparent car size causes the less bright car to disappear. This problem is solved when distinct heatmaps are maintained for each scale, but we have another problem instead: the same car can be detected at multiple scales.

The code handles this using the blockmaps. The blockmaps both limit the case of multiple detection of the same car at different scales, and are a speed optimisation (maybe...). Assuming the slowest operation is HOG feature extraction, the test at line 3371 skips the search position in the sliding window algorithm, if that search position has more than 50% intersection with the blockmap.

The blockmap is not scaled, and is not cumulative. It is boolean and contains True in the areas where cars have already been found at a larger scale (of course, even though that's tunable as well, it is a good idea to search the scales from the largest to the smallest using the `scan_scales` tunable if the blockmaps are going to be used).

There are actually two blockmaps: the blockmap called `blockmap` is zeroed at every scale, and only contains the areas matched at the next larger scale, which was processed just before in the loop starting at line 3601. This second blockmap is zeroed at line 3687. The cumulative blockmap `cumulative_bm` is only initialized once per video frame, and is the one used in `find_at_scale()` to avoid detecting the same car at multiple scales. `blockmap` is part of mechanism to eliminate false positives (see below).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The seventh cell contains a set of prototype functions: the first working pipeline. The useful part of the code cell starts at line 198, where it loads the classifier, the scaler and the camera calibration from disk. All the files provided in the folder `test_images` were used for testing. The full results are available in the folder `output_images` under the same file name (for images annotated with boxes) and the corresponding heatmaps.

At line 228, the code now runs the `find_cars()` method, which normally confirms car detections for a significant number of frames before it draws any red box. The tunable parameter `new_car` is set to 1, to make it flag cars the first time they are seen. The tunable `draw_boxes` asks it to draw all the boxes (in blue) resulting from the analysis of heatmaps at all the scales. The default 4 scales and associated search areas are used.

Please note that the heatmap shown here is a composite of the heatmaps at all the scales made specifically when the method argument `vis_heatmap` is True (or when debugging using method `track()`). It is built at line 3736 by resizing the scaled heatmaps to image size. The default RoadImage resize method tends to smooth the contour, and an effort of imagination is necessary to intepret the heatmap as it really is, with clear edges. This rendering of the heatmap is not actually used by the algorithm.

In the case of the heatmap shown here, only scale 1.5 contributes, with several detections forming a rectange on the white car.

![Annotated test1 image showing 96 pixel high red boxes around both cars][test1]
![and the corresponding heatmap][test1hm]
![Annotated test5 showing a detection at scale 4 : 256 pixels][test5]
![Annotated test2 showing no detections when there no cars][test2]

The last image show that when there are no cars in view, there are no detections. The corresponding heat map is completely black. The image of test5 illustrates a detection at scale 4 with a 256x256 pixel window, which looks too large for the black car. There was a single match on the black car, and a single match at scale 2 on the white car. This explains why the two areas are perfect squares.

Although scale 4 is theoretically what we need to capture the black car as it passes ours, it looks like some intermediate scale would be needed to avoid such large boxes around cars which are a bit farther.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)

The vehicle is correctly tracked, including when the black and white cars overlap, then separate. The red rectangles are detected cars. The thinner blue rectangles are raw outputs from the heatmaps and illustrate all the false-positives which are eliminated by the algorithms.

There are two false positives of short duration. The first at 0:09 matches a car-shaped (?) small tree on the road side and the second is not really a false positive, but a match on the front bumper and wheel of the white car in the last seconds of the video. This is behaviour was apparent in the classifier from the very beginning and the negative examples I added include that one, but it is not enough to avoid this match.

![Negative example I added][negative]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The raw window matches are combined in a heatmap dedicated to each scale at which the algorithm runs. As currently configured for the video, the algorithm runs at scales 1.5, 1. and 0.5 (lines 29-30 in the last code cell). This is performed in the sub-function `find_at_scale()` which returns a list of raw window matches. 

At line 3607, the loop on scales updates the heatmap. It first checks if the old boxes FIFO is full, and if it's the case, it substracts the oldest boxes from the heatmap, providing a decay of old detections. Then it applies the new detections and `apply_heat()` is requested to perform box detection (it is the default), and returns a list of boxes representing **tracks**, not cars yet.

When there are no cars, random window matches will be remembered in the heat maps until the boxes are removed from the deque and discarded. If the depth of the deque is 10, like it is for the video, a random match with be detected 10 times, unless another object at the same scale quickly makes another area of the heatmap very bright. If there is no car in sight, thresholding, which takes place at line 3420, is not effective.

In `apply_heat()`, the non-cumulative blockmap is used to prevent reporting of tracks, which have already been reported at the previous scale. This logic eliminates cases where the same car is counted as two cars as it comes closer to the camera and matches at different scales. Of course, there is already logic in the sliding windows algorithm which disables search locally in areas where a match is found at a larger scale, but if a car is getting closer to the camera it already has a very bright image in the heatmap of the smaller scale, and only a very dim image at the larger scale. Without the non-cumulative blockmap, `apply_heat()` would continue to report the car at the smaller scale for many frames, and we would have two cars recorded where there is only one which is getting closer. This behaviour was observed on the white car as the car carrying the camera overtakes it near the end of the video, but no longer occurs.

My implementation of `apply_heat()` uses `scipy.label` to identify topologically disconnected blobs in heatmaps. Equating the resulting array with a label number gives a boolean array with only one box in it. The steps used at line 3426 to compute the coordinates of the bounding box seem slightly more efficient than relying on `np.nonzero()` and making lists of X and Y coordinates of all the pixels inside the box. Remembering that my scaled heatmaps have only 5 or 6 lines I take sums along those lines. Zeroes indicate if the matching window is on the first, second or third line. Isolating the nonzero lines (normally 4, for 1 window), the mean value gives access to x2-x1, the width of the box. x1 is found by the analysis of a single line in chosen to be in the middle of the box. There is only one 2D operation acting on a tiny scaled heatmap, and the rest are only 1D ops, one along the height of the heatmap, one along the width. The location of the first non zero element would even be faster to find, since `argmax` has to scan the whole line, but it is faster to use a C++ function doing approximately the right thing, than to implement the right thing with a loop in Python. The fastest implementation might use a bissection approach to first locate a True element, and then to locate the False to True boundary. Let's keep in mind that `area` is a small numpy array of boolean values.

The main mecanism used to prevent false positives is confirmation. A track is converted into a car only after it has reached 25 detections. We have seen above that random blips only get recorded in the heatmap for 10 frames (the depth of the old boxes queue), so in order to reach 25 detections, a car must be seen on multiple frames.

The logic which converts tracks into cars begins after the heatmap updates at line 3626. It first looks in `state.vehicles`, where all the tracks and their history is kept, if it knows tracks, or cars nearby. For this problem, only the x coordinate is used to measure proximity. The known cars are tracks which have been seen at last `state.new_car` times. All the tracks younger than that are unconfirmed, and may be noise.

The logic varies according to the quantity and status of tracks found nearby. If the algorithm is already tracking multiple nearby cars (cars whose x position places them in the currently considered box), we check if they have been seen recently. If those cars are frequent hit, we add the box to a list of boxes to discard, because we will attempt to see those multiple cars at a lower scale, which hopefully will allow us to discriminate between those multiple cars.

If there are multiple cars, but they no longer trigger detections at any scales, we increase the views counter on all the immature tracks we have locally. The reason for this, is that the box we are considering might be any of those young objects or a combination of them. We count on statistics, frame after frame to make a winner emerge from multiple initialized tracks in the same region. The tracks which match more frequently will end up on top, and because priority is given to a continuous tracking of known cars, once one of the tracks reaches this status, the others will die off.

If there are no cars and no tracks nearby, the logic creates a new track, unless the scale is already very small (less than 1) because those small scale generate lots and lots of false-positives. The only way to track a car at scale 0.5, is to begin tracking it at a larger scale.

If there are known cars nearby, but a mix of scales, we sort them using age of last detection as the primary key, scale difference as the first secondary key and distance to the center of the considered box as the second secondary key. Only the first ranked car is "seen".

If we have found exactly one known car close to the considered box, we associate them together. The algorithm remembers that it has seen that car in that frame (see function `seen_car()`). Otherwise, if we have no confirmed car nearby, we mark all the neighboring unconfirmed tracks as 'seen' counting on context change, andmotion relative to the search grids, to select the winner.

Finally if we have nothing recorded nearby and the scale is not very small, we create a new track.

The relatively high confirmation count (25) is the most effective measure against false-positives. The drawback is the visible latency before a new car gets identified as such.

Car processing continues on line 3707. For all the cars which have not been seen in the current frame, the number of consecutive detections is zeroed, and the number of consecutive nondetections is increased. At line 3714, the logic decides whether the car is gone forever. In the case of random noise, the value in stop tracking takes effect once the detection has traversed the FIFO of the heatmap.

Finally red boxes are drawn around the remaining cars, and in the small loop which begins on line 3727, the cars which are gone forever are removed from `state.vehicles`. The box which is drawn is the average value of the 10 most recently detected boxes for that car, which are stored in a fixed depth FIFO. This FIFO also serves to compute the position of a track, which is used to find cars or tracks which are inside a new box. 

In the folder `output_images`, the video `project_video_v4.mp4`, shows the project video with search zones at scale 1 and 0.5 positioned slightly higher. There is one additional false positive on the overhead traffic sign of the opposite lane.

![video with tracks and cars][video]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main weakness of the pipeline is the primary detection by the classifier. It is slow and only as reliable as the data sets. If I had more time, I would investigate a Deep Neural Network approach like Single Shot Detector, because it can take advantage of the whole data set for training. Despite the large quantity of vehicle pictures, it was only possible to use 10 thousand for training the classifier, because those classifiers must receive all the training data in a single batch.

A better computer with more memory would be able to use more data to train the classifier, but a deep neural network would train iteratively and would not need so much memory to use the entire data set.

The car tracking logic seems purpose built, and unlikely to survive in more complex tracking / data fusion problems.
