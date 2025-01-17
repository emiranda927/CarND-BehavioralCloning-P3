# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 83-103) 

The model includes LeakyReLU layers to introduce nonlinearity (code line 87), and the data is normalized in the model using a Keras lambda layer (code line 85).

Using LeakyReLU in place of the normal ReLU activation function was purely an excercise out of curiosity. Upon evaluation of the simulator's performance, it was clear to me that the LeakyReLU provided better overall results than the previous model.

#### 2. Attempts to reduce overfitting in the model

The model contains gaussian dropout and regular dropout layers in order to reduce overfitting (model.py lines 90 & 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 73). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also reversing the driving direction to prevent steering angle bias. I collected two rounds of each and multiple rounds of recovery data providing 5 pools of data to draw from.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple in order to prove the model was working as intended, then build on-top of that for additional feature extraction and performance.

My first step was to use a convolution neural network model similar to the NVIDIA End-to-End Deep Learning Network. This model was appropriate because it was designed with a similar intent to clone human driving behavior with images from three cameras. The NVIDIA Network Architecture consists of 9 layers as shown below (courtesy of NVIDIA):

!["NVIDIA End-to-End Deep Learning Network"](./examples/cnn-architecture-624x890.png)

For the most part, I kept the architecture the same with a few minor tweaks that provided better results. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I did two things: Augment the data set through image preprocessing and add dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. The first few trys were a complete bust where the car drove straight off the track. After recording a few more runs and augmenting the data set, the vehicle performed better but some trouble spots included:
	- The left curve right before the bridge
    - The bridge itself
    - The right curve after the bridge
    - The left curve just before a large dirt path
    
To improve the driving behavior in these cases, I took more recovery data focusing on those areas. Below is an image for one of those trouble spots while collecting recovery data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. My subjective evaluation of the run is that it's incredibly controlled and difficult to distinguish from my own driving. In fact, it's probably a little smoother. There are are one or two instances where it seems like it may begin heading off the road, but it quickly regroups and recovers back to the center of the lane.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer					|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Cropping Layer  		| Cropped image to remove unwanted artifacts	| 
| Normalization	     	| Used Keras Lambda Layer					 	|
| Conv2D + LeakyReLU	| 5x5 kernel, 2x2 strides, 24 filters			|
| Conv2D + LeakyReLU	| 5x5 kernel, 2x2 strides, 36 filters			|
| Gaussian Dropout	    | 50% dropout to prevent overfitting			|
| Conv2D + LeakyReLU	| 5x5 kernel, 2x2 strides, 48 filters			|
| Conv2D + LeakyReLU	| 3x3 kernel, 64 filters						|
| Dropout				| 50% dropout to prevent overfitting			|
| Conv2D + LeakyReLU	| 3x3 kernel,  64 filters						|
| Flatten				| Stack of neurons								|
| Fully Connected		| 100											|
| Fully Connected		| 50											|
| Dropout               | 50% dropout to prevent overfitting            |
| Fully Connected		| 10											|
| Fully Connected		| 1												|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

!["Normal Center Camera Image"](./examples/center_2017_05_03_18_55_43_038.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to increase the steering angle when it detected the edge of the road. These images show what a recovery looks like :

!["Recovery Data"](./examples/left_2017_05_10_21_37_37_468.jpg)

!["Recovery Data"](./examples/center_2017_05_07_17_35_23_974.jpg)

Adding data from track two didn't seem to improve my model. It was sufficient using only data from track 1.

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize better and prevent any sort of lane bias.

After the collection process, I had approximately 87,000 data points. With this amount of data in storage, I began running into memory storage issues on my GPU. After further investigation, I learned that the culprit was of the spike in memory usage was process of casting the array of images as a Numpy array.

```sh
X_train = np.array(flipped_images)
y_train = np.array(flipped_angles)
```
Because the numpy array was not pre-allocated, it effectively needed to store every image in memory while converting. In order to combat this, I used a generator object to train the model, providing smaller batches of data that could be individually released from memory once it was done processing.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used generator objects to provide training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 8 or 9 as evidenced by validation and training loss in the figure below:

!["Model Visualization"](./examples/figure_1.png)

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Overall, the model took a few tries but ended up performing much better than expected. I would like to dive further into this by attempting to merge behavioral cloning with computer vision techniques by creating "boundary zone" masks on the training data to prevent it from ever crossing over the track boundaries.
