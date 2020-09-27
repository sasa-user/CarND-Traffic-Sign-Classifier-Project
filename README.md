# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/visualization.png "Visualization"
[image2]: ./writeup_imgs/origin_img.png "Original img"
[image3]: ./writeup_imgs/processed_img.png "Preprocessed img"
[image4]: ./new_traffic_signs/img1.jpg "Traffic Sign 1"
[image5]: ./new_traffic_signs/img2.jpg "Traffic Sign 2"
[image6]: ./new_traffic_signs/img3.jpg "Traffic Sign 3"
[image7]: ./new_traffic_signs/img4.jpg "Traffic Sign 4"
[image8]: ./new_traffic_signs/img5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

[Link to data set.](https://drive.google.com/file/d/1Fufvl-dEmwUyQL6KA9xBu7pVf-MtXNLC/view?usp=sharing)

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed among classes:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it's simple and effective way to reduce noise and reduce training time.

As for the second step I normalized the data as suggested, and as I learned it is good for learnability and accuracy as reduces over and under compesation of a corrections in some weights.
I was playing around with ranges and decided to use -0.3 - 0.3 as I was getting noticeable better results then the original -1 to 1. Data set mean was, in that way, reduced to ~ -0.1.

Before preprocessing:
![alt text][image2]

After preprocessing:
![alt text][image3]

Reducing noise and improving accuracy with these two steps made great difference when training the network.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grey image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x50 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x50 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x100  				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x4x250 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x250  				|
| Flatten       		| input 2x2x250, outputs 1000 					|
| Fully connected		| Input = 1000, Output = 120					|
| RELU					|												|
| Fully connected		| Input = 120, Output = 84  					|
| RELU					|												|
| Fully connected		| Input = 84, Output = 43   					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used default set of hyperparameters with exception of changing the number of epochs to 30. So the batch size is 128, mu 0, sigma 0.1. Other changes I made in LeNet architecture.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.945%
* validation set accuracy of ~0.962% 
* test set accuracy of ~94.378%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
- First architecture that I tried was LeNet, it's a complete architecture, that we have implemented during the lessons so, and even suggested as a starting point.
* What were some problems with the initial architecture?
- Initial problem was under fitting, as I had ~90% accuracy with the validation set. So, I needed to improve.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
- Well, as I sad I was looking into fixing under fitting and improving accuracy and I was going for better shape detection, that's why my first choise was convolutional layer. But, I was guessing. LeNet architecture was adjusted with adding additional convolution layer (third) and with that reLu and pooling and adjusting the parameters of next layer to match. The rest was the same as original LeNet. These changes improved accuracy by 5-6%, which was nice. I tried to go further but I had decrease in accuracy.
* Which parameters were tuned? How were they adjusted and why?
- First thing was number of epochs, before changing anything else. It just made sense to try training more times.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
- I wrote in point above about my adding of the third convolutional layer, and what was the thinking. Dropout would be a strategy for over fitting, with which I didn't have problem. I think.

If a well known architecture was chosen:
* What architecture was chosen?
- Covered in the section above. - I am using major part of LeNet.
* Why did you believe it would be relevant to the traffic sign application?
- Covered in the section above. Influence of the lessons. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
- Well, since those are different sets, confirming accuracy against this sets is good enough information.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| 30 km/h     			| 30 km/h 										|
| Road work 			| Beware of ice/snow 							|
| Stop sign	      		| Stop sign 					 				|
| 70 km/h    			| 70 km/h           							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ~94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second to last cell of the Ipython notebook.

For the first image, the model is very sure that this is a No entry sign (probability of 1, the rest almost 0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~ 1         			| No entry   									| 
| 3.87047708e-16		| Ahead only 									|
| 3.19264453e-17		| Go straight or left							|
| 6.31305236e-18		| Stop      					 				|
| 2.19108124e-19	    | No passing        							|


For the second image, the model is very sure that this is a 30 km/h sign (probability of 1, the rest almost 0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~ 1         			| 30km/h    									| 
| 2.69519897e-19 		| 20km/h    									|
| 7.42671189e-26		| 80km/h          								|
| 1.24297337e-30		| 50km/h      					 				|
| 1.79615184e-32	    | Roundabout mandatory   						|

For the third image, the model is very sure that this is a Beware of ice/snow sign (probability of 0.95). That was wrong. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.95         			| Beware of ice/snow    						| 
| 0.035 				| Road work   									|
| 0.01					| Right-of-way at the next intersection  		|
| 0.0005				| Road narrows on the right 	 				|
| 0.0005			    | Pedestrians      								|

For the fourth image, the model is very sure that this is a Stop sign (probability of .97). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.97         			| Stop sign    									| 
| 0.02 					| Go straight or left   						|
| 0.0055				| Keep left          							|
| 0.0011				| 60km/h      					 				|
| 0.0008			    | Turn left ahead  								|

For the fifth image, the model is very sure that this is a 70 km/h sign (probability of 1, the rest almost 0). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~ 1         			| 70 km/h   									| 
| 6.33098662e-13		| 20 km/h 										|
| 1.20018999e-15		| 30 km/h										|
| 2.03776745e-19		| General caution      		 					|
| 3.48570356e-20	    | Road narrows on the right    					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


