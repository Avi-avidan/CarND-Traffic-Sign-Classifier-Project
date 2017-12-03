# **Traffic Sign Recognition** 

## Writeup

---

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/all_classes.png "one image of each class"
[image2]: ./figures/all_datasets.png "five images of each dataset"
[image3]: ./figures/augmentation1.png "augmantation example1"
[image4]: ./figures/augmentation2.png "augmantation example2"
[image5]: ./figures/train_set_class_distribution.png "training set distribution"
[image6]: ./figures/augmented_set_class_distribution.png "augmented set distribution"
[image7]: ./figures/top_5_predictions.png "top 5 predictions bar chart"
[image8]: ./figures/final_test.png "final test on 5 unseen images"
[image9]: ./figures/classification_report.png "classification report"
[image10]: ./figures/LeNet.png "Le-Net Architechture"

## Rubric Points 

---
### Writeup / README

#### 1. link to my [project code](https://github.com/Avi-avidan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_20171022.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set:

loaded 34799 train samples & 34799 train labels
loaded 4410 validation samples & 4410 validation labels
loaded 12630 test samples & 12630 test labels
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

one image of each class is visualized -
![alt text][image1]

5 random images visualized for each data set -
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Main preprocess & techniques used

distribution of classes in the training set was evaluated -
![alt text][image5]

each classes was further populated to create even distribution accross all classes.
![alt text][image6]

population of under represented classes was carried by random roatation & random cropping.
augmantation example 1 -
![alt text][image3]
augmantation example 1 -
![alt text][image4]

lastly, image data was normalized to achieve quicker & more efficient optimization.


#### 2. model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution1 5x5     	| 1x1 stride, VALID padding ==> 28x28x6         |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| Convolution2 5x5	    | 1x1 stride, VALID padding ==> 10x10x16        |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| flatten (400)        							|
| Fully connected		| 400 ==> 120        							|
| RELU					|												|
| Fully connected		| 120 ==> 84        							|
| RELU					|												|
| Fully connected		| 84 ==> 43        								|
| RELU					|												|



#### 3. training the model

After testing various options, I have used Adam optimizer with learning rate of 0.0005.
tained the model using my augmented dataset (total of 86430 samples - original + augmented) for 10 EPOCHs with batch size of 40.
at this point, results are nice but not brilliant - 
INFO:tensorflow:Restoring parameters from ./trained_models/lenet_0.896
Test Accuracy = 0.870
Validation Accuracy = 0.896

#### 4. further improvement

Further improvement was achived by training for additional 5 EPOCHs on the [German Traffic Sign Dataset] (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
this dataset was split 0.75:0.25 for training and testing respectively -
loaded 29406 samples & 29406 labels for additional training
loaded 9803 test samples & 9803 labels for testing

Results at this point -
INFO:tensorflow:Restoring parameters from ./trained_models/lenet_0.989
Original Test Accuracy = 0.916
GTRSB Test Accuracy = 0.992
Validation Accuracy = 0.990

#### Some comments over chosen architechture - 
* LeNet was chosen as a good starting point (didnt see any need to temper much was a proven model)
* I did most of the adjustments to the data set (significant augmantation to create an evenly distributed dataset)
* learning rate > 0.0005 as well as batch size < 40 resulted in a bumpy accuracy results scheme
* over fitting was apperent and very significant when training for more than 20 EPOCHs.
* test set was only used once. 
* No iterative approach implemented.
* dropout layers might be useful to allow further training. However, since great performance was achieved using very limited training, the additional benefit of dropout was negligible / irrelevant.
![alt text][image10]


### Test a Model on New Images

#### 1. five German traffic signs were loaded to the model. great results for all. 
(labels were added manually for comparison with prediction)
![alt text][image8]

#### 2. Discussion 

The model was able to correctly predict 5 of the 5 traffic signs, which gives an accuracy of 100%. of course, this 'test' is far limited to be considered significant. Results are consistant with expectations. all in all very fun project :)

#### 3. top 5 softmax probabilities for each image along with the sign type of each probability

for all images (I actually loaded much more than 5 to see if I could find error ;), confidence level as calculated by softmax, seems very high for most of the images.

![alt text][image9]



#### (I did not visualized feature maps)


# CarND-Traffic-Sign-Classifier-Project
