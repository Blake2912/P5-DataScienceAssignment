# P5-DataScienceAssignment

## Assignment 1 Notes

### Build the Linear regression and Logistic regression model on the dataset. Tune the parameters. Visualize the results. Measure the model performance using confusion matrix and ROC curve. Conclude with the summary of your findings.

The steps for the implementation are as follows

1. We are first importing all the required module for performing, and then we are importing our datasets downloaded from
the internet, and the data-set which I m using is the data for `Lung Cancer` downloaded from <a href="https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer">Kaggle.com</a>

2. After downloading the data then we are duplicating the data-set into data1.csv and then we are checking various parameters in the data before further processing
	-  While we check the data we are Checking is if the `LUNG_CANCER` param is YES or not, if it is YES then assign the value to 1 else assign the value to 0.
	-  Assign the variable y to data['LUNG_CANCER'] for setting the y axis.
	-  Drop the columns: 'GENDER','LUNG_CANCER','NaN' as these are not required.
	-  Assign the variable X to the remaining data after we have dropped the above mentioned columns.

3. Training the data-set, use the X and y values to train the data set using the function "train_test_spilt()' pass the X and y params and store the X_train X_test y_train y_test for regression process

4. Linear Regression
	- Intialize the model using the LinearRegression() function.
	- Fit the model with the model.fit() where args passed are X_train and y_train.
	- Do the prediction and find the prediction values.

5. Logistic Regression
	- Initialize the model using the LogisticRegression() function.
	- Fit the model with the model.fit() where args passed are X_train and y_train.
	- Perform prediction and find the predicted values.

6. Confusion Matrix
	- cm1 =  Create the confusion matrix for predicted values from LinearRegression
	- cm2 = Create the confusion matrix for predicted values from LogisticRegression
	- Plot the graph using the confusion matrix found!

7. Print the accuracy of the model

8. Plot ROC Curve (Reciever Operating Characterstic Curve) for both the Regression

<hr>

## Assignment 2 (a) Notes
### Apply and understand the Naïve Bayes Classifier and Support Vector Machine (SVM) on spam SMS detection/spam email detection/bot detection. Interpret the results using suitable plots.

The steps for the implementation are as follows:

1. We are first importing all the required module for performing, and then we are importing our datasets downloaded from
the internet, and the data-set which I m using is the data for `Spam Mail Detection` downloaded from <a href="https://www.kaggle.com/datasets/venky73/spam-mails-dataset">Kaggle.com</a>

2. After downloading the dataset, we will first clean the data by removing unwanted columns here those columns are `Id` and `label_num`, and we will be renaming the columns for our convienience.

3. We will then count the dataset by sorting it by label, then draw a bar graph with the resulting data

4. Now we are using Multinomial Bayes Algorithm, As you can see that I have incorporated a recall test and precision test also to access my model more accurately as how much good my model is performing.
Now for different values of alpha, I would make a table to see various measures such as Train Accuracy, Test Accuracy, Test Recall, Test Precision.

5. We will then see the best index for Test Precision

6. We will then be implementing the Random Forest, so we are using the random classifier function, fit the model and train it to calculate the precision

7. Using TensorFlow we will then tokenise the input and then send it for further procession

8. At last we will write the function which will have an input string as its argument and predict whether the string is a spam mail or not

## Assignment 2 (b) Notes
### Build a Random Forest Classifier and KNN to classify the CIFAR-10 images or any suitable image dataset.

The steps for the implementation are as follows:

1. In dataset acquisition we will be using the `CIFAR-10` dataset, it can be easily downloaded with a simple code. The `CIFAR-10` dataset consists of 10 classes namely,airplane, automobile,bird,cat,deer,dog,frog,horse,ship and truck.

2. First we will import the required packages.

3. Then load the `CIFAR-10` dataset, obtain the shape of the train and test datasets.

4. Dataset Pre-processing
	- Normailzation takes place we divide `x_train` by 255.0 for the train dataset and `x_test` by 255.0 for the test dataset.

5. We then train the dataset in sklearn, which creates a 4D Array.

6. Similarly we then test the dataset.

7. Implement the Random Forest Classifier, to classify the CIFAR-10 Images, then we will start training the model

8. The model will return the class of the data ranging from 0-9 with the corresponding class names like
	```
	0 	->	Ariplane
	1	-> 	Automobile
	... so on
	```

9. Then evaluate the accuracy of the model by getting the confusion matrix and classifcation report

10. Implement the KNN i.e. K-Nearest Neighbours and train the model with the CIFAR-10 dataset

11. Then we test this using a custom input, here i took a picture of a bird


<hr>

## Assignment 3 Notes
### Write a Python program to calculate clusters using the Hierarchical Clustering method. Perform Text analysis using topic modeling algorithms.

The steps for the implementation are as follows

<strong>a. Clustering</strong>

1. Import the required Modules

2. The dataset used here is the IPL-2018 Batting stats

3. Then draw the dendogram for the columns of my choice here I have chosen Runs and No

4. Draw the clusters

<strong>b. Text Analysis</strong>

1. Import required modules and clean the data

2. Draw the word cloud

3. Perform LDA

4. Remove all stop words

5. Create a corpus dictionary

6. Train the model

<hr>


## Assignment 4 Notes
### Design a Tableau dashboard on any one of this Sales Forecast Analysis Dashboard/Patient Risk Healthcare Dashboard/COVID Analytics dashboard. Use any suitable dataset to design a dashboard.

The link for the Tableau file can be accessed here <a href="https://public.tableau.com/app/profile/k.c.varun/viz/Assignment4_16620949771540/Dashboard1?publish=yes">Assignment4</a>

<hr>