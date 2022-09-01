# P5-DataScienceAssignment

## Assignment 1 Notes

### Build the Linear regression and Logistic regression model on the dataset. Tune the parameters. Visualize the results. Measure the model performance using confusion matrix and ROC curve. Conclude with the summary of your findings.

The steps for the implementation are as follows

1. We are first importing all the required module for performing, and then we are importing our datasets downloaded from
the internet, and the data-set which I m using is the data for `Lung Cancer` downloaded from <a href="https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer">Kaggle.com</a>

2. After downloading the data then we are duplicating the data-set into data1.csv and then we are checking various parameters in the data before further processing
	2.1. While we check the data we are Checking is if the `LUNG_CANCER` param is YES or not, if it is YES then assign the value to 1 else assign the value to 0
	2.2. Assign the variable y to data['LUNG_CANCER'] for setting the y axis
	2.3. Drop the columns: 'GENDER','LUNG_CANCER','NaN' as these are not required
	2.4. Assign the variable X to the remaining data after we have dropped the above mentioned columns

3. Training the data-set, use the X and y values to train the data set using the function "train_test_spilt()' pass the X and y params and store the X_train X_test y_train y_test for regression process

4. Linear Regression
	4.1. Intialize the model using the LinearRegression() function
	4.2. Fit the model with the model.fit() where args passed are X_train and y_train
	4.3. Do the prediction and find the prediction values

5. Logistic Regression
	5.1. Initialize the model using the LogisticRegression() function
	5.2. Fit the model with the model.fit() where args passed are X_train and y_train
	5.3. Perform prediction and find the predicted values

6. Confusion Matrix
	6.1. cm1 =  Create the confusion matrix for predicted values from LinearRegression
	6.2. cm2 = Create the confusion matrix for predicted values from LogisticRegression
	6.3. Plot the graph using the confusion matrix found!

7. Print the accuracy of the model

8. Plot ROC Curve (Reciever Operating Characterstic Curve) for both the Regression

<hr>