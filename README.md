# Naive_Bayes_Classifier
This is a school project I did in the course "Data Mining and Data Warehousing"

This is a desktop application of a generic Classifier based on "Naive Bayes" algorithm using m-estimator (m=2)
https://en.wikipedia.org/wiki/Naive_Bayes_classifier

## Workflow
1. Constructing the structure of the model using Structure.txt file.
2. Data Pre-processing: 
    Data Cleaning: Fill in missing values, Identify outliers and smooth out noisy data (using the Equal-width Partitioning Discretization     Method) , Correct inconsistent data.
3. Loading the train set 
4. Building the classifier using the train set
5. Loading the test set
6. Classifying the records with Naive Bayes classifier using m-estimator (m=2)

## Resources
This project includes data files to test the classifier with:
1. Dataset general info.txt - general information about the data base from which the data is taken
2. Structure.txt - Description of the data set attributes.
3. train.csv - the train set
4. test.csv - the test set

## Prerequisites
Install Python 2.7 (Since the project uses pandas library, best to use Anaconda Distribution) can download here: 
https://www.anaconda.com/download/

## Running The Program
python Prog.py

## Usage 
1. Browse the directory with the Structure.txt , train.csv and test.cxs files
2. Type the desired number of Discretization Bins
3. Click Build
4. Click Classify



