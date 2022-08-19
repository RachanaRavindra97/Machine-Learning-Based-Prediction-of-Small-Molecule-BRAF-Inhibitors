BRAF-Detector Program(R) Version 1.0         12/15/2021

General Usage Notes
——————
This program creates a best-performance SVM model for the purpose of identifying small drug molecules that target BRAF. 
Requires a test and training data set in the form of a CSV
       Note that a default test and training data sets are provided

Installation
——————
Runs on Python 3.8 and Python 3.9
Requires that you have sklearn, numpy, and pandas packages installed

Usage
——————
Once installed, first run create_braf_train.py program to get proper training data set.
Then you can start running with default test and training data sets. 

To change training data set, change the constants __TRAINING_DS_NAME and __TEST_DS_NAME on lines 22 and 16 of create_braf_train.py program

To change test data set, change the constant __TEST_DS_NAME on lines 18 of BRAF_detector.py