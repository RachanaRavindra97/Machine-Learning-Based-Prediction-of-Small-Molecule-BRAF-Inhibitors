# Program: CS123A project 
# Author: Cyril Bhoomagoud, Rheya Mirani, Cleoma Arnold,
# Last update: 5/9/21

import csv,sys,os
import numpy as np
from sklearn import svm, model_selection , preprocessing  # train_test_split is now in model_selection 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV           # GridSearchCV is now in model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import pdb

# definition of constants #########################
__INFINITY=float('inf')
__TRAIN_DS_NAME='BRAF_train_SVM_new.csv'       #<------ enter file name of training  data
__TEST_DS_NAME='BRAF_test_SVM.csv'       #<------- enter file name of test data set



__DIMENSION_REDUCTION_PERCENTAGE = 0.90   #  for BRAF
__SVD_COVARIANCE_THRESHOLD = 0.5

# HELP Functions #
def get_num_svd_components(var_ratios):
    num = 1
    total_ratio = 0
    for r in var_ratios:
        if r == np.nan:
            return (num, np.nan)
        if total_ratio+r    >=   __SVD_COVARIANCE_THRESHOLD:
            return (num, total_ratio+r) 
        total_ratio = total_ratio + r
        num += 1

    # If we get here, we have reached the end of our components, so
    # return the current number of components to use
    return (num, total_ratio)


def make_sure_isnumber(n, row_index, col_index, compound, header, nan_locs):
    try:
        if n==np.nan or float(n)>=__INFINITY or float(n)==np.nan:
            print  ("*** Encountered value =", n, " for the compound named ", compound," and descriptor named ", header[col_index])
            nan_locs = nan_locs.append((row_index, col_index))
            return np.nan
        return float(n)
    except ValueError:
        return 0.
    
#Main program: Training SVM model
print ("Reading and processing the training data set named ", __TRAIN_DS_NAME)
print ("==================================================")

try:
    f=open(__TRAIN_DS_NAME)
    cr=csv.reader(f,delimiter=',')
except IOError:
    raise IOError("Problems locating or opening the training dataset named " + __TRAIN_DS_NAME)

header=np.array(cr.__next__())
data=np.arange(0) # vector that holds the training dataset
data_header=header[3:]
nan_locs=[]
row_index=0

print(">>> Building training data set.")
for row in cr:
    data_row = row[3:]
    new_data_row = np.arange(0)

    if len(data_header) == len(data_row):
        for col_index in range(len(data_header)):
            new_data_row = np.concatenate((new_data_row, [(make_sure_isnumber(data_row[col_index], \
                                                                              row_index, col_index, row[0], data_header,
                                                                              nan_locs))]))

        if len(data) > 0:
            data = np.vstack((data, np.concatenate((row[:3], new_data_row))))
        else:
            data = np.concatenate((row[:3], new_data_row))

f.close()

# Extract compound names
print (">>> Shape of training data = ", data.shape)
compound_names = data[:,0]

print (">>> Extracting training data set CID numbers.")
# Extract CID numbers 
cid_numbers = data[:,1]

print (">>> Extracting training data set class information.")
# Extract class information and make sure they are float/int types 
print ("data[:,2] = ", data[:,2])
class_info = np.array([int(x) for x in data[:,2] ])

# Make sure class is either 0 or 1. If so, append it to class_info list else raise error.
print ("class_info = ", class_info)

for c in class_info:
    if c not in [0,1]:
        raise ValueError("The column named ",header[2], " in " + __TRAIN_DS_NAME + " has a value not equal to 0 or 1.")

# At this point the data set has been read in and data contains just the data, and header contains the column
#  titles/names  and  class_info contains the class membership (i.e., 1 or 0) for each entery (row) in data.

# Now perform "gridding" to help find the best SVM kernel and parameters.
print (">>> Performing SVM gridding on training data set.")

#The following variable specifies the kernels and parameters we wish to test for
tuned_parameters = [{'kernel': ['linear','rbf','poly'], 'gamma': [0.2, 0.3, 0.5, 1], 'C': [1, 10, 100] }]

# Optimizes for the factors that we want
# for ex. factors = [ ('accuracy', 'accuracy'), ('average_precision', 'average_precision'), ('recall', 'recall')]
scores = [ ('accuracy', 'accuracy')]
print (">>> Optimizing for ", map(lambda x: x[1], scores) )

# Create np arrays of the data and class data sets. 
# Common names are X and y respectively
print (">>> Scaling training data set between [-1, 1]")
X = np.array(data[:,3:], dtype = float)
X = preprocessing.scale(X)   #  scale data between [-1, 1]
y = np.array(class_info, dtype = int)

print (">>> Starting the gridding process.")
# find out how many class 0 and class 1 entries we have.
# we need to use the minimum number for cross validation purposes
num_class_0 = list(y).count(0)
num_class_1 = list(y).count(1)
cv_size = min(num_class_0, num_class_1)

for score_name, score_func in scores:
    print ("      Tuning SVM parameters for %s" % score_name)

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, scoring = score_func)
    clf.fit(X, y)
    clf_scores = model_selection.cross_val_score(clf, X, y, cv = cv_size)
    print ("      CLF SCORES: ")
    print ("      ", score_name, ": %0.2f (+/- %0.2f)" % (clf_scores.mean(), clf_scores.std() * 2))
    print ("      Best parameters set found on development set:")
    print ("      =============================================")
    print ("clf.best_estimator_ = ", clf.best_estimator_)
    print ("clf.best_params_ = ", clf.best_params_)
    
    # Save best estimator and best parameter info
    best_estimator = clf.best_estimator_
    best_params = clf.best_params_
"""
Below is an example of how to generate training and test data sets
using sklearn's  functions. In this example, the test_size=0.2
parameter extracts a test data set that is 20% of the entire
dataset. You can change the percentage to whatever you like, but
values between 20% and 50% are not unreasonable, depending on
the size of the original data set.
"""
__TEST_DS_SPLIT_SIZE = 0.15

while (__TEST_DS_SPLIT_SIZE < 0.70):
    print("The following are the results of making predictions of ~")
    print(str(int(100. * __TEST_DS_SPLIT_SIZE)) + "% of the original training data set")
    X_train, X_test1, y_train, y_test1 = model_selection.train_test_split(X, y, test_size=__TEST_DS_SPLIT_SIZE,
                                                                          random_state=0)  ### reaplace cross_valadation with model_selectio may need fixing
    print("X_train shape =", X_train.shape, "  y_train shape=", y_train.shape)
    print("X_test1 shape =", X_test1.shape, "  y_test1 shape=", y_test1.shape)

    print("clf.get_params(deep=True) =", clf.get_params(deep=True))
    print("clf.predict(X_test1) = ", clf.predict(X_test1))
    print("clf.decision_function(X_test1) = ", clf.decision_function(X_test1))
    print("clf.score(X_test1, y_test1) = {0}%".format(int((clf.score(X_test1, y_test1) * 10000)) / 100.))
    print("=======================")
    __TEST_DS_SPLIT_SIZE += 0.10

# Now read in and prep the test data set
print ("Reading and processing the test data set named ", __TEST_DS_NAME)
print ("============================================== ")

# Reading the test data
try:
    f = open(__TEST_DS_NAME)
    cr = csv.reader(f, delimiter = ',')
except IOError:
    raise IOError("Problems locating or opening the test dataset named " + __TEST_DS_NAME)

# Save the first row that is a header row and convert it to a numpy array
header_test = np.array(cr.__next__()) 

data_test = np.arange(0)  # Create an empty 1D array that will hold the training data.

# Extract column header names starting from 4th column and higher
data_header_test = header_test[3:]

nan_locs = []   
row_index = 0

print (">>> Building test data set.")
for row in cr:
    data_row = row[3:]
    new_data_row = np.arange(0)

    if len(data_header_test) == len(data_row): 
        for col_index in range(len(data_header_test)):
                new_data_row = np.concatenate((new_data_row, [(make_sure_isnumber(data_row[col_index], \
                                row_index, col_index, row[0], data_header_test, nan_locs))])) 
            
        if len(data_test) > 0:
            data_test = np.vstack((data_test, np.concatenate((row[:3], new_data_row))))
        else:
            data_test= np.concatenate((row[:3], new_data_row))

# Close the test data set
f.close()

# Extract compound names
print (">>> Shape of test data set = ", data_test.shape)
compound_names_test = data_test[:,0]

# Extract CID numbers 
print (">>> Extract test data CID numbers.")
cid_numbers_test = data_test[:,1]

# Extract test class information and make sure they are 
# float/int types 
print (">>> Extracting test data set class information.")
print ("data_test[:,2] = ", data_test[:,2])
#class_info_test = np.array(map(lambda x: int(float(x)), data_test[:,2]))
class_info_test = [int(x) for x in data_test[:,2]]

# Make sure test class is either 0 or 1. If so, append it to 
# class_info list else raise error.
for c in class_info_test:
    if c not in [0,1]:
        raise ValueError("The column named ",header_test[2], " in " + __TEST_DS_NAME + " has a value not equal to 0 or 1.")

# At this point the test data set has been read in and 
# data_test contains just the test data and header_test 
# contains the test column titles/names and  
# class_info_test contains the test class membership (i.e., 1 or 0)
# for each entery (row) in data_test.

# Create np arrays of the data_test and class_info_test data sets. 
# Common names are X and y respectively
print (">>> Scaling the test data set between [-1, 1]")
X_test = np.array(data_test[:,3:], dtype = float)
X_test = preprocessing.scale(X_test)   #  scale data between [-1, 1]
y_test = np.array(class_info_test, dtype = int)

# Get estimator and its parameters and use for future 
# predictions
estimator = best_estimator
print (">>> Best trained estimator = ", estimator)

# Set clf parameters to that of the best estimator
# and then refit X and y
clf = estimator
clf.fit(X,y)
print (">>> SVM re-fit coefficients, vectors, support vectors, ...")
print ("   clf.dual_coef_ = ", clf.dual_coef_)
print ("   clf.support_vectors_ = ", clf.support_vectors_)
print ("   clf.support_ = ", clf.support_)
print ("   clf.n_support_ = ", clf.n_support_)
print ("   clf.intercept_ = ", clf.intercept_)
print ("   ***clf.get_params(deep = False) = ", clf.get_params(deep = False))

"""
The following lines make predictions of the test dataset 
using the previously built SVM model that was constructed  
using the training dataset and re-fit SVM. 
The percentage prediction accuracy is also printed
"""
print (">>> Test data set predictions, decision function,...")
print ("   clf.predict(X_test) = ", clf.predict(X_test))
print ("   clf.decision_function(X_test) = ", clf.decision_function(X_test))
print ("   clf.score(X_test, y_test) = {0}%".format(int((clf.score(X_test, y_test) * 10000))/100.))
print ("=======================")
print()
print()
print (" =================  DONE ===================")


















