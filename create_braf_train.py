# NAME: create_braf_train.py

import csv,sys,os
import numpy as np
import pandas as pd
from sklearn import svm, model_selection , preprocessing  # train_test_split is now in model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV           # GridSearchCV is now in model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import os


# Read in test ds
testds = pd.read_csv("BRAF_test_SVM.csv")
testds_colnames = testds.columns
print("testds_head: ", testds_colnames)
print()

# Read in train ds
trainds = pd.read_csv("BRAF_train_SVM_new.csv")
trainds_colnames = trainds.columns
print("trainds_head: ", trainds_colnames)
print()

# Delete cols in train DS that are not in test DS
colnames_not_in_newtrainds = trainds_colnames.difference(testds_colnames)
print("colnames_not_in_newtrainds: ", colnames_not_in_newtrainds)
print()

# Drop columns listed in colnames_not_in_newtrainds from trainds
# and create a new train ds called newtrainds
newtrainds = trainds.drop(colnames_not_in_newtrainds, index=None, axis=1)

# Write out new csv file
newtrainds.to_csv("BRAF_train_SVM.csv")



print()
print("DONE")

