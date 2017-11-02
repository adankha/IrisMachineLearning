# Load libraries
import pandas
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# STEP 1: Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# STEP 2: Get a basic idea of our data and some statistical information

# 150 instances with 5 attributes
print(dataset.shape)

# Print from the head, 20 rows and 4 columns
# [sepal-length, sepal-width, petal-length, petal-width, class]
print(dataset.head(20))

# Looking at the summary of each attribute (columns)
print(dataset.describe())

# Look at the class distribution
print(dataset.groupby('class').size())

# After reaching to this point, we have a basic idea of the data.
# That is, we have what the items in the data are, basic statistical summary
# How many rows are associated with a specific class, and etc.

# STEP 3: Data Visualization [2 types: Univariate plots and multivariate]

# Univariate Plots: Help understand each attributes
# Multivariate Plots: Help understand the relationships between attributes


# Univariate: Box and Whisker Plots:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# Univariate: Hisogram Plots:
dataset.hist()
#plt.show()

# Multivariate Plots: Scatter Plot Matrix
scatter_matrix(dataset)
plt.show()


# STEP 4: Set-up for Evaluating Algorithms
# Evaluating Algorithms will consist of a few steps:
#   1. Separate out a validation dataset
#   2. Set-up the test harness to use 10-fold cross validation
#   3. Build 5 different models to predit species from flower measurements
#   4. Select the best model

# Create a Validation Dataset:

# Split-out validation dataset
# test_size is 20%... Typically we have 80% train, 20% test
# X_train and Y_train will hold the data points for our train set
# X_test and Y_test will hold the data points for our test set (or validation set)
arr = dataset.values
X = arr[:,0:4]
Y = arr[:,4]
test_size = 0.20
seed = 7
scoring = 'accuracy'
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# Step 5 Evaluating Algorithms / Build Models:
#   6 different algorithms will be shown here
#   Logistic Regression (LR)
#   Linear Discimninant Analsis (LDA)
#   K-Nearest Neighbors (KNN)
#