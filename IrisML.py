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


# STEP 4: Evaluating Algorithms: