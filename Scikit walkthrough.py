# -*- coding: utf-8 -*-

# -- Sheet --

# ## Task 1: A classification example: fetal heart condition diagnosis


# #### Step 1. Reading the data


import pandas as pd
from sklearn.model_selection import train_test_split
  
# Read the CSV file.
data = pd.read_csv('CTG.csv', skiprows=1)

# Select the relevant numerical columns.
selected_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                 'Median', 'Variance', 'Tendency', 'NSP']
data = data[selected_cols].dropna()

# Shuffle the dataset.
data_shuffled = data.sample(frac=1.0, random_state=0)

# Split into input part X and output part Y.
X = data_shuffled.drop('NSP', axis=1)

# Map the diagnosis code to a human-readable label.
def to_label(y):
    return [None, 'normal', 'suspect', 'pathologic'][(int(y))]

Y = data_shuffled['NSP'].apply(to_label)

# Partition the data into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

X.head()

# #### Step 2. Training the baseline classifier


# Creating a dummy classifier
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy='most_frequent')

#Compute average score from cross validation
from sklearn.model_selection import cross_val_score

dummy_classifier_scores = cross_val_score(dummy_classifier, Xtrain, Ytrain)

dummy_classifier_score = dummy_classifier_scores.mean()

print('Accuracy of dummy classifier: ',dummy_classifier_score)

# #### Step 3. Trying out some different classifiers


#Random forest classifier
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier()

random_forest_classifier.fit(Xtrain, Ytrain)

random_forest_classifier_scores = cross_val_score(random_forest_classifier, Xtrain, Ytrain)

random_forest_classifier_score = random_forest_classifier_scores.mean()

print('Accuracy of random forest classifier: ',random_forest_classifier_score)

#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_classifier = GradientBoostingClassifier()

gradient_boosting_classifier.fit(Xtrain, Ytrain)

gradient_boosting_classifier_scores = cross_val_score(gradient_boosting_classifier, Xtrain, Ytrain)

gradient_boosting_classifier_score = gradient_boosting_classifier_scores.mean()

print('Accuracy of gradient boosting classifier: ',gradient_boosting_classifier_score)

#Creating, fitting, predicting and scoring using a Perceptron model
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(Xtrain,Ytrain)

perceptron_scores = cross_val_score(perceptron, Xtrain, Ytrain)

perceptron_score = perceptron_scores.mean()

print('Accuracy of perceptron classifier: ',perceptron_score)

# ##### Hyperparameter selection of the best classifier


#Using randomized search to evaluate different combinations of hyperparameters.
#The parameters generating the highest score are saved as best_params_ and later used to create an optimal model.
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': [1, 2, 4, 8, 10, 20, 30, None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}

tuned_gradient_boosting_classifier = RandomizedSearchCV(estimator = gradient_boosting_classifier, param_distributions = random_grid, n_iter = 3, cv = 3, verbose=2, random_state=0, n_jobs = -1)

tuned_gradient_boosting_classifier.fit(Xtrain, Ytrain)

print(tuned_gradient_boosting_classifier.best_params_)

print(tuned_gradient_boosting_classifier.best_score_)

# #### Step 4. Final evaluation


#Test tuned and non-tuned gradient boosting classifiers on test data 
from sklearn.metrics import accuracy_score

#Initialize gradient boosting classifier with the found optimal values
tuned_gradient_boosting_classifier = GradientBoostingClassifier(n_estimators = 300, min_samples_split = 5, min_samples_leaf = 4, max_depth = 8)

tuned_gradient_boosting_classifier.fit(Xtrain,Ytrain)

Ypred = tuned_gradient_boosting_classifier.predict(Xtest)

print('Accuracy of tuned Gradient Boosting classifier on test data: ',accuracy_score(Ytest, Ypred))

Ypred1 = gradient_boosting_classifier.predict(Xtest)

print('Accuracy of non-tuned Gradient Boosting classifier on test data: ',accuracy_score(Ytest, Ypred1))

# **Description of classifier**
# The selected classifier for this task was a Gradient boosting (GB) classifier. The reason for choosing GB was that it demonstrated the highest accuracy, measured in share of correctly predicted labels, for the cross validation test on the training data of the evaluated classifiers. After the choice was made to go ahead with GB we tried to optimize the hyperparameters before testing it on the test data. In particular, the choice was made to optimize parameters n_estimators, max_depth, min_samples_split and min_samples_leaf. By trying out different combinations for these values, we found that the optimal combination of the hyperparameters based on accuracy on training data was n_estimators = 300, max_depth = 8, min_samples_split = 5 and min_samples_leaf = 4. When deploying the tuned GB on the test data it achieved an accuracy of 0.957, compared to an accuracy 0.949 with the standard parameters. When tuning parameters based on training data there is always a risk of overfitting, but in this case the tuned model performed better also on the test data, with an accuracy of 0.936.
# 
# Gradient Boosting is based in optimizing a loss function on a dataset. The loss function used in our GB classifier was the default one in the scikit learn model, deviant, which uses logistic regression. The classifier creates decision trees to classify the data into the pre-defined labels. The part of the data that the model had the hardest time classifying are given a higher weight when the next decision tree is generated. The next generated tree will then try to improve the prediction by focusing on the part of the data that is hardest to predict. The loss function is optimized by minimizing the errors of the predictions, improving the predictive ability of the GB classifier in every boosting iteration, i.e. creation of a new decision tree.


# ## Task 2: Decision trees for classification


import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg' 
plt.style.use('seaborn')
%matplotlib inline 

data = pd.read_csv('iris.csv')
data_shuffled = data.sample(frac=1.0, random_state=0)
X = data_shuffled.drop('species', axis=1)
Y = data_shuffled['species']

data_shuffled.head()

X.to_numpy()[:5]

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.4, random_state=0)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)
Xtest.head(1)

one_instance = Xtest.head(1)

clf.predict(one_instance)

Ytest.head(1)

all_predictions = clf.predict(Xtest)

all_predictions

from sklearn.metrics import accuracy_score

accuracy_score(Ytest, all_predictions)

clf.score(Xtest, Ytest)

from sklearn.model_selection import cross_val_score

cross_val_score(clf, Xtrain, Ytrain, cv=5)

from sklearn.model_selection import cross_validate

cross_validate(clf, Xtrain, Ytrain, cv=5, scoring='accuracy')

from sklearn.svm import LinearSVC

clf2 = LinearSVC(max_iter=10000)
cross_validate(clf2, Xtrain, Ytrain, cv=5, scoring='accuracy')

from sklearn.dummy import DummyClassifier

majority_baseline = DummyClassifier(strategy='most_frequent')
cross_validate(majority_baseline, Xtrain, Ytrain, cv=5, scoring='accuracy')

X2 = X[['petal_length', 'petal_width']]
Y_encoded = Y.replace({'setosa':0, 'versicolor':1, 'virginica':2})

plt.figure(figsize=(5,5))
plt.scatter(X2.petal_length, X2.petal_width, c=Y_encoded, cmap='tab10');

def plot_boundary(clf, X, Y, cmap='tab10', names=None):

    if isinstance(X, pd.DataFrame):
        if not names:
            names = list(X.columns)
        X = X.to_numpy()

    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()

    x_off = (x_max-x_min)/25
    y_off = (y_max-y_min)/25
    x_min -= x_off
    x_max += x_off
    y_min -= y_off
    y_max += y_off
    
    xs = np.linspace(x_min, x_max, 250)
    ys = np.linspace(y_min, y_max, 250)
    
    xx, yy = np.meshgrid(xs, ys)
    
    lenc = {c:i for i, c in enumerate(clf.classes_)}
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([lenc[z] for z in Z])
    Z = Z.reshape(xx.shape)
    Yenc = [lenc[y] for y in Y]
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.15)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.2)
        
    sc = plt.scatter(X[:,0], X[:,1], c=Yenc, cmap=cmap, alpha=0.5, edgecolors='k', linewidths=0.5);
    
    plt.legend(handles=sc.legend_elements()[0], labels=list(clf.classes_))
    
    if names:
        plt.xlabel(names[0])
        plt.ylabel(names[1])

from sklearn.svm import LinearSVC

cls = LinearSVC()
cls.fit(X2, Y)
plot_boundary(cls, X2, Y)

cls = DecisionTreeClassifier()
cls.fit(X2, Y)
plot_boundary(cls, X2, Y)

class DecisionTreeLeaf:

    def __init__(self, value):
        self.value = value

    # This method computes the prediction for this leaf node. This will just return a constant value.
    def predict(self, x):
        return self.value

    # Utility function to draw a tree visually using graphviz.
    def draw_tree(self, graph, node_counter, names):
        node_id = str(node_counter)
        val_str = f'{self.value:.4g}' if isinstance(self.value, float) else str(self.value)
        graph.node(node_id, val_str, style='filled')
        return node_counter+1, node_id
        
    def __eq__(self, other):
        if isinstance(other, DecisionTreeLeaf):
            return self.value == other.value
        else:
            return False

class DecisionTreeBranch:

    def __init__(self, feature, threshold, low_subtree, high_subtree):
        self.feature = feature
        self.threshold = threshold
        self.low_subtree = low_subtree
        self.high_subtree = high_subtree

    # For a branch node, we compute the prediction by first considering the feature, and then 
    # calling the upper or lower subtree, depending on whether the feature is or isn't greater
    # than the threshold.
    def predict(self, x):
        if x[self.feature] <= self.threshold:
            return self.low_subtree.predict(x)
        else:
            return self.high_subtree.predict(x)

    # Utility function to draw a tree visually using graphviz.
    def draw_tree(self, graph, node_counter, names):
        node_counter, low_id = self.low_subtree.draw_tree(graph, node_counter, names)
        node_counter, high_id = self.high_subtree.draw_tree(graph, node_counter, names)
        node_id = str(node_counter)
        fname = f'F{self.feature}' if names is None else names[self.feature]
        lbl = f'{fname} > {self.threshold:.4g}?'
        graph.node(node_id, lbl, shape='box', fillcolor='yellow', style='filled, rounded')
        graph.edge(node_id, low_id, 'False')
        graph.edge(node_id, high_id, 'True')
        return node_counter+1, node_id

pip install graphviz

from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod

class DecisionTree(ABC, BaseEstimator):

    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        
    # As usual in scikit-learn, the training method is called *fit*. We first process the dataset so that
    # we're sure that it's represented as a NumPy matrix. Then we call the recursive tree-building method
    # called make_tree (see below).
    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            self.names = X.columns
            X = X.to_numpy()
        elif isinstance(X, list):
            self.names = None
            X = np.array(X)
        else:
            self.names = None
        Y = np.array(Y)        
        self.root = self.make_tree(X, Y, self.max_depth)
        
    def draw_tree(self):
        graph = Digraph()
        self.root.draw_tree(graph, 0, self.names)
        return graph
    
    # By scikit-learn convention, the method *predict* computes the classification or regression output
    # for a set of instances.
    # To implement it, we call a separate method that carries out the prediction for one instance.
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return [self.predict_one(x) for x in X]

    # Predicting the output for one instance.
    def predict_one(self, x):
        return self.root.predict(x)        

    # This is the recursive training 
    def make_tree(self, X, Y, max_depth):

        # We start by computing the default value that will be used if we'll return a leaf node.
        # For classifiers, this will be the most common value in Y.
        default_value = self.get_default_value(Y)

        # First the two base cases in the recursion: is the training set completely
        # homogeneous, or have we reached the maximum depth? Then we need to return a leaf.

        # If we have reached the maximum depth, return a leaf with the majority value.
        if max_depth == 0:
            return DecisionTreeLeaf(default_value)

        # If all the instances in the remaining training set have the same output value,
        # return a leaf with this value.
        if self.is_homogeneous(Y):
            return DecisionTreeLeaf(default_value)

        # Select the "most useful" feature and split threshold. To rank the "usefulness" of features,
        # we use one of the classification or regression criteria.
        # For each feature, we call best_split (defined in a subclass). We then maximize over the features.
        n_features = X.shape[1]
        _, best_feature, best_threshold = max(self.best_split(X, Y, feature) for feature in range(n_features))
        
        if best_feature is None:
            return DecisionTreeLeaf(default_value)

        # Split the training set into subgroups, based on whether the selected feature is greater than
        # the threshold or not
        X_low, X_high, Y_low, Y_high = self.split_by_feature(X, Y, best_feature, best_threshold)

        # Build the subtrees using a recursive call. Each subtree is associated
        # with a value of the feature.
        low_subtree = self.make_tree(X_low, Y_low, max_depth-1)
        high_subtree = self.make_tree(X_high, Y_high, max_depth-1)

        if low_subtree == high_subtree:
            return low_subtree

        # Return a decision tree branch containing the result.
        return DecisionTreeBranch(best_feature, best_threshold, low_subtree, high_subtree)
    
    # Utility method that splits the data into the "upper" and "lower" part, based on a feature
    # and a threshold.
    def split_by_feature(self, X, Y, feature, threshold):
        low = X[:,feature] <= threshold
        high = ~low
        return X[low], X[high], Y[low], Y[high]
    
    # The following three methods need to be implemented by the classification and regression subclasses.
    
    @abstractmethod
    def get_default_value(self, Y):
        pass

    @abstractmethod
    def is_homogeneous(self, Y):
        pass

    @abstractmethod
    def best_split(self, X, Y, feature):
        pass

from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class TreeClassifier(DecisionTree, ClassifierMixin):

    def __init__(self, max_depth = 10, criterion= 'maj_sum'):
        super().__init__(max_depth)
        self.criterion = criterion
        
    def fit(self, X, Y):
        # For decision tree classifiers, there are some different ways to measure
        # the homogeneity of subsets.
        if self.criterion == 'maj_sum':
            self.criterion_function = majority_sum_scorer
        elif self.criterion == 'info_gain':
            self.criterion_function = info_gain_scorer
        elif self.criterion == 'gini':
            self.criterion_function = gini_scorer
        else:
            raise Exception(f'Unknown criterion: {self.criterion}')
        super().fit(X, Y)
        self.classes_ = sorted(set(Y))

    # Select a default value that is going to be used if we decide to make a leaf.
    # We will select the most common value.
    def get_default_value(self, Y):
        self.class_distribution = Counter(Y)
        return self.class_distribution.most_common(1)[0][0]
    
    # Checks whether a set of output values is homogeneous. In the classification case, 
    # this means that all output values are identical.
    # We assume that we called get_default_value just before, so that we can access
    # the class_distribution attribute. If the class distribution contains just one item,
    # this means that the set is homogeneous.
    def is_homogeneous(self, Y):
        return len(self.class_distribution) == 1
        
    # Finds the best splitting point for a given feature. We'll keep frequency tables (Counters)
    # for the upper and lower parts, and then compute the impurity criterion using these tables.
    # In the end, we return a triple consisting of
    # - the best score we found, according to the criterion we're using
    # - the id of the feature
    # - the threshold for the best split
    def best_split(self, X, Y, feature):

        # Create a list of input-output pairs, where we have sorted
        # in ascending order by the input feature we're considering.
        sorted_indices = np.argsort(X[:, feature])        
        X_sorted = list(X[sorted_indices, feature])
        Y_sorted = list(Y[sorted_indices])

        n = len(Y)

        # The frequency tables corresponding to the parts *before and including*
        # and *after* the current element.
        low_distr = Counter()
        high_distr = Counter(Y)

        # Keep track of the best result we've seen so far.
        max_score = -np.inf
        max_i = None

        # Go through all the positions (excluding the last position).
        for i in range(0, n-1):

            # Input and output at the current position.
            x_i = X_sorted[i]
            y_i = Y_sorted[i]
            
            # Update the frequency tables.
            low_distr[y_i] += 1
            high_distr[y_i] -= 1

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            #x_next = XY[i+1][0]
            x_next = X_sorted[i+1]
            if x_i == x_next:
                continue

            # Compute the homogeneity criterion for a split at this position.
            score = self.criterion_function(i+1, low_distr, n-i-1, high_distr)

            # If this is the best split, remember it.
            if score > max_score:
                max_score = score
                max_i = i

        # If we didn't find any split (meaning that all inputs are identical), return
        # a dummy value.
        if max_i is None:
            return -np.inf, None, None

        # Otherwise, return the best split we found and its score.
        split_point = 0.5*(X_sorted[max_i] + X_sorted[max_i+1])
        return max_score, feature, split_point

def majority_sum_scorer(n_low, low_distr, n_high, high_distr):
    maj_sum_low = low_distr.most_common(1)[0][1]
    maj_sum_high = high_distr.most_common(1)[0][1]
    return maj_sum_low + maj_sum_high
    
def entropy(distr):
    n = sum(distr.values())
    ps = [n_i/n for n_i in distr.values()]
    return -sum(p*np.log2(p) if p > 0 else 0 for p in ps)

def info_gain_scorer(n_low, low_distr, n_high, high_distr):
    return -(n_low*entropy(low_distr)+n_high*entropy(high_distr))/(n_low+n_high)

def gini_impurity(distr):
    n = sum(distr.values())
    ps = [n_i/n for n_i in distr.values()]
    return 1-sum(p**2 for p in ps)
    
def gini_scorer(n_low, low_distr, n_high, high_distr):
    return -(n_low*gini_impurity(low_distr)+n_high*gini_impurity(high_distr))/(n_low+n_high)

cls = TreeClassifier(max_depth=2)
cls.fit(X2, Y)
cls.draw_tree()

plot_boundary(cls, X2, Y)

cls = TreeClassifier(max_depth=4, criterion='gini')
cls.fit(X2, Y)
score = cross_validate(cls, Xtrain, Ytrain, cv=5, scoring='accuracy')
print(score['test_score'].mean())
cls.draw_tree()

plot_boundary(cls, X2, Y)

#Tuning the max_depth hyperparameter by looking at accuracies for different scores
max_depths = range(0,30)
scores = []

for max_depth in max_depths:
    cls = TreeClassifier(max_depth=max_depth, criterion='gini')
    cls.fit(Xtrain, Ytrain)
    score = cross_validate(cls, Xtrain, Ytrain, cv=5, scoring='accuracy')
    scores.append(score['test_score'].mean())

print(scores)

#Plotting the accuracies for different max_depths
import matplotlib.pyplot as plt

plt.plot(scores)

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision tree accuracy for different max_depths')
plt.grid(True)
plt.show()

# **Max depth**
# 
# The max_depth is set to 2, because it generates the highest accuracy (together with max_depth = 3, but max_depth = 3 will give higher complexity/computation time). For the test set this results in an accuracy of 0.9. 


#Creating, fitting, predicting and scoring a model using the optimal max_depth
cls = TreeClassifier(max_depth=2, criterion='gini')
cls.fit(Xtrain, Ytrain)
Ypred = cls.predict(Xtest)
print(accuracy_score(Ytest, Ypred))
cls.draw_tree()

# ## Task 3: A regression example: predicting apartment prices


# Read the CSV file using Pandas.
alldata = pd.read_csv('sberbank.csv')

# Convert the timestamp string to an integer representing the year.
def get_year(timestamp):
    return int(timestamp[:4])
alldata['year'] = alldata.timestamp.apply(get_year)

# Select the 9 input columns and the output column.
selected_columns = ['price_doc', 'year', 'full_sq', 'life_sq', 'floor', 'num_room', 'kitch_sq', 'full_all']
alldata = alldata[selected_columns]
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop('price_doc', axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled['price_doc'].apply(np.log)

# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

#Creating and evaluating a dummy regressor model
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor()

dummy_regressor_score = -cross_validate(dummy_regressor, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()

print('Mean squared error of dummy regressor: ',dummy_regressor_score)

#Creating and evaluating a linear regressor model
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor_score = -cross_validate(linear_regressor, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()

print('Mean squared error of linear regressor model: ',linear_regressor_score)

#Creating and evaluating a random forest regressor model
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()

RFR_score = -cross_validate(RFR, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()

print('Mean squared error of Random Forest Regressor: ',RFR_score)

#Creating and evaluating a gradient boosting regressor model
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting_regressor = GradientBoostingRegressor()

gradient_boosting_regressor_score = -cross_validate(gradient_boosting_regressor, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()

print('Mean squared error of gradient boosting regressor: ',gradient_boosting_regressor_score)

# ##### Hyperparameter selection of the best regressor


from sklearn.model_selection import RandomizedSearchCV

#intuitively chosen values for different hyperparameters that will be tested in different combinations
random_grid = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': [1, 2, 4, 8, 10, 20, 30, None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}

tuned_gradient_boosting_regressor = RandomizedSearchCV(estimator = gradient_boosting_regressor, param_distributions = random_grid, n_iter = 3, cv = 3, verbose=2, random_state=1)

tuned_gradient_boosting_regressor.fit(Xtrain, Ytrain)

#Print the combination of values for the parameters that resulted in the highest accuracy
print(tuned_gradient_boosting_regressor.best_params_)

print('Mean squared error for tunet boosting regressor:',tuned_gradient_boosting_regressor.best_score_)

#Evaluating the tuned gradient boosting regressor using mean squared errors
from sklearn.metrics import mean_squared_error

tuned_gradient_boosting_regressor = GradientBoostingRegressor(...).fit(Xtrain, Ytrain)

print(mean_squared_error(Ytest, tuned_gradient_boosting_regressor.predict(Xtest)))

gradient_boosting_regressor.fit(Xtrain,Ytrain)

print(mean_squared_error(Ytest, gradient_boosting_regressor.predict(Xtest)))

# **Description of regressor**
# 
# The selected regressor for this task was a Gradient boosting (GB) regressor. The reason for choosing GB was that it demonstrated the highest accuracy, measured in mean squared errors, for the cross validation test on the training data of the evaluated regressors. After the choice was made to go ahead with GB we tried to optimize the hyperparameters before testing it on the test data. In particular, the choice was made to optimize parameters n_estimators, max_depth, min_samples_split and min_samples_leaf. By trying out different combinations for these values, we found that the optimal combination of the hyperparameters based on accuracy on training data was n_estimators = 300, max_depth = 2, min_samples_split = 10 and min_samples_leaf = 2. When deploying the tuned GB on the test data it achieved an accuracy of XXX, compared to accuracy AAA with the standard parameters. When tuning parameters based on training data there is always a risk of overfitting, but in this case the tuned model performed better also on the test data, with an accuracy of WWW.
# 
# The Gradient Boosting regressor works at the same way as the classifier, but it predicts a value instead of a label. For a description of the Gradient Boosting model, see Task 1.


# ## Task 4: Decision trees for regression


# #### Step 1. Implementing the regression model
# 
# Hint. Computing the variances from scratch at each possible threshold, for instance by calling np.var, will be time-consuming if the dataset is large. (What is the time complexity?) It's better to rely on the formula


#Method for efficiently calculating variation of values in a dataset
import math
def calculate_variation(data):
        data_pow = [number ** 2 for number in data]
        variation = (1/len(data))*sum(data_pow) - (1/(len(data)**2))*(sum(data_pow)**2)
        return variation

#Creating our own tree regressor class
from sklearn.base import RegressorMixin

class TreeRegressor(DecisionTree, RegressorMixin):

    def __init__(self, max_depth=10, criterion='maj_sum'):
        super().__init__(max_depth)
        self.criterion = criterion
        self.Y = []
        
    def fit(self, X, Y):
        if self.criterion == 'maj_sum':
            self.criterion_function = majority_sum_scorer
        elif self.criterion == 'info_gain':
            self.criterion_function = info_gain_scorer
        elif self.criterion == 'gini':
            self.criterion_function = gini_scorer
        else:
            raise Exception(f'Unknown criterion: {self.criterion}')
        super().fit(X, Y)
        self.classes_ = sorted(set(Y))
        self.Y = Y

    #Changed to returning the mean of the Y values
    def get_default_value(self, Y):
        return np.mean(Y)
    
    #Checks whether a set of output values is homogeneous by comparing the variation of the input data to
    # the square root of the variation in the original Y values divided by the size of the Y values.
    def is_homogeneous(self, data):
        return np.var(data) <= np.sqrt(np.var(self.Y))/len(self.Y)
        
    # Finds the best splitting point for a given feature by using variance reduction.
    def best_split(self, X, Y, feature):

        # Create a list of input-output pairs, where we have sorted
        # in ascending order by the input feature we're considering.
        sorted_indices = np.argsort(X[:, feature])        
        X_sorted = list(X[sorted_indices, feature])
        Y_sorted = list(Y[sorted_indices])

        n = len(Y)

        # Keep track of the best result we've seen so far.
        max_score = -np.var(Y)
        max_i = None

        # Go through all the positions (excluding the last position).
        for i in range(1, n-1):

            y_left = Y_sorted[0:i]
            y_right = Y_sorted[i:n]


            if X_sorted[i] == X_sorted[i+1]:
                continue

            # Compute the score as the variance reduction for the splitting point.
            score = calculate_variation(Y_sorted) - (len(y_left)/n)*calculate_variation(y_left) - (len(y_right)/n)*calculate_variation(y_right)

            # If this is the best split, remember it.
            if score > max_score:
                max_score = score
                max_i = i

        # If we didn't find any split (meaning that all inputs are identical), return
        # a dummy value.
        if max_i is None:
            return -np.inf, None, None

        # Otherwise, return the best split we found and its score.
        split_point = 0.5*(X_sorted[max_i] + X_sorted[max_i+1])
        return max_score, feature, split_point

# #### Step 2. Sanity check


#Method for creating random data with some patterns
def make_some_data(n):
    x = np.random.uniform(-5, 5, size=n)
    Y = (x > 1) + 0.1*np.random.normal(size=n)
    X = x.reshape(n, 1) # X needs to be a 2-dimensional matrix
    return X, Y

#Create data and a model to predict Y using X
X, Y = make_some_data(300)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.4, random_state=0)

tree_regressor = TreeRegressor(max_depth=1)

tree_regressor.fit(Xtrain,Ytrain)

Ypred = tree_regressor.predict(Xtest)

score = -cross_validate(tree_regressor, Xtrain, Ytrain, scoring='neg_mean_squared_error')['test_score'].mean()

print(score)

#Plot predictions made by tree regressor for the generated data
plt.scatter(Xtest, Ypred)

print(Ypred)
plt.xlabel('Input value')
plt.ylabel('Prediction')
plt.title('Prediction of random input values using Tree Regressor')
plt.grid(True)
plt.show()

#Plot the tree
tree_regressor.draw_tree()

# After training the decision tree on a dataset of 300 values created with the function make_some_data, we chose a very low max_depth, since the value of the prediction should almost only be dependent on if the X value is larger or smaller than 1. If the X value is larger than 1, the model should predict values close to (or equal to) 1 and if the X value is smaller than 1, the model should predict values lower close to (or equal to) 0. Large values of max_depth would therefore make the predicition based on noise (the random noise added around 0 and 1 by the make_some_data function), the only real difference between the values will be found by looking at the limit of X = 1. The max_depth value was therefore set to 1. This makes sense looking at both the plot of the predictions as well as the drawn tree, since the model now is only looking at if X > 1 or < 1, which is what it should do.


# #### Step 3. Predicting apartment prices using decision tree regression


#Taking back the sberbank data
alldata = pd.read_csv('sberbank.csv')

# Convert the timestamp string to an integer representing the year.
def get_year(timestamp):
    return int(timestamp[:4])
alldata['year'] = alldata.timestamp.apply(get_year)

# Select the 9 input columns and the output column.
selected_columns = ['price_doc', 'year', 'full_sq', 'life_sq', 'floor', 'num_room', 'kitch_sq', 'full_all']
alldata = alldata[selected_columns]
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop('price_doc', axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled['price_doc'].apply(np.log)

# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

#Initialize tree regressor with max depth = 7, as this is the optimal depth showed in plot below. 
tree_regressor = TreeRegressor(max_depth = 7)

tree_regressor.fit(Xtrain, Ytrain)

score = -cross_validate(tree_regressor, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error')['test_score'].mean()

print('Mean squared error of tree regressor with depth = 7: ',score)

#Tuning the max_depth hyperparameter by iterating through 0-12
max_depths = range(0,13)
scores = []

for max_depth in max_depths:
    cls = TreeRegressor(max_depth=max_depth)
    cls.fit(Xtrain, Ytrain)
    score = -cross_validate(cls, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error')['test_score'].mean()
    scores.append(score)
    print(max_depth)

print(scores)

#Plotting the mean squared errors for the different max_depth values
plt.plot(scores)

plt.xlabel('max_depth')
plt.ylabel('Mean squared errors')
plt.title('Decision tree regressor mean squared errors for different max_depths')
plt.grid(True)
plt.show()

# We used max_depth = 7 since this value gave us the lowest mean squared errors when evaluating on the train data.


print("Test accuracy of tree regressor: " + str(mean_squared_error(Ytest, tree_regressor.predict(Xtest))))

# #### Step 4. Underfitting and overfitting


#Calculate squeare error for both test and training sets, with various max depths 
max_depths = range(0,2)
test_scores = []
train_scores = []

for max_depth in max_depths:
    cls = TreeRegressor(max_depth=max_depth)
    cls.fit(Xtrain, Ytrain)
    Ypred = cls.predict(Xtest)
    train_score = -cross_validate(cls, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error')['test_score'].mean()
    test_score = mean_squared_error(Ypred, Ytest)
    train_scores.append(train_score)
    test_scores.append(test_score)
    "Progress: " + str(max_depth) + "/" + str(12)

print(test_scores)
print(train_scores)

#Plot square error for both test and training sets, with various max depths
plt.plot(train_scores, color = 'orange')
plt.plot(test_scores, color = 'blue')

plt.xlabel('max_depth')
plt.ylabel('Mean squared error')
plt.title('Decision tree regressor mean squared errors for different max_depths')
plt.grid(True)
plt.show()

# The plot above shows that the train mean squared errors are slightly lower than the mean squared errors for the test data for more or less all the different max_depths, meaning that it performs better on the train data than on the test data. This could be interpreted as that the model created is a slightly overfitted to the data it was trained on. Even though both the train data and test data are random samples from the same dataset, the model is sensitive to the example it encounters in only the part of the data that it is training on. This part of the data could, even though it is randomly selected, have biases that the test data does not. It is therefor common that the accuracy when evaluating the model on the data it was trained and/or evaluated on is higher then the data it later is tested on. Because it is simply fitted to the train data.


