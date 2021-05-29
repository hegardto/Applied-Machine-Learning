# -*- coding: utf-8 -*-

# -- Sheet --

# # Assignment 2 - DAT340
# ## Group 20
# ### David Arvidsson: ardavid@student.chalmers.se
# ### Johan Hegardt: johheg@student.chalmers.se


# ## Task 1: Working with a dataset with categorical features


# The point of this experiment is that we want to investigate overfitting, and to do this we need to look at the performance of a model when we train and evaluate on the *same* data. That is, you need to fit the model on the full training set and also evaluate it on the full training set (and, of course, on the test set as well).
#  
# When we write "train and evaluate on the training set", we don't mean that you should use cross-validation. The point of cross-validation is that we can get an estimate of the model's performance on held-out data, since each fold in the cross-validation will act as a test set once. So we can't use this to investigate overfitting.
#  
#  To be clear, as a general best practice when working with ML, it is of course good to use cross-validation while developing and tuning models. But what we are doing now is not to optimize model performance, we are just investigating the problem of overfitting.
#  
# So, in summary fit a model with the entire training data, compute the accuracy scores on the training data and compare that with accuracy scores from testing data.


# ### Step 1. Reading the data


#Import the dataset and converting to input and output data
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('adult_train.csv')
df_test = pd.read_csv('adult_test.csv')

X_train = df_train.iloc[:,0:13]
Y_train = df_train.iloc[:,14]

X_test = df_test.iloc[:,0:13]
Y_test = df_test.iloc[:,14]

X_test.head()

# ### Step 2: Encoding the features as numbers.


#Encoding features as numbers using DictVectorizer
dv = DictVectorizer()

X_train_encoded = dv.fit_transform(X_train.to_dict('records'))
X_test_encoded = dv.transform(X_test.to_dict('records'))

#Creating and validating a Gradient Boosting Classifer on the data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

gradient_boosting_classifier = GradientBoostingClassifier()

gradient_boosting_classifier.fit(X_train_encoded, Y_train)

Y_pred = gradient_boosting_classifier.predict(X_train_encoded)

gradient_boosting_classifier_score = accuracy_score(Y_train, Y_pred)

print('Gradient Boosting Classifier accuracy: ' + str(gradient_boosting_classifier_score))

# ### Step 3. Combining the steps.


#Creating a pipeline, combining the vectorizing and classifier steps
from sklearn.pipeline import make_pipeline
  
pipeline = make_pipeline(
  DictVectorizer(),
  GradientBoostingClassifier()
)

pipeline.fit(X_train.to_dict('records'), Y_train)

Y_pred = pipeline.predict(X_train.to_dict('records'))

pipeline_score = accuracy_score(Y_train, Y_pred)

print('Pipeline accuracy: ' + str(pipeline_score))

# ## Task 2: Decision trees and random forests


# ### Underfitting and overfitting in decision tree classifiers.


#Evaluating train and test data for different max_depths for a decision tree classifier
max_depths = [1,10,20,50,100]
train_scores = []
test_scores = []

for max_depth in max_depths:
    decision_tree_classifier = tree.DecisionTreeClassifier(max_depth=max_depth)
    decision_tree_classifier.fit(X_train_encoded, Y_train)
    Y_pred_train = decision_tree_classifier.predict(X_train_encoded)
    Y_pred_test = decision_tree_classifier.predict(X_test_encoded)
    train_score = accuracy_score(Y_pred_train, Y_train)
    test_score = accuracy_score(Y_pred_test, Y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("Progress: " + str(max_depth))

#Plotting the train and test accuracies for the different max_depths
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg' 
plt.style.use('seaborn')
%matplotlib inline 

plt.plot(max_depths, train_scores, color = 'orange',label='Train accuracy')
plt.plot(max_depths, test_scores, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision tree classifier, train and test accuracy for different max_depths')
plt.grid(True)
plt.legend()
plt.show()

# **Effect of varying max_depth**
# 
# To be able to examine the effect of overfitting we made the choice to evaluate the accuracy obtained on max_depth values 1, 10, 20, 50, 100. When examining the graph it is clear that the obtained accuracy is similiar for both the train and test accuracy for small max_depth values, but as the max_depth value increases the difference between the train and test accuracy increases. This is a consequence of overfitting, i.e. that the model becomes too complex and "remembers" the training data and its low-level characteristics, being affected by the biases of that specific data. When evaluating on the training data, the model can use that knowledge to obtain a 100% accuracy for deeper depths (it basically remembers the dataset completely). 
# 
# However, when trying to transfer that knowledge and evaluating on a different dataset, e.g. the test data, the biases are not favorable and the overfitting's negative consequences are clear. For the lowest values of max_depth, the test accuracy is actually slightly higher than the training accuracy which in turn could be due to underfitting, i.e. keeping a simple model learning only the high level distribution of the dataset and by that scoring higher when being evaluated on the test data than the train data.


# ### Underfitting and overfitting in random forest classifiers.


#Evaluating train and test data for different max_depths and different amount of ensambles (random forest classifier)
max_depths = range(1,13)
train_scores = []
test_scores = []
n_estimators = [1,10,20,50,100,150,200]

for i in n_estimators:

    for max_depth in max_depths:

        rf_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=i,n_jobs=-1)
        rf_classifier.fit(X_train_encoded, Y_train)
        Y_pred_train = rf_classifier.predict(X_train_encoded)
        Y_pred_test = rf_classifier.predict(X_test_encoded)
        train_score = accuracy_score(Y_pred_train, Y_train)
        test_score = accuracy_score(Y_pred_test, Y_test)
        train_scores.append([i, max_depth, train_score])
        test_scores.append([i, max_depth, test_score])
        print('Progress: ' + str(i) + ' ' + str(max_depth)+'/12')

print(train_scores)
print(test_scores)

#Because of heavy computations when computing scores we made the choice to save these hardcoded. 
train_scores_temp = [[1, 1, 0.7609103471378921], [1, 2, 0.7800744617361385], [1, 3, 0.7861849161250358], [1, 4, 0.7948471229908355], [1, 5, 0.8208288535384343], [1, 6, 0.8261718719054049], [1, 7, 0.823438975385083], [1, 8, 0.8338198390593601], [1, 9, 0.8334205007109199], [1, 10, 0.8325914368579038], [1, 11, 0.8349557217072187], [1, 12, 0.8330824223788296], [10, 1, 0.7591904454179904], [10, 2, 0.7880601535541655], [10, 3, 0.8051352077549684], [10, 4, 0.8283532773802234], [10, 5, 0.8408524615859946], [10, 6, 0.8434018232670928], [10, 7, 0.8496053489316961], [10, 8, 0.8519703929883571], [10, 9, 0.855931936919961], [10, 10, 0.8547034121135917], [10, 11, 0.857989624606391], [10, 12, 0.8565462252588001], [20, 1, 0.7592518704794153], [20, 2, 0.7748843788514448], [20, 3, 0.8053808655605064], [20, 4, 0.8383343122864082], [20, 5, 0.8418048235413504], [20, 6, 0.8459507935555841], [20, 7, 0.8519395437060109], [20, 8, 0.8552256147316027], [20, 9, 0.8550720143534516], [20, 10, 0.858235305989797], [20, 11, 0.8579589780937086], [20, 12, 0.8604158956703867], [50, 1, 0.7591904454179904], [50, 2, 0.7794910368263661], [50, 3, 0.811768991783962], [50, 4, 0.8354166690244534], [50, 5, 0.8434017713957834], [50, 6, 0.8477628422987704], [50, 7, 0.8536593133898525], [50, 8, 0.8550106175854678], [50, 9, 0.8568225720171828], [50, 10, 0.858788136258196], [50, 11, 0.8588802738503338], [50, 12, 0.8598015931848266], [100, 1, 0.7591904454179904], [100, 2, 0.778815210252336], [100, 3, 0.812413530527303], [100, 4, 0.8362766717557136], [100, 5, 0.8434018232670926], [100, 6, 0.8504346014076554], [100, 7, 0.8532294558492163], [100, 8, 0.8555327306075811], [100, 9, 0.8570990178026106], [100, 10, 0.8594023868724466], [100, 11, 0.8594944773088485], [100, 12, 0.8606615393292039], [150, 1, 0.7591904454179904], [150, 2, 0.7787845024371969], [150, 3, 0.808021605626396], [150, 4, 0.8370137819239615], [150, 5, 0.8442309908627272], [150, 6, 0.8497897184274429], [150, 7, 0.8536900259205649], [150, 8, 0.8552563697024775], [150, 9, 0.8579588884978107], [150, 10, 0.8588803210060696], [150, 11, 0.8600473358706893], [150, 12, 0.860692289584505], [200, 1, 0.7591904454179904], [200, 2, 0.777678884340561], [200, 3, 0.8147479233557078], [200, 4, 0.8371979486500445], [200, 5, 0.8459507982711575], [200, 6, 0.8496669154603287], [200, 7, 0.853413660299888], [200, 8, 0.8559934421461367], [200, 9, 0.8577439573697058], [200, 10, 0.858941684765038], [200, 11, 0.8598630701175611], [200, 12, 0.8599859202404112]]
test_scores_temp = [[1, 1, 0.8049259873472145], [1, 2, 0.7677046864443215], [1, 3, 0.8042503531724096], [1, 4, 0.7964498495178429], [1, 5, 0.8240280081076101], [1, 6, 0.8348995761931085], [1, 7, 0.8315214053190836], [1, 8, 0.8450340888151834], [1, 9, 0.8325041459369817], [1, 10, 0.8415944966525398], [1, 11, 0.8412873902094465], [1, 12, 0.8453411952582766], [10, 1, 0.7637737239727289], [10, 2, 0.7645107794361525], [10, 3, 0.7974940114243597], [10, 4, 0.8368650574289048], [10, 5, 0.8382163257785148], [10, 6, 0.8444812972176157], [10, 7, 0.8483508384005897], [10, 8, 0.8540015969535041], [10, 9, 0.8530188563356059], [10, 10, 0.8547386524169277], [10, 11, 0.8581782445795713], [10, 12, 0.8598366193722744], [20, 1, 0.7637737239727289], [20, 2, 0.7648178858792457], [20, 3, 0.814446287083103], [20, 4, 0.821693999140102], [20, 5, 0.8425158159818193], [20, 6, 0.8458325655672256], [20, 7, 0.8517904305632332], [20, 8, 0.855352865303114], [20, 9, 0.8572569252502917], [20, 10, 0.859406670351944], [20, 11, 0.8611878877218844], [20, 12, 0.8592838277747067], [50, 1, 0.7637737239727289], [50, 2, 0.785823966586819], [50, 3, 0.8159203980099502], [50, 4, 0.8350224187703458], [50, 5, 0.8417173392297771], [50, 6, 0.8479823106688779], [50, 7, 0.8536330692217923], [50, 8, 0.8563970272096308], [50, 9, 0.8554757078803513], [50, 10, 0.8604508322584608], [50, 11, 0.8610036238560285], [50, 12, 0.8618635218966894], [100, 1, 0.7637737239727289], [100, 2, 0.7820158466924636], [100, 3, 0.8189914624408821], [100, 4, 0.8375406916037098], [100, 5, 0.8440513481972852], [100, 6, 0.8497635280388183], [100, 7, 0.8538787543762668], [100, 8, 0.8554757078803513], [100, 9, 0.8579325594250967], [100, 10, 0.8597137767950371], [100, 11, 0.8606965174129353], [100, 12, 0.8610650451446471], [150, 1, 0.7637737239727289], [150, 2, 0.7833056937534549], [150, 3, 0.8132792825993489], [150, 4, 0.8377249554695657], [150, 5, 0.8455254591241325], [150, 6, 0.8490878938640133], [150, 7, 0.8530802776242246], [150, 8, 0.8548000737055463], [150, 9, 0.859222406486088], [150, 10, 0.8602665683926048], [150, 11, 0.8616178367422148], [150, 12, 0.8615564154535962], [200, 1, 0.7637737239727289], [200, 2, 0.7831828511762177], [200, 3, 0.810576745900129], [200, 4, 0.835820895522388], [200, 5, 0.8434985565997175], [200, 6, 0.8491493151526319], [200, 7, 0.8536944905104109], [200, 8, 0.857379767827529], [200, 9, 0.8570726613844358], [200, 10, 0.858731036177139], [200, 11, 0.8611264664332657], [200, 12, 0.8614949941649775]]

print(len(train_scores_temp))

#Create plot for train and test scores, n_estimators=1
train_scores_1 = [0]
test_scores_1 = [0]

for i in range(0,12):
    train_scores_1.append(train_scores[i][2])
    test_scores_1.append(test_scores[i][2])

plt.plot(train_scores_1, color = 'orange',label='Train accuracy')
plt.plot(test_scores_1, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 1')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=10
train_scores_2 = [0]
test_scores_2 = [0]

for i in range(12,24):
    train_scores_2.append(train_scores[i][2])
    test_scores_2.append(test_scores[i][2])

plt.plot(train_scores_2, color = 'orange',label='Train accuracy')
plt.plot(test_scores_2, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 10')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=20
train_scores_3 = [0]
test_scores_3 = [0]

for i in range(24,36):
    train_scores_3.append(train_scores[i][2])
    test_scores_3.append(test_scores[i][2])

plt.plot(train_scores_3, color = 'orange',label='Train accuracy')
plt.plot(test_scores_3, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 20')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=50
train_scores_4 = [0]
test_scores_4 = [0]

for i in range(36,48):
    train_scores_4.append(train_scores[i][2])
    test_scores_4.append(test_scores[i][2])

plt.plot(train_scores_4, color = 'orange',label='Train accuracy')
plt.plot(test_scores_4, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 50')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=100
train_scores_5 = [0]
test_scores_5 = [0]

for i in range(48,60):
    train_scores_5.append(train_scores[i][2])
    test_scores_5.append(test_scores[i][2])

plt.plot(train_scores_5, color = 'orange',label='Train accuracy')
plt.plot(test_scores_5, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 100')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=150
train_scores_6 = [0]
test_scores_6 = [0]

for i in range(60,72):
    train_scores_6.append(train_scores[i][2])
    test_scores_6.append(test_scores[i][2])

plt.plot(train_scores_6, color = 'orange',label='Train accuracy')
plt.plot(test_scores_6, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 150')
plt.grid(True)
plt.legend()
plt.show()

#Create plot for train and test scores, n_estimators=200
train_scores_7 = [0]
test_scores_7 = [0]

for i in range(72,84):
    train_scores_7.append(train_scores[i][2])
    test_scores_7.append(test_scores[i][2])

plt.plot(train_scores_7, color = 'orange',label='Train accuracy')
plt.plot(test_scores_7, color = 'blue',label = 'Test accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0.7, 0.9])
plt.title('Decision tree classifier, n_estimators = 200')
plt.grid(True)
plt.legend()
plt.show()

# **What's the difference between the curve for a decision tree and for a random forest with an ensemble size of 1, and why do we see this difference?**
# 
# When examining the curves for a decision tree and for a random forest with an ensemble size of 1, it becomes clear that in this case the decision tree was able to produce predicitions that were much more stable. In this case there were no significant difference in the accuracy generated by the two models, but slightly higher for the single decision tree. In general the accuracy for both models tended to be higher as max_depth increased. But the predicitions made by the random forest were not as stable and in some cases the accuracy actually decreased compared to the previous max_depth value. In contrast, the decision tree followed a stable curve that increased for each incremental increase of the max_depth value. 
# 
# The reason for this difference is because in a random forest only a subset of the available features are considered in each tree. When the ensemble size is 1, the accuracy provided by a random forest classifier will therefore vary heavily, as some forests will consist of a tree with adequate features and the tree in other forests will only consitute of "low quality" features with no significant impact on the accuracy of predictions. On top of this, a random forest classifier also implements bagging, where each tree in the ensemble is trained on its own sampled training set. Because of this, the single tree in the random forest have access to less training data and is therefore less likely to be able to make correct predicitions. 
# 
# **What happens with the curve for random forests as the ensemble size grows?**
# 
# The ensemble size, n_estimators, is the amount of trees that are created to find the average of the predictions. The variation in the predictions, and therefore the variation in accuracy, becomes smaller. If a larger amount of trees are chosen a natural consequence is that the average will become more accurate, not necessarily in predicting the right value but at predicting the same value over and over again. This is especially visible when moving from a single n_estimator to 10 and to 50, while the effect of adding more trees becomes slightly lower after about 100.
# 
# **What happens with the best observed test set accuracy as the ensemble size grows?**
# 
# The test accuracy not only becomes more stable but also slightly higher when more estimator trees are added, compared to if a single or a few estimators are chosen. This is because the model gets more predictions to base its average prediction on, which in most cases will lead to more accurate predictions due to the fact that more data to base a decision on will often lead to a better decision/prediction.
# 
# **What happens with the training time as the ensemble size grows?**
# 
# As the number of n_estimors grow, the variation in prediction drops and the accuracy becomes slightly higher. This is something that we want to strive for, but it comes with the price of a higher training time. That is because the model will have to create a new decision tree for every increase in n_estimator, and that decision tree will in its create a prediction value. This means that if the ensamble size grows, more trees will be created and more predictions will be made, which in turn leads to a more stable model with slightly higher accuracy higher but with the price of higher training time.


# ## Task 3: Feature importances in random forest classifiers


#Initiate new pipleline in order to effectively obtain importance score of features for random forest classifier. 
pipeline = make_pipeline(
  DictVectorizer(),
  RandomForestClassifier()
)

pipeline.fit(X_train.to_dict('records'), Y_train)

#Create a list of feature importance, ranked from highest to lowest
import numpy as np

feature_names = pipeline.steps[0][1].feature_names_
importances = pipeline.steps[1][1].feature_importances_
important = []

indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(len(importances)):
    important.append([feature_names[indices[f]], importances[indices[f]]])

print(important)

# **Importance score**
# 
# The importance score of different features tells us, based on the predictions made in the trees that constitutes the random forest, what features are good at distniguishing between the two labels in this data set. A high importance score means that the Gini impurity is lower, which in its turn can be interpreted as that the probability that a datapoint is incorrectly labeled is lower for this feature. The five most important features measured by the importance score are the following:
# 
# - *fnlwgt* (Final weight. The, by the Census Bureau, estimated number of individuals having these characteristics.)
# - *age*
# - *capital-gain* (capital gains in USD)
# - *hours-per-week* (the amount of hours an individual works per week)
# - *marital-status=Married-civ-spouse*
# 
# An important point is that these features are not necessarily correlated with a high salary. However, the model identifies these as the most important features to distinguish between the two labels, i.e. a *high* and a *low* salary. Some of them are intuitive, such as that age and hours-per-week are good features to label salary levels. It is likely that a higher age and more hours worked are correlated to a higher salary. Appearently, capital gain and checking if individuals are married or not are also important parameters in order to distinguish between the two salary labels. The fifth, and most important, feature is the fnlwgt. This feature is supposed to represent the number of people that belongs to the specific characteristics described by that row/data point in the dataset. An intuitive meaning of the high importance score is that a lower *fnlwgt* value implies that the group belongs to a minority group of the society (e.g. coming from a specific country and having a specific gender, race and occupation) which seems to have a salary impact. Being able to find a threshold value to distinguish between minority and majority groups of the society could be the reason to the high importance score of the *fnlwgt* feature.


# **An alternative method: Drop Column**
# 
# Parr et al.(2018) describe an alternative way which is the *drop feature method*. When implementing this strategy the model would compute a baseline performance score, then drop a column entirely, retrain the model on the data missing the column and then recompute the performance score once again. The measured importance score for a specific feature would with this method  be the difference between the baseline score and the score from the model missing the feature. This strategy could adequately capture how important a certain feature is to overall model performance.  
# 
# As this strategy involves training the model several times on different data sets and calculating the accuracy seperately on these sets, it is expensive and requires extensive computational power. This cost can be mitigated somewhat by only using a subset of the training data for each test, but still it would be expensive in form of computational power. In addition, this approach only evaluates features individually, and could potentially have a hard time evaluate the importance that several features could have when applied/dropped together (redundant effects). 


