# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Import data
train_data = pd.read_csv('a3_train_final.tsv', sep='\t')
test_data = pd.read_csv('a3_test_final.tsv', sep='\t')

#Adding column names
train_data.columns = ['Label','Comment']
test_data.columns = ['Label','Comment']

#Drop rows containing uncertainty (-1)
train_data = train_data[train_data["Label"].str.contains("-|–")==False]

#Drop rows with missing data
train_data.dropna()

train_data

#Drop rows with ambigous anotating (both 0 and 1) and adding the 100% consensus label to the rest
labels = []
for index, row in train_data.iterrows():
    ones = 0
    zeros = 0
    for char in row[0]:
        if char == '1':
            ones += 1
        if char == '0':
            zeros += 1

    if ones != 0 and zeros != 0:
        train_data = train_data.drop(index)
    else:
        if '1' in row[0]:
            labels.append(1)
        else: labels.append(0)

train_data['Label'] = labels

#Vectorizing the features
tfidf_vectorizer = TfidfVectorizer()

comment_vector_train = tfidf_vectorizer.fit_transform(train_data['Comment'])
comment_vector_test = tfidf_vectorizer.transform(test_data['Comment'])

#Train/test split of test data
X_train, X_test, y_train, y_test = train_test_split(comment_vector_train, train_data['Label'], test_size=0.25, random_state=1, stratify=train_data['Label'])

#Function to test model performance
def model_test(model, name):

    #Train the model using the training sets
    model.fit(comment_vector_train, train_data['Label'])

    score = cross_val_score(model, X_train, y_train).mean()

    print(name + ' train accuracy: ' + str(score))

#Testing the MNB model
model_test(MultinomialNB(), 'Multinomial Naive Bayes')

#Testing the SVC model
model_test(SVC(), 'Support Vector Classifier')

#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

gs_clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1)

gs_clf = gs_clf.fit(comment_vector_train, train_data['Label'])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#Final Test
model = SVC(C = 100, gamma = 1, kernel = 'rbf')

model.fit(comment_vector_train, train_data['Label'])

y_pred = model.predict(comment_vector_test)

print('Tuned SVC test accuracy: ' + str(accuracy_score(test_data['Label'],y_pred)))

#Create a confusion matrix to visualize predictions and true negative/true positive absolute numbers
disp = plot_confusion_matrix(model, comment_vector_test, test_data['Label'], display_labels=["Anti-vaccination","Pro-vaccination"], cmap=plt.cm.Blues)   
plt.grid(False) 
disp.ax_.set_title('Confusion matrix')
plt.grid(False)
plt.savefig('Confusion_matrix.png')
plt.show()

#Find and print the true positive/true negative rates
tn, fp, fn, tp = confusion_matrix(test_data['Label'],y_pred).ravel()
tpRate = tn/(fp+tn)
tnRate = tp/(fn+tp)
print ("True positive rate: " + str(tpRate))
print ("True negative rate: " + str(tnRate))

#Feature importance test
def remove_word(word):

    new_train_data = train_data[~train_data['Comment'].str.contains(word)]
    comment_vector_train = tfidf_vectorizer.transform(new_train_data['Comment'])
    
    print("Number of removed rows: " + str(len(train_data)-len(new_train_data)))

    X_train, X_test, y_train, y_test = train_test_split(comment_vector_train, new_train_data['Label'], test_size=0.25, random_state=1, stratify=new_train_data['Label'])
   
    model.fit(comment_vector_train, new_train_data['Label'])
    
    score = cross_val_score(model, X_train, y_train).mean()
    print("Tuned SVC train accuracy without the word '" + word + "': " + str(score))

remove_word("vacc")

remove_word("anti")

remove_word("not")

remove_word("I")

remove_word("government")

remove_word("take")

#False negative and false positve labeling analysis
y_test = test_data['Label']
incorrect = []
for i in range(0,len(y_pred)):
    if y_pred[i] != y_test[i]:
        incorrect.append([test_data['Comment'][i],y_test[i]])

for comment in incorrect:
    print('Comment: ' + comment[0])
    print('True label: ' + str(comment[1]))
    print('-----------------------------------')

