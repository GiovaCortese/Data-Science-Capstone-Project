# Building Classifiers

# Classifiers
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

file = open('dl.pickle', 'rb')
dl_dict = pickle.load(file)

print(dl_dict.keys())
len(dl_dict['labels'])
len(dl_dict['data'])
len(dl_dict['labels'][0])
len(dl_dict['data'][0])

labels = np.asarray(dl_dict['labels'])
data = np.asarray(dl_dict['data'])
labels.shape
data.shape

# Using stratify=labels so that class proportions are preserved
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, stratify=labels)

x_train[0]
y_train[0]
x_test[0]
y_test[0]
x_train.shape

#*******************************************************************
# Create Model #1: RandomForestClassifier 
model1 = RandomForestClassifier(random_state=1)
model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)
y_pred1

results1 = accuracy_score(y_pred1, y_test)
results1

y_test[0], y_pred1[0]
y_test[1], y_pred1[1]
y_test[2], y_pred1[2]

#*******************************************************************
# Create Model #2: MLPClassifier 
model2 = MLPClassifier(random_state=1, max_iter=30, warm_start=True, learning_rate='adaptive')
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
y_pred2

results2 = accuracy_score(y_pred2, y_test)
results2

y_test[0], y_pred2[0]
y_test[1], y_pred2[1]
y_test[2], y_pred2[2]

#*******************************************************************
# Create Model #3: DecisionTreeClassifier 
model3 = DecisionTreeClassifier(random_state=1)
model3.fit(x_train, y_train)

y_pred3 = model3.predict(x_test)
y_pred3

results3 = accuracy_score(y_pred3, y_test)
results3

y_test[0], y_pred3[0]
y_test[1], y_pred3[1]
y_test[2], y_pred3[2]
#*******************************************************************

print(f'Model #1 - RandomForestClassifier Accuracy: {results1}')
print(f'Model #2 - MLPClassifier Accuracy: {results2}')
print(f'Model #3 - DecisionTreeClassifier Accuracy: {results3}')


f1 = open('model1.pickle', 'wb')
pickle.dump({'model1': model1}, f1)
f1.close()

f2 = open('model2.pickle', 'wb')
pickle.dump({'model2': model2}, f2)
f2.close()

f3 = open('model3.pickle', 'wb')
pickle.dump({'model3': model3}, f3)
f3.close()
