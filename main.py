# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 11:16:11 2021

@author: Me
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('train_ctrUa4K.csv')
test = pd.read_csv('test_lAUu6dG.csv')

train_original = train.copy()
test_original = test.copy()

# Visualising Data
# Categorical Variables

train['Loan_Status'].replace('Y', 1, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)

y = train['Loan_Status']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

best = 0
best_x = []
import pickle
for i in range(1, 100):
    x = train[['Credit_History']]
    x.fillna(inplace=True, method='bfill')
    x = scaler.fit_transform(x)
    print(f'Filled x {i}')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test, pred)
    
    if score > best:
        print(best)
        best = score
        best_x = x
        print(score)
        with open('bestmodel.pickle', 'wb') as f:
            pickle.dump(model, f)



    
test_data = test[['Credit_History']]
test_data.fillna(test_data['Credit_History'].mode()[0], inplace=True)
test_data.fillna(method='ffill', inplace=True)
test_data = scaler.fit_transform(test_data)

with open('bestmodel.pickle', 'rb') as file:
    pickled = pickle.load(file)

accuracy_score(y_test, pickled.predict(x_test))

from sklearn.svm import SVC
model = SVC(C=5)
model.fit(x_train, y_train)
pred = model.predict(test_data)


submission = pd.read_csv('sample_submission_49d68Cx.csv')
submission['Loan_ID'] = test['Loan_ID']
submission['Loan_Status'] = pred

submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission['Loan_Status'].replace(0, 'N', inplace=True)

pd.DataFrame(submission).to_csv('RandomForest.csv', index=False)


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(fpr, tpr, label='validation, auc='+str(auc))
plt.legend()
plt.show()





























