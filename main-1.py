# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 11:16:11 2021

@author: Me
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('Loan/train_ctrUa4K.csv')
test = pd.read_csv('Loan/test_lAUu6dG.csv')

train_original = train.copy()
test_original = test.copy()


# Univariate Analysis
# Categorical Variables
train['Gender'].value_counts(normalize=1).plot.bar()
# Most of the loan approvals where for males

train['Married'].value_counts(normalize=1).plot.bar()
# Most people were married

train['Dependents'].value_counts(normalize=1).plot.bar()
# Most of them had no dependents

train['Education'].value_counts(normalize=1).plot.bar()
# most of the people were graduates

train['Self_Employed'].value_counts(normalize=1).plot.bar()
# most of the people were not self employed

train['Credit_History'].value_counts(normalize=1).plot.bar()
# Most people who were approved had  settled their previous loans

train['Property_Area'].value_counts(normalize=1).plot.bar()
# Most people were from the semiurban areas


# Numerical Varibales
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
sns.boxplot(train['ApplicantIncome'])
plt.show()
# Upper Outliers in ApplicantIncome

plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
sns.boxplot(train['CoapplicantIncome'])
plt.show()
# Upper Outliers in CoapplicantIncome

plt.subplot(121)
sns.distplot(train['LoanAmount'])
plt.subplot(122)
sns.boxplot(train['LoanAmount'])
plt.show()
# Fairly normal

# Bivariate Analysis
# Visualising Data
# Categorical Variables
gender = pd.crosstab(train[train['Gender'], train['Loan_Status'])
gender.div(gender.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Loan Approval is almost the same  for both Gender

married = pd.crosstab(train['Married'], train['Loan_Status'])
married.div(married.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Married people have a higher proportion of approval

dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
dependents.div(dependents.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Dependents of 2 have the hoghest proportion, 1 and 3 have the lowest

education = pd.crosstab(train['Education'], train['Loan_Status'])
education.div(education.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Graduates have a higher proportion

employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
employed.div(employed.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Almost same

credit = pd.crosstab(train['Credit_History'], train['Loan_Status'])
credit.div(credit.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Those with settled Credit History have a very high proportion

area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
area.div(area.sum(1).astype(float), axis=0).plot.bar(stacked=True)
# Semiurban Urban then Rural

bins = [150, 2800, 4000, 8000, 81000]
groups = ['low', 'average', 'high', 'very high']

train['income_bins'] = pd.cut(train['ApplicantIncome'], bins, labels=groups)
income = pd.crosstab(train['income_bins'], train['Loan_Status'])
income.div(income.sum(1).astype(float), axis=0).plot.bar(stacked=True)

bins = [0, 1200, 2300, 42000]
groups = ['low', 'average', 'high']

train['co_bins'] = pd.cut(train['CoapplicantIncome'], bins, labels=groups)
coapplicant = pd.crosstab(train['co_bins'], train['Loan_Status'])
coapplicant.div(coapplicant.sum(1).astype(float), axis=0).plot.bar(stacked=True)

train['totalincome'] = train['ApplicantIncome']+train['CoapplicantIncome']

bins = [0, 2000, 5000, 8000, 81000]
groups = ['low', 'average', 'high', 'very high']
train['total_bins'] = pd.cut(train['totalincome'], bins, labels=groups)
total = pd.crosstab(train['total_bins'], train['Loan_Status'])
total.div(total.sum(1).astype(float), axis=0).plot.bar(stacked=True)

bins = [0, 100, 300, 700]
groups = ['low', 'average', 'high']

train['loan_bins'] = pd.cut(train['LoanAmount'], bins, labels=groups)
loan = pd.crosstab(train['loan_bins'], train['Loan_Status'])
loan.div(loan.sum(1).astype(float), axis=0).plot.bar(stacked=True)

del train['income_bins']
del train['co_bins']
del train['totalincome']
del train['total_bins']
del train['loan_bins']



# Missing Values Computation
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

# Numerical Variables
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median(), inplace=True)

#### CREDIT HISTORY MAY HAVE A DIFFERENT EFFECT BASED ON THIS PREPROCESSING
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

# Test data
test.isnull().sum()
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(train['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)

# Numerical Variables
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].median(), inplace=True)

#### CREDIT HISTORY MAY HAVE A DIFFERENT EFFECT BASED ON THIS PREPROCESSING
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)



# Variable Transformation
# Outliers in Applicant Income
uv = np.percentile(train['ApplicantIncome'], [99])[0]
train.ApplicantIncome[train['ApplicantIncome']>uv] = 1.5*uv
train['ApplicantIncome'] = np.log(train['ApplicantIncome'])

uv = np.percentile(train['CoapplicantIncome'], [99])[0]
train.CoapplicantIncome[train['CoapplicantIncome']>uv] = 1.5*uv
train['CoapplicantIncome'] = np.log(train['CoapplicantIncome'] + 1)

train['LoanAmount'] = np.log(train['LoanAmount'])


# Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['Gender'] = encoder.fit_transform(train['Gender'])

train['Married'] = encoder.fit_transform(train['Married'])

train['Education'] = encoder.fit_transform(train['Education'])

train['Self_Employed'] = encoder.fit_transform(train['Self_Employed'])

train['Property_Area'] = encoder.fit_transform(train['Property_Area'])

train['Dependents'].replace('3+', 3, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)

test['Gender'] = encoder.fit_transform(test['Gender'])

test['Married'] = encoder.fit_transform(test['Married'])

test['Education'] = encoder.fit_transform(test['Education'])

test['Self_Employed'] = encoder.fit_transform(test['Self_Employed'])

test['Property_Area'] = encoder.fit_transform(test['Property_Area'])

test['Dependents'].replace('3+', 3, inplace=True)


co = train.corr()
sns.heatmap(co, annot=True)

x = train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train['Loan_Status']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)

from sklearn.svm import SVC

model = SVC(kernel='sigmoid')
model.fit(x_train_s, y_train)
y_pred = model.predict(x_test_s)
accuracy_score(y_test, y_pred)

x_training = StandardScaler().fit_transform(x)

test = test.drop('Loan_ID', axis=1)
testing = StandardScaler().fit_transform(test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=5.0, random_state=1)
model.fit(x_training, y)
prediction = model.predict(testing)

submission = pd.read_csv('sample_submission_49d68Cx.csv')
submission['Loan_ID'] = test_original['Loan_ID']
submission['Loan_Status'] = prediction

submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission['Loan_Status'].replace(0, 'N', inplace=True)

submission.to_csv('submission1.csv', index=False)


# Stratified KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
i = 1

kf = StratifiedKFold(random_state=1, shuffle=True)
for train_index, test_index in kf.split(x, y):
    print(f'{i} of kfold {kf.n_splits}')
    xtr, xvl = x.iloc[train_index], x.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    prediction = model.predict(xvl)
    score = accuracy_score(yvl, prediction)
    print('accuracy score', score)
    i += 1
    pred = model.predict_proba(xvl)[:, 1]
    

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, prediction)
auc = metrics.roc_auc_score(yvl, pred)
plt.plot(fpr, tpr, label='validation, auc='+str(auc))
plt.legend()
plt.show()



























