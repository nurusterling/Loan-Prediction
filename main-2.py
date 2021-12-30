# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 11:16:11 2021

@author: Me
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('train_ctrUa4K.csv')
test = pd.read_csv('test_lAUu6dG.csv')

train_original = train.copy()
test_original = test.copy()


# Visualising the Independent variables
# Categorical Variables
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=1).plot.bar()

plt.subplot(222)
train['Married'].value_counts(normalize=1).plot.bar()

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=1).plot.bar()

plt.subplot(224)
train['Credit_History'].value_counts(normalize=1).plot.bar()
plt.show()

# Ordinal Variables
plt.subplot(131)
train['Dependents'].value_counts(normalize=1).plot.bar()

plt.subplot(132)
train['Education'].value_counts(normalize=1).plot.bar()

plt.subplot(133)
train['Property_Area'].value_counts(normalize=1).plot.bar()
plt.show()

# Numerical Variables
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

plt.subplot(122)
train['ApplicantIncome'].plot.box()

train.boxplot(column='ApplicantIncome', by = 'Education')

plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])

plt.subplot(122)
train['CoapplicantIncome'].plot.box()

plt.subplot(121)
sns.distplot(train['LoanAmount'])

plt.subplot(122)
train['LoanAmount'].plot.box()

plt.subplot(121)
sns.distplot(train['Loan_Amount_Term'])

plt.subplot(122)
train['Loan_Amount_Term'].plot.box()

# Univariate Analysis

# Categorical Variables
gender = pd.crosstab(train['Gender'], train['Loan_Status'])
gender.div(gender.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Proportions almost the same for both sexes for both Approved and Unapproved

married = pd.crosstab(train['Married'], train['Loan_Status'])
married.div(married.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Married People have more approved Loans

employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
employed.div(employed.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Proportions almost the same for Self employment status 

dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
dependents.div(dependents.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Distribution for dependents 1 and 3+ is almost same with the lowest proportion
# Dependent of 0 follows with 2 with the highest proportion

education = pd.crosstab(train['Education'], train['Loan_Status'])
education.div(education.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Graduates have a higher proportion of approved loans

married = pd.crosstab(train['Married'], train['Loan_Status'])
married.div(married.sum(1).astype(float), axis=0).plot.bar(stacked=1)

area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
area.div(area.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# People in semiurban areas have highest proportion of approved loans
# followed by Urban, then rural

history = pd.crosstab(train['Credit_History'], train['Loan_Status'])
history.div(history.sum(1).astype(float), axis=0).plot.bar(stacked=1)
# Those who paid previous loans have a higher proportion of approved loans


# Numerical Variables
ApplicantIncome = pd.crosstab(train['ApplicantIncome'], train['Loan_Status'])
ApplicantIncome.div(married.sum(1).astype(float), axis=0).plot.bar(stacked=1)

bins = [0, 2500, 5000, 7500, 81000]
groups = ['low', 'average', 'high', 'very high']

train['Income_bins'] = pd.cut(train['ApplicantIncome'], bins, labels=groups)

income = pd.crosstab(train['Income_bins'], train['Loan_Status'])
income.div(income.sum(1).astype(float), axis=0).plot.bar(stacked=1)

bins = [0, 1200, 2500, 4200]
groups = ['low', 'Average', 'high']

train['co_bin'] = pd.cut(train['CoapplicantIncome'], bins, labels=groups)

coincome = pd.crosstab(train['co_bin'], train['Loan_Status'])
coincome.div(coincome.sum(1).astype(float), axis=0).plot.bar(stacked=1)

train['Total_income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

bins = [1200, 3200, 5200, 8200, 81000]
groups = ['low', 'average', 'high', 'very high']

train['total_bin'] = pd.cut(train['Total_income'], bins, labels=groups)

totalincome_bin = pd.crosstab(train['total_bin'], train['Loan_Status'])
totalincome_bin.div(totalincome_bin.sum(1).astype(float), axis=0).plot.bar(stacked=1)


bins = [0, 100, 130, 170, 700]
groups = ['low', 'average', 'high', 'very high']

train['loanamount_bin'] = pd.cut(train['LoanAmount'], bins, labels=groups) 

loanamount_bin = pd.crosstab(train['loanamount_bin'], train['Loan_Status'])
loanamount_bin.div(loanamount_bin.sum(1).astype(float), axis=0).plot.bar(stacked=1)

train = train.drop(['Income_bins', 'co_bin', 'Total_income', 'total_bin', 
                    'loanamount_bin'], axis=1)

train['Dependents'].replace('3+', 3, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)

correlation = train.corr()
sns.heatmap(correlation, annot=True)


# Missing values computation
train.isnull().sum()
# Missing values in Gender, Married, Dependents, Self_Employed, LoanAmount,
# Loan_Amount_Term, Credit_History

# Input mode for the categorical Variables
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# Input for Numerical Attributes
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# Missing Values treatment for test data
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)


test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)


# Input for Numerical Attributes
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

sns.distplot(train['LoanAmount'])
train['LoanAmount'] = np.log(train['LoanAmount'])
test['LoanAmount'] = np.log(test['LoanAmount'])


# Building the model
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)


from sklearn.preprocessing import LabelEncoder
g_encoder = LabelEncoder()
train['Gender'] = g_encoder.fit_transform(train['Gender'])

m_encoder = LabelEncoder()
train['Married'] = m_encoder.fit_transform(train['Married'])

e_encoder = LabelEncoder()
train['Education'] = e_encoder.fit_transform(train['Education'])

s_encoder = LabelEncoder()
train['Self_Employed'] = s_encoder.fit_transform(train['Self_Employed'])

p_encoder = LabelEncoder()
train['Property_Area'] = p_encoder.fit_transform(train['Property_Area'])


test['Gender'] = g_encoder.fit_transform(test['Gender'])

m_encoder = LabelEncoder()
test['Married'] = m_encoder.fit_transform(test['Married'])

e_encoder = LabelEncoder()
test['Education'] = e_encoder.fit_transform(test['Education'])

s_encoder = LabelEncoder()
test['Self_Employed'] = s_encoder.fit_transform(test['Self_Employed'])

p_encoder = LabelEncoder()
test['Property_Area'] = p_encoder.fit_transform(test['Property_Area'])

x = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

x = pd.get_dummies(x, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
scaler1 = StandardScaler()
test = scaler1.fit_transform(test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator=regressor, n_estimators=1000)
bag.fit(x, y)
prediction = bag.predict(test)
accuracy_score(y_test, pred)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
i = 1

best = 0
import pickle

kf = StratifiedKFold(random_state=1, shuffle=True)
for train_index, test_index in kf.split(x, y):
    print(f'{i} of kfold {kf.n_splits}')
    xtr, xvl = x[train_index], x[test_index]
    ytr, yvl = y[train_index], y[test_index]
    best_model = BaggingClassifier(base_estimator=regressor, n_estimators=1000)
    best_model.fit(xtr, ytr)
    prediction = best_model.predict(xvl)
    score = accuracy_score(yvl, prediction)
    
    if score > best:
        best = score
        with open('model.pickle', 'wb') as f:
            pickle.dump(best_model, f)
    print('accuracy score', score)
    i += 1
    pred = best_model.predict_proba(test)[:, 1]

with open('model.pickle', 'rb') as f:
    pickled = pickle.load(f)
    
accuracy_score(yvl, pickled.predict(xvl))
    



submission = pd.read_csv('sample_submission_49d68Cx.csv')
submission['Loan_ID'] = test_original['Loan_ID']
submission['Loan_Status'] = prediction

submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission['Loan_Status'].replace(0, 'N', inplace=True)

pd.DataFrame(submission).to_csv('Naive.csv', index=False)



























