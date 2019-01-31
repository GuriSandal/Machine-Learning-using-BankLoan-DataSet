import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train_original=train.copy()
test_original=test.copy()

print(train.columns)

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

print(train['Loan_Amount_Term'].value_counts())

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

print(train.isnull().sum())


test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)



train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1)
y = train.Loan_Status

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
acc = accuracy_score(y_cv,pred_cv)
print(acc)
pred_test = model.predict(test)

submission=pd.read_csv("Sample_Submission.csv")

submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


