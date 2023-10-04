import numpy as np

import pandas as pd

pd.set_option('display.width', 1000)

pd.set_option('display.max_column', None)

pd.set_option('display.precision', 2)

import matplotlib.pyplot as plt

import seaborn as sbn

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('bank-full.csv.csv')

test = pd.read_csv('bank.csv.csv')

print( train.describe() )

print( "\n"  )

print( train.describe(include="all")  )
print(  "\n"  )


print( train.describe() )

print( "\n"  )

print( train.describe(include="all")  )

print(  "\n"  )

print(  "\n\nTraining Data: \n" , train.columns  )

print(  "\n\nTesting Data:\n" , test.columns  )

print("\n\nFirst 5 rows of training data : \n")

print( train.head()  )

print()

print( train.sample(5)  )

print( "Data types for each feature : -"  )

print( train.dtypes  )

print( train.describe(include="all"))

print()

print( pd.isnull(train).sum()  )

print( "------------------\n\n"  )

print("Entire train.csv \n", train  )

print( "------------------\n\n"  )

print( "------------------\n\n"  )

print( "------------------\n\n"  )
train = train.drop(['euribor3m'], axis = 1)

train = train.drop(['cons.conf.idx'], axis = 1)

train = train.drop(['cons.price.idx'], axis = 1)

train = train.drop(['emp.var.rate'], axis = 1)

train = train.drop(['nr.employed'], axis = 1)

train.isnull().sum()

test.isnull().sum()

train['subscribed'].value_counts()

sns.countplot(data=train, x='subscribed')

train['subscribed'].value_counts(normalize=True)

train['job'].value_counts()

sns.set_context('paper')

train['job'].value_counts().plot(kind='bar', figsize=(10,6))

train['marital'].value_counts()

sns.countplot(data=train, x='marital')

sns.countplot(data=train, x='marital', hue='subscribed')

sns.distplot(train['age'])

print(pd.crosstab(train['job'],train['subscribed']))

job = pd.crosstab(train['job'],train['subscribed'])

job_norm = job.div(job.sum(1).astype(float), axis=0)

job_norm.plot.bar(stacked=True,figsize=(8,6))

pd.crosstab(train['marital'], train['subscribed'])

marital = pd.crosstab(train['marital'], train['subscribed'])

marital_norm = marital.div(marital.sum(1).astype(float), axis=0)

marital_norm

marital_norm.plot.bar(stacked=True, figsize=(10,6))

pd.crosstab(train['default'], train['subscribed'])

dflt = pd.crosstab(train['default'], train['subscribed'])

dflt_norm = dflt.div(dflt.sum(1).astype(float), axis=0)

dflt_norm

dflt_norm.plot.bar(stacked=True, figsize=(6,6))

train['subscribed'].replace('no', 0,inplace=True)

train['subscribed'].replace('yes', 1,inplace=True)

train['subscribed']

tc = train.corr()

tc
fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(tc, annot=True, cmap='YlGnBu')

target = train['subscribed']

train = train.drop('subscribed', axis=1)

train = pd.get_dummies(train)

train.head()

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=12)

from sklearn.linear_model import LogisticRegression

lreg = LogisticRegression()

lreg.fit(X_train,y_train)

pred = lreg.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy_score(y_val,pred)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, random_state=0)

clf.fit(X_train, y_train)

DecisionTreeClassifier(max_depth=4, random_state=0)

predict = clf.predict(X_val)

accuracy_score(y_val,predict)

test = pd.get_dummies(test)

test.head()

test_pred = clf.predict(test)

test_pred

submissions = pd.DataFrame()

submissions['ID'] = test['ID']

submissions['subscribed'] = test_pred

submissions['subscribed']

submissions['subscribed'].replace(0,'no',inplace=True)

submissions['subscribed'].replace(1,'yes',inplace=True)

submissions['subscribed']

submissions.to_csv('submission file.csv', header=True, index=False)
