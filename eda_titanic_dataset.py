# -*- coding: utf-8 -*-
"""EDA_Titanic_dataset

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bhNGVJuouBfdAxknqo1FMb5W7A1CM1ZX

**Exploratory Data Analysis and applying logistic regression**
"""

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Loading dataset"""

df = pd.read_csv('/content/titanic_train.csv')

"""**Data Understanding**"""

df.shape

df.describe()

df.info()

df.head()

"""**Checking for missing values**"""

df.isnull().sum()

"""**Data Cleaning**"""

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.isnull().sum()

"""**Handling Outliers**"""

sns.boxplot(x=df['Fare'])
plt.show()

sns.boxplot(x=df['Ticket'])
plt.show()

sns.boxplot(x=df['Age'])
plt.show()

"""**Training the dataset as Train**"""

train = df.copy()

train.info()

train.head()

"""**Plot**"""

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')

sns.distplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(color='green', bins=40, figsize=(8,4))

sns.countplot(x='Cabin', data=train)

train.head()

"""**Converting Categorical Features**"""

train.info()

pd.get_dummies(train['Embarked'], drop_first=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)

train.head()

train = pd.concat([train, sex, embark], axis=1)

train.head()

"""**Bulding the Logistic Regression Model**"""

train.drop('Survived', axis=1).head()

mapping = {'male': 1, 'Q': 1, 'S': 1, 'true': 1, 'false': 0}
# Apply mapping to each column separately
for col in ['male', 'Q', 'S']:
    if col in train.columns:  # Check if column exists before applying mapping
        train[col] = train[col].map(mapping)

train.head()

train['Survived'].head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

train.head()

train.drop(columns=['male', 'Q', 'S', 'Cabin_Encoded'], inplace=True)

"""**Training and Predicting**"""

train.head()

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

train.head()

print("Shape of train data before splitting:", train.shape)
print("Number of Survived instances:", train['Survived'].value_counts())
# Display count of 'Survived' values (0 and 1) to make sure there are values for the model to learn.

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy = confusion_matrix(y_test, predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)

predictions

