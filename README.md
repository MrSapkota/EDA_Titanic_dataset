# EDA_Titanic_dataset
This commit encompasses the Exploratory Data Analysis (EDA) and feature engineering steps performed on the Titanic dataset to prepare it for machine learning modeling.



Exploratory Data Analysis and applying logistic regression

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/titanic_train.csv')

df.shape
![image](https://github.com/user-attachments/assets/e5a322e0-a21f-49fa-b272-7c6d16f562eb)

df.describe()
![image](https://github.com/user-attachments/assets/19af7db3-04ad-47ee-a5ab-cb25f07cce2e)

df.info()

![image](https://github.com/user-attachments/assets/5753edd4-1f3b-44c7-8b49-736bfc0f75c6)

df.head()

![image](https://github.com/user-attachments/assets/21059dfe-cfbf-4eb6-a785-8c8326b90d95)

Checking for missing values

df.isnull().sum()

![image](https://github.com/user-attachments/assets/28ea8acd-7f6e-486c-b859-8641ebe9cfa9)

Data Cleaning

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.isnull().sum()

![image](https://github.com/user-attachments/assets/f3963c83-bff7-41e4-9ff1-733e292c85b0)

Handling Outliers

sns.boxplot(x=df['Fare'])
plt.show()

![image](https://github.com/user-attachments/assets/14ccbde0-1ab7-4473-8c18-ecc966625061)

sns.boxplot(x=df['Ticket'])
plt.show()

![image](https://github.com/user-attachments/assets/f59f9853-707e-4a2c-95c7-4ef1c6717808)

sns.boxplot(x=df['Age'])
plt.show()


![image](https://github.com/user-attachments/assets/aa111bf3-252e-4f9b-a06b-499d3597d518)

Training the dataset as Train

train = df.copy()

train.info()

![image](https://github.com/user-attachments/assets/0df878e2-3609-476e-a889-9cf077ee2fb7)


train.head()


![image](https://github.com/user-attachments/assets/62df6b4c-60f0-462f-9fad-174581d0faeb)

Plot

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu_r')


![image](https://github.com/user-attachments/assets/ba99c921-e090-4c72-90ba-ed552dc0f1cf)

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')


![image](https://github.com/user-attachments/assets/624db09d-26ea-4450-9077-0ebe80917ee4)

 sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')


![image](https://github.com/user-attachments/assets/4d231214-2b1d-4306-8c59-ca5ce0de79db)

sns.distplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)


![image](https://github.com/user-attachments/assets/6120a7e7-8005-4e46-987a-77b886d9a50b)

sns.countplot(x='SibSp', data=train)


![image](https://github.com/user-attachments/assets/03f8b682-5e20-4282-9398-5be51f2e55ef)

train['Fare'].hist(color='green', bins=40, figsize=(8,4))


![image](https://github.com/user-attachments/assets/aa6ddb78-09e1-40a1-876c-ed9b162ed82f)

Converting Categorical Features

pd.get_dummies(train['Embarked'], drop_first=True)

![image](https://github.com/user-attachments/assets/9c5f47ea-c75a-42f1-b097-8845e52b2a92)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)

train.head()


![image](https://github.com/user-attachments/assets/ac8b21fb-c016-41ff-a363-76aa1c7dd36c)

train = pd.concat([train, sex, embark], axis=1)

Bulding the Logistic Regression Model

train.drop('Survived', axis=1).head()


![image](https://github.com/user-attachments/assets/e79d152f-b69c-4c5a-badb-f5136d98b72b)

mapping = {'male': 1, 'Q': 1, 'S': 1, 'true': 1, 'false': 0}
# Apply mapping to each column separately
for col in ['male', 'Q', 'S']:
    if col in train.columns:  # Check if column exists before applying mapping
        train[col] = train[col].map(mapping)


train['Survived'].head()


![image](https://github.com/user-attachments/assets/587bb5eb-6594-4db7-bc5e-7a1f4850ab0b)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

train.drop(columns=['male', 'Q', 'S', 'Cabin_Encoded'], inplace=True)



Exploratory Data Analysis and applying logistic regression


[ ]
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install sklearn

Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.2)
Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.1)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (1.26.4)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (3.10.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (4.56.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (1.4.8)
Requirement already satisfied: numpy>=1.23 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (24.2)
Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (11.1.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (3.2.1)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib) (2.8.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
Requirement already satisfied: seaborn in /usr/local/lib/python3.11/dist-packages (0.13.2)
Requirement already satisfied: numpy!=1.24.0,>=1.20 in /usr/local/lib/python3.11/dist-packages (from seaborn) (1.26.4)
Requirement already satisfied: pandas>=1.2 in /usr/local/lib/python3.11/dist-packages (from seaborn) (2.2.2)
Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /usr/local/lib/python3.11/dist-packages (from seaborn) (3.10.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.56.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.2)
Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.1.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.1)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.1)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)
Collecting sklearn
  Downloading sklearn-0.0.post12.tar.gz (2.6 kB)
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  Preparing metadata (setup.py) ... error
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

[ ]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Loading dataset


[ ]
df = pd.read_csv('/content/titanic_train.csv')
Double-click (or enter) to edit

Data Understanding


[ ]
df.shape
(891, 12)

[ ]
df.describe()


[ ]
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

[ ]
df.head()

Checking for missing values


[ ]
df.isnull().sum()

Data Cleaning


[ ]
df['Age'].fillna(df['Age'].mean(), inplace=True)
<ipython-input-10-7ee7fe972bc2>:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Age'].fillna(df['Age'].mean(), inplace=True)

[ ]
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
<ipython-input-11-75133ca0fccb>:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)

[ ]
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
<ipython-input-12-808ebb813aa0>:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

[ ]
df.isnull().sum()

Handling Outliers


[ ]
sns.boxplot(x=df['Fare'])
plt.show()


[ ]
sns.boxplot(x=df['Ticket'])
plt.show()


[ ]
sns.boxplot(x=df['Age'])
plt.show()


Training the dataset as Train


[ ]
train = df.copy()

[ ]
train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          891 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        891 non-null    object 
 11  Embarked     891 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

[ ]
train.head()

Plot


[ ]
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, palette='RdBu_r')


[ ]
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')


[ ]
 sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')


[ ]
sns.distplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)


[ ]
sns.countplot(x='SibSp', data=train)


[ ]
train['Fare'].hist(color='green', bins=40, figsize=(8,4))


[ ]
sns.countplot(x='Cabin', data=train)


[ ]
train.head()

Converting Categorical Features


[ ]
train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          891 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        891 non-null    object 
 11  Embarked     891 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

[ ]
pd.get_dummies(train['Embarked'], drop_first=True)


[ ]
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

[ ]
train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)

[ ]
train.head()


[ ]
train = pd.concat([train, sex, embark], axis=1)

[ ]
train.head()

Bulding the Logistic Regression Model


[ ]
train.drop('Survived', axis=1).head()


[ ]
mapping = {'male': 1, 'Q': 1, 'S': 1, 'true': 1, 'false': 0}
# Apply mapping to each column separately
for col in ['male', 'Q', 'S']:
    if col in train.columns:  # Check if column exists before applying mapping
        train[col] = train[col].map(mapping)

[ ]
train.head()


[ ]
train['Survived'].head()


[ ]
from sklearn.model_selection import train_test_split

[ ]
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

[ ]
train.head()


[ ]
train.drop(columns=['male', 'Q', 'S', 'Cabin_Encoded'], inplace=True)


Training and Predicting

train.head()


![image](https://github.com/user-attachments/assets/e24001d9-2d15-422c-b216-6a5d7baa561a)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

print("Shape of train data before splitting:", train.shape)
print("Number of Survived instances:", train['Survived'].value_counts())
# Display count of 'Survived' values (0 and 1) to make sure there are values for the model to learn.

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)


![image](https://github.com/user-attachments/assets/b2bc30c2-af8b-4454-9708-59ce62dc880a)

logmodel.fit(X_train, y_train)


![image](https://github.com/user-attachments/assets/29178ea6-9f12-4f32-b92a-1c16e1db8986)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy = confusion_matrix(y_test, predictions)

accuracy


![image](https://github.com/user-attachments/assets/018dc9ee-d182-4b22-b404-b7e8262f2e5d)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)


![image](https://github.com/user-attachments/assets/bba86d78-fca8-4f2c-b7a6-29242d259f4a)

predictions


![image](https://github.com/user-attachments/assets/50806d19-fe11-46d9-a65b-a7c3311462f1)


















