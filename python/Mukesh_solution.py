#https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide
#strategy from Mukesh Chapagain

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from func import load_data

#outline
#1.EDA: Exploratory Data Analysis with Visualization
#2.Feature Extraction
#3.Data Modeling
#4.Model Evaluation

#1.EDA: Exploratory Data Analysis with Visualization

#1.1 load data
train, test = load_data()
#1.2 data structure
print train.shape#891*12
train.describe()#statistics on numerical variables
train.describe(include=['O'])#categorical data
train.info()#check data tyoe and missing value
train.isnull().sum()
train['Embarked'].value_counts(normalize=True)

#1.3 relationship btw features and target variable
#target var distribution
survived = train['Survived'][train['Survived']==1]
not_survived = train['Survived'][train['Survived']==0]
print "Survived: %i (%.1f%%)"%(len(survived),float(len(survived))/len(train)*100.0)
print "Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0)

#features and target var

#Passenger class
train.Pclass.value_counts()
#average survive rate is 38%
train.groupby('Pclass').Survived.value_counts(normalize=True)
#most 1st class survived (63%)
#47% 2nd class survived 
#only 24% class 3 survived
train[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', data=train)

#Sex
train.groupby('Sex').Survived.value_counts(normalize=True)
#female is much more likely to survive(74%) in comparison to male(19%)
train[['Sex','Survived']].groupby('Sex', as_index=False).mean()
sns.barplot(x='Sex', y='Survived', data=train)
#sex and passenger class
tab = pd.crosstab(train['Pclass'],train['Sex'])
tab.div(tab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.xlable('Pclass')
plt.ylable('Survived')
sns.factorplot('Sex','Survived', hue='Pclass', size=2, aspect=3, data=train)
#women in the 1st and 2nd class were almost all survived (close to 100%)
#men from the 2nd and 3rd class were almost all died(90%)

#Embarked
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=train)
#Parch
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
sns.barplot(x='Parch', y='Survived', data=train)
#SibSp
train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x='SibSp', y='Survived', data=train)

#Age
#violin plot
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x='Embarked', y='Age', data=train, hue='Survived', split=True, ax=ax1)
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train, split=True, ax=ax2)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train, split=True, ax=ax3)
#1st class have many more old ppl but less children
#almost all children in the 2nd class survived
#most children in the 3rd class survived
#younger ppl in the first class survived in comparison to the old

#check correlation
#Heatmap of correlation btw diff features
#focus on features have strong pos or neg correlation with target var
plt.figure(figsize=(25,10))
corr = train.drop('PassengerId', axis=1).corr()
sns.heatmap(corr, vmax=0.6, square=True, annot=True)
#Pclass and Fare have relative strong corr

#Feature Extraction
#generate Title
#use pd.Series.str.extract(' ([A-Za-z]+)\.')
train_test_data = [train, test]#return a list
for data in train_test_data:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.')
train.head()
#distribution
train.groupby('Title').Survived.value_counts()
#re-group
#use pd.Series.replace(to_be_replaced, new_value)
to_be_replaced = ['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
for data in train_test_data:
    data['Title'] = data['Title'].replace(to_be_replaced, 'Other')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
train['Title'].head()
#see new distribution
train[['Title','Survived']].groupby('Title', as_index=False).mean()
#convert categorical into ordinal
#use Series.map(a_map_object)
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}
for data in train_test_data:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)


for data in train_test_data:
    data['Sex'] = data['Sex'].map({'male':0, 'female':1}).astype(int)


#missing Embarked
for data in train_test_data:
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)


#Age
#fill na by a random num in (age_mean-std, age_mean+std)
for data in train_test_data:
    avg = data['Age'].mean()
    std = data['Age'].std()
    null_count = data['Age'].isnull().sum()
    random_list = np.random.randint(avg-std, avg+std, size=null_count)
    #impute nan
    data['Age'][np.isnan(data['Age'])] = random_list
    data['Age'] = data['Age'].astype(int)

for data in train_test_data:
    data['AgeBand'] = pd.cut(data['Age'], 5)


#map category to int
#use df.ioc[condition, a_col] = new_value
for data in train_test_data:
    data.loc[data['Age']<=16, 'Age'] = 0
    data.loc[(data['Age']>16)&(data['Age']<=32), 'Age'] = 1
    data.loc[(data['Age']>32)&(data['Age']<=48), 'Age'] = 2
    data.loc[(data['Age']>48)&(data['Age']<=64), 'Age'] = 3
    data.loc[data['Age']>64, 'Age'] = 4

#Fare
#fill na
for data in train_test_data:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
#cut
for data in train_test_data:
    data['Fareband'] = pd.qcut(data['Fare'],4)
#map to category
for data in train_test_data:
    data.loc[data['Fare']<=7.91, 'Fare'] = 0
    data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454), 'Fare'] = 1
    data.loc[(data['Fare']>14.454)&(data['Fare']<=31.0), 'Fare'] = 2
    data.loc[data['Fare']>31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)


#SibSp and Parch
#FamilySize
for data in train_test_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
train[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean()
#Travel alone
for data in train_test_data:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1 , 'IsAlone'] = 1

#Feature Selection
drops = ['FamilySize', 'Name', 'Parch', 'SibSp', 'Ticket', 'Cabin']
train = train.drop(drops, axis=1)#drop column
test = test.drop(drops, axis=1)#drop column
train = train.drop(['PassengerId', 'AgeBand', 'Fareband'], axis=1)    
test = test.drop(['AgeBand', 'Fareband'], axis=1)    
train.head()

#Classification and Accuracy
#Random Forest

#prepare data
x_train = train.drop('Survived', axis=1)
y_train = train['Survived']
x_test = test.drop("PassengerId", axis=1).copy()

#classify
forest = RandomForestClassifier(max_depth=5, min_samples_split=2, 
                                n_estimators=100, random_state=1)
forest = forest.fit(x_train, y_train)
my_prediction = forest.predict(x_test)
#submit
PassengerId = np.array(test['PassengerId']).astype(int)
submit = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'] )
submit.to_csv('submit.csv', index_label=['PassengerId'])




















