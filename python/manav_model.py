#https://www.kaggle.com/startupsci/titanic-data-science-solutions
#Solution of Manav Sehgal

from func import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#Solution Outline
#1.Analyze by describing data
#2.Data Wrangling
#3.Modeling and Predit


#1.Analyze by describing data

train, test, combine = load_data()
#check features
print train.columns.values
print train.info()
train.head()
#distribution of numerical variables
print train.describe()
#distribution of categorical variables
print train.describe(include=['O'])
#cabin has several duplicate values
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)

#Correlating numerical features
#histogram to help with continuous binning
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#by studying the distribution
#Age should be in the model
#we need to complete Age
#we should band age groups

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#Correlate categorical features
#Embarked, Sex and Survived
g = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()
#from the chart
#Embarked is correlated with survival rate, but this could be
#from the correaltion of 'Embarked' with 'Pclass'
#females are more likely to survive in comparison to male
#but Embarked C is an exception
#survival rate for pclass 3 varied among three embarked sites
#in this case, Sex should be added
#Embarked should be completed and added

#Correlating categorical and numerical features
#Fare, Embarked and Sex
g = sns.FacetGrid(train, row='Embarked', col='Survived',size=2.2, aspect=1.6)
g.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
g.add_legend()


#2.Data Wrangling

#drop variables
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]

#create new features
#extract title from Name
for d in combine:
    d['Title'] = d.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex']).sort_values(by=['male', 'female'], ascending=False)
#replace some titles
rare_title = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
for d in combine:
    d['Title'] = d['Title'].replace(rare_title, 'Rare')
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby('Title', as_index=False).mean()
#convert to ordinal
maps = {'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4, 'Rare':5}
for d in combine:
    d['Title'] = d['Title'].map(maps)
    d['Title'] = d['Title'].fillna(0)
train['Title'].head()

#drop variables
train = train.drop(['Name','PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
train.shape, test.shape

#Converting a categorical feature
#convert Sex to (female:1, male:0)
for d in combine:
    d['Sex'] = d['Sex'].map({'male':0, 'female':1}).astype(int)
train.head()

#Completing a numerical continuous feature
#use category median to impute missing age values
train['Age'].isnull().sum()  
test['Age'].isnull().sum()  
#category from sex(2 values) and pclass(3 values)
#so there are 2*3=6 possible median
guess_age=np.zeros((2,3))
for d in combine:
    #this loop is to get 6 median
    #iterate values of sex
    for i in range(0,2):
        #iterate values of pclass
        for j in range(0,3):
            age_median = d[(d['Sex']==i)&(d['Pclass']==j+1)]['Age'].dropna().median()
            #convert random float to nearest 0.5
            guess_age[i,j] = int(age_median/0.5+0.5)*0.5

    #this loop is to impute missing values by the 6 median values obtained above
    for i in range(0,2):
        for j in range(0,3):
            d.loc[(d.Age.isnull())&(d.Sex==i)&(d.Pclass==j+1), 'Age'] = guess_age[i,j]
    
    #update data type as int after imputation
    d['Age'] = d['Age'].astype(int)

train.Age.isnull().sum()
test.Age.isnull().sum()
#convert into num
train['AgeBand'] = pd.cut(train.Age, 5)
train[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean()

for d in combine:
    d.loc[d['Age']<=16, 'Age'] = 0
    d.loc[(d['Age']>16)&(d['Age']<=32), 'Age'] = 1
    d.loc[(d['Age']>32)&(d['Age']<=48), 'Age'] = 2
    d.loc[(d['Age']>48)&(d['Age']<=64), 'Age'] = 3
    d.loc[(d['Age']>64), 'Age'] = 4
    
train[['Age', 'Survived']].groupby('Age', as_index=False).mean()

train = train.drop(['AgeBand'], axis=1)
combine = [train, test]

#Create new feature combining existing features
for d in combine:
    d['FamilySize'] = d['Parch'] + d['SibSp'] + 1

for d in combine:
    d['IsAlone'] = 0
    d.loc[d['FamilySize']==1, 'IsAlone'] = 1

#drop variables
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]

#create combined feature
#age*pclass
for d in combine:
    d['AgePclass'] = d['Age']*d['Pclass']

#check multiple / related columns
#slice or change value of data frame
#use df.loc[condition_on_row, list_of_col] = some_value
train.loc[ :, ['Age', 'Pclass', 'AgePclass']].head()


#Completing a categorical feature
freq_embarked = train.Embarked.dropna().mode()[0]
for d in combine:
    d['Embarked'] = d['Embarked'].fillna(freq_embarked)

#Converting categorical feature to numeric
#use map
maps = {'S':0, 'C':1, 'Q':2}
for d in combine:
    d['Embarked'] = d['Embarked'].map(maps).astype(int)

#Quick completing and converting a numeric feature
#Fare
#need to impute one missing value in test
train.Fare.isnull().sum()
test.Fare.isnull().sum()
test.Fare.fillna(test.Fare.dropna().median(), inplace=True)
#equal-sized binning
#pandas.qcut()
#Quantile-based discretization function
#Discretize variable into equal-sized buckets based on rank or based on sample quantiles
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean()
for d in combine:
    d.loc[d['Fare']<=7.91, 'Fare'] = 0
    d.loc[(d['Fare']>7.91)&(d['Fare']<=14.454), 'Fare'] = 1
    d.loc[(d['Fare']>14.454)&(d['Fare']<=31.0), 'Fare'] = 2
    d.loc[(d['Fare']>31.0), 'Fare'] = 3
    d['Fare'] = d['Fare'].astype(int)
    
train = train.drop(['FareBand'],axis=1)    
combine = [train, test]
  

#3.Modeling and Predit

#prepare data
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

#classify
forest = RandomForestClassifier(max_depth=5, min_samples_split=2, 
                                n_estimators=100, random_state=1)
forest = forest.fit(X_train, Y_train)
my_prediction = forest.predict(X_test)
#submit
PassengerId = np.array(test['PassengerId']).astype(int)
submit = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'] )
submit.to_csv('submit.csv', index_label=['PassengerId'])






















































