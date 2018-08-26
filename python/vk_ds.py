#https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy
#solution from vk_ds
import pandas as pd
from func import load_data,missing_value,get_title
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import numpy as np

#1.Data Wrangling
#load data
train, test, combine = load_data()

#missing data visualization
missing_train= missing_value(train)
missing_test= missing_value(test)

#drop variable with too many missing values
for d in combine:
    d.drop(['Cabin'], axis=1, inplace=True)

#impute missing value in place
#test:Fare
#continuous, replace by median
test['Fare'].fillna(test['Fare'].median(), inplace=True)
#train:Embarked
#categorical, replace by mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
#Age
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train.isnull().sum()
test.isnull().sum()
combine = [train, test]

#2.Feature Engineering
for d in combine:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

#create title
for d in combine:
    d['Title'] = d['Name'].apply(get_title)

#clean title
rare_title = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
for d in combine:
    d['Title'] = d['Title'].replace(rare_title, 'Rare')
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby('Title', as_index=False).mean()

#create bins for age
age_label = ['Child', 'Teenage', 'Adult', 'Elder']
for d in combine:
    d['AgeBin'] = pd.cut(d['Age'], bins=[0,12,20,40,120],labels=age_label)

#create bins for fare
fare_label = ['Low', 'Median', 'Average', 'High']
for d in combine:
    d['FareBin'] = pd.cut(d['Fare'], bins=[0,7.91,14.45,31,120],labels=fare_label)

for d in combine:
    d.drop(['Age', 'Fare', 'Name', 'Ticket'],axis=1,inplace=True)

train.drop(['PassengerId'], axis=1, inplace=True)
combine = [train, test]

col_names = ['AgeBin', 'Sex', 'FareBin', 'Title', 'Embarked']
pre_names = ['AgeType', 'SexType', 'FareType', 'TitleType', 'EmType']
train = pd.get_dummies(train, columns=col_names, prefix=pre_names)
test = pd.get_dummies(test, columns=col_names, prefix=pre_names)
combine = [train, test]

#3. Feture Selection
#correlation matrix
sns.heatmap(train.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(20,12)
plt.show()

train.columns.values

#drop sex_male
for d in combine:
    d.drop(['SexType_male'],axis=1,inplace=True)

#pair plots
g = sns.pairplot(data=train, hue='Survived', palette = 'seismic',
                 size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])    


#4. Modeling
#Data Prepariation
all_features = train.drop(['Survived'],axis=1)
target = train['Survived']
X_train,X_test,y_train,y_test = train_test_split(all_features,target,test_size=0.3,random_state=40)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#LR
model = LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))


#cross validation
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_lr=cross_val_score(model,all_features,target,cv=10,scoring='accuracy')
print('The cross validated score for Logistic REgression is:',round(result_lr.mean()*100,2))
y_pred = cross_val_predict(model,all_features,target,cv=10)
sns.heatmap(confusion_matrix(target,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)

#submit
PassengerId = np.array(test['PassengerId']).astype(int)
my_prediction = model.predict(test.drop(['PassengerId'], axis=1))
submit = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'] )
submit.to_csv('submit.csv', index_label=['PassengerId'])







































 