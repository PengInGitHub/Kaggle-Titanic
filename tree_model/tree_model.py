from sklearn import tree
import pandas as pd
import numpy as np

file_path = "/Users/pengchengliu/go/src/github.com/user/Titanic/data/"
train = pd.read_csv(file_path+"train.csv")
test = pd.read_csv(file_path+"test.csv")

#gender matters
train['Survived'].value_counts(normalize=True)# Perished 62% vs. Survived 38%
train['Survived'][train['Sex']=='male'].value_counts(normalize=True)# Perished 81% vs. Survived 19%
train['Survived'][train['Sex']=='female'].value_counts(normalize=True)# Perished 25% vs. Survived 75%
#woman is much more likely to survive than man

#age matters
train['Survived'][train['Age']<18].value_counts(normalize=True)# Perished 45% vs. Survived 55%
#most kids suvived(54%)
train['Survived'][(train['Age']<18) & (train['Sex']=='male')].value_counts(normalize=True)# Perished 40% vs. Survived 60%
train['Survived'][train['Sex']=='male'].value_counts()
train['Survived'][(train['Age']<18) & (train['Sex']=='male')].value_counts()
#among 109 males who survived, 35 are under 18 years old.

#first prediction
#gender-age model
test['Survived'] = 0
test['Survived'][(test['Sex']=='female')|(test['Age']<18)] = 1
#test['Survived'][(test['Sex']=='female')& (test['Pclass']==3)] = 0
test['Survived'].value_counts(normalize=True)# Perished 58% vs. Survived 42%

#submit
submit = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':test['Survived']})
submit.head()
filename = 'submit.csv'
submit.to_csv(filename, index=False)

#clean up data for decision tree model
full = train.append(test)
#missing imputation
#sex
full['Age'] = full['Age'].fillna(full['Age'].median()) 
full['Sex'][full['Sex']=='male'] = 0
full['Sex'][full['Sex']=='female'] = 1
#embark
full['Embarked'] = full['Embarked'].fillna('S')
full['Embarked'][full['Embarked']=='S'] = 0
full['Embarked'][full['Embarked']=='C'] = 1
full['Embarked'][full['Embarked']=='Q'] = 2

#decision tree
train, test = full[:891], full[891:]
target = train['Survived'].values 
feature_ones = train[['Fare', 'Age', 'Sex', 'Pclass']].values
#fit
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(feature_ones, target)
#check importance
print my_tree_one.feature_importances_
print my_tree_one.score(feature_ones, target)
print type(my_tree_one)

#predict and submit
full['Fare'] = full['Fare'].fillna(full['Fare'].median()) 
train, test = full[:891], full[891:]
print test['Fare'].median()
test_features= test[['Fare', 'Age', 'Sex', 'Pclass']].values
my_prediction = my_tree_one.predict(test_features)
#get id
PassengerId = np.array(test['PassengerId']).astype(int)
submit = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'] )
print submit.shape
submit.to_csv('submit.csv', index_label=['PassengerId'])









































