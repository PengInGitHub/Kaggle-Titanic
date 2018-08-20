#resource
#https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818

#simple gender model
#if male: 0 else: 1
setwd("~/go/src/github.com/user/Titanic/data/")
test <- read.csv("test.csv", stringsAsFactors = FALSE)
test$Survied[test$Sex=='male']=0
test$Survied[test$Sex=='female']=1
submit = data.frame(PassengerId=test$PassengerId, Survived=test$Survied)
write.csv(submit, "gender_model.csv",row.names=F)

#improve gender model
#which male survied and which female perished
train <- read.csv("train.csv", stringsAsFactors = FALSE)

#boy, survive rate 50%
table(train$Survived[train$Sex=='male' & train$Age<16])
#female in class 3, survive rate only 50%
table(train$Survived[train$Sex=='female'&train$Pclass==3])

#focuse on identifying these two sub groups
train$Title <- substring(train$Name,regexpr(",",train$Name)+2,regexpr("\\.",train$Name)-1)
head(train$Title)






