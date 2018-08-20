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

#get title - man, female and boy
#in column 'Name', retrieve string from ',' to '.', eliminate space and punctuations
train$Title <- substring(train$Name,regexpr(",",train$Name)+2,regexpr("\\.",train$Name)-1)
male_title <- c("Capt", "Don", "Major", "Col", "Rev", "Dr", "Sir", "Mr", "Jonkheer")
female_title <- c("Mrs", "the Countess", "Dona", "Mme", "Mlle", "Ms", "Miss", "Lady")
boy_title <- c("Master")
train$Title[train$Title %in% male_title] <- "man"
train$Title[train$Title %in% female_title] <- "woman"
train$Title[train$Title %in% boy_title] <- "boy"

#get surname
train$Surname <- substring(train$Name, 0, regexpr(",", train$Name)-1)
head(train$Surname)

#build feature 'woman-child-group'
#preclude male
train$Surname[train$Title=="man"] <- "noGroup" 
train$SurnameFreq <- ave(1:nrow(train), train$Surname, FUN=length)
table(train$Surname)
#preclude single ones
train$Surname[train$SurnameFreq<=1] <- "noGroup"
#calculate survival rate
train$SurviveRate <- ave(train$Survived, train$Surname)
table(train$SurviveRate[train$Title != 'noGroup'])

#some statistics on the woman-child-group
#all perish
all_perished = train$Surname[train$SurviveRate == 0]
unique(all_perished[order(all_perished)])
#all survived
all_survived = train$Surname[train$SurviveRate==1]
unique(all_survived[order(all_survived)])
#sum of this two conditions
#mistake made here, nrow(train[train$SurviveRate == 0])
nrow(train[train$SurviveRate == 0 | train$SurviveRate == 1,])
nrow(train[train$Surname != 'noGroup',])
#124 of 142 woman-child pairs have either all survived or all perished


