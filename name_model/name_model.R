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
#ave: Group Average Over Level Combinations of Factors
#usage: ave(a_variable, grouping_variables, func_to_apply_for_each_factor_level_combination)
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

#test new feature
#cross validation
library(ggplot2)

#adjusted survival rate
train$AdjustedSurvival <- (train$SurviveRate * train$SurnameFreq - train$Survived) / (train$SurnameFreq-1)
table(train$AdjustedSurvival)
all_survived = train$Surname[train$AdjustedSurvival==1]

#apply gender model + new feature
train$Predict = 0
train$Predict[train$Title == 'woman'] = 1
train$Predict[train$Title == 'boy' & train$AdjustedSurvival==1] = 1
train$Predict[train$Title == 'woman' & train$AdjustedSurvival==0] = 0
table(train$Predict, train$Title)

#plot how new feature improves simple gender model
#35 women are correctly predicted as perished 
ggplot(train[train$Title=='woman',]) +
    geom_jitter(aes(x=Pclass,y=Predict,color=factor(Survived))) + 
    labs(title="36 female predictions change from gender model on training set") +
    labs(x="Pclass",y="New Predictor") +
    geom_rect(alpha=0,color="black",aes(xmin=2.5,xmax=3.5,ymin=-0.45,ymax=0.45))
table(train$Survived[train$Title=='woman' & train$Predict==0])

#15 of 16 boys are correctly predicted as survived
ggplot(train[train$Title!='woman',]) +
    geom_jitter(aes(x=Title,y=Predict,color=factor(Survived))) +
    labs(title="16 male predictions change from gender model on training set") +
    labs(x="Title",y="New Predictor") +
    geom_rect(alpha=0,color="black",aes(xmin=0.5,xmax=1.5,ymin=0.55,ymax=1.45))
table(train$Survived[train$Title!='woman' & train$Predict==1])
#in overall the new predictor made 36+16=52 corrections from the gender model

#cross validation
#to test if new feature could improve the prediction

#Perform 25 trials of 10-fold cross validation
trials = 25; sum = 0
for (j in 1:trials){
x = sample(1:890); s = 0
for (i in 0:9){
    # Engineer "woman-child-groups" from training subset
    train$Surname <- substring(train$Name,0,regexpr(",",train$Name)-1)
    train$Surname[train$Title=='man'] <- 'noGroup'
    train$SurnameFreq <- ave(1:891,train$Surname,FUN=length)
    train$Surname[train$SurnameFreq<=1] <- 'noGroup'
    train$SurnameSurvival <- NA
    # calculate training subset's surname survival rate
    train$SurnameSurvival[-x[1:89+i*89]] <- ave(train$Survived[-x[1:89+i*89]],train$Surname[-x[1:89+i*89]])
    # calculate testing subset's surname survival rate from training set's rate
    for (k in x[1:89+i*89]) 
    train$SurnameSurvival[k] <- train$SurnameSurvival[which(!is.na(train$SurnameSurvival) & train$Surname==train$Surname[k])[1]]
    # apply gender model plus new predictor
    train$Predict <- 0
    train$Predict[train$Title=='woman'] <- 1
    train$Predict[train$Title=='boy' & train$SurnameSurvival==1] <- 1
    train$Predict[train$Title=='woman' & train$SurnameSurvival==0] <- 0
    c = sum(abs(train$Predict[x[1:89+i*89]] - train$Survived[x[1:89+i*89]]))
    s = s + c
}
cat( sprintf("Trial %d has 10-fold CV accuracy = %f\n",j,1-s/890))
sum = sum + 1-s/890
}
cat(sprintf("Average 10-fold CV accuracy from %d trials = %f\n",trials,sum/trials))

#submission







