#Loading required packages:
require(tidyverse)
require(caret)
require(randomForest)

#----------------------------------------------------------------------------------------------------------------------------------------------------

#Set working directory:
setwd("C:/Users/punee/Desktop/New/Titanic Kaggle")

#----------------------------------------------------------------------------------------------------------------------------------------------------

#Load CSVs:
train.dat <- read_csv("train.csv",col_names = TRUE,na = c("", "NA"),quoted_na = TRUE)
test.dat <- read_csv("test.csv",col_names = TRUE,na = c("", "NA"),quoted_na = TRUE)
#----------------------------------------------------------------------------------------------------------------------------------------------------

#CLEANING PROCESS, #Feature Engineering:
#Merging data before training and testing:
test.dat$Survived <- NA
comb.data <- rbind(train.dat, test.dat)

#Check structure of combined data
str(comb.data)

#Checking Empty/NA column entries
na_count <-sapply(comb.data, function(comb.data) sum(length(which(is.na(comb.data)))))
na_count <- data.frame(na_count)
na_count

#Impute mean Age for NAs in Age column:
comb.data$Age <- ifelse(is.na(comb.data$Age), median(comb.data$Age, na.rm = T), comb.data$Age)
comb.data$Age <- as.integer(comb.data$Age)

#Embarked:
which(is.na(comb.data$Embarked))
median(comb.data$Embarked,na.rm = T)
comb.data$Embarked[c(62, 830)] <- 'S'

#Fare:
which(is.na(comb.data$Fare))
comb.data$Fare[c(1044)] <- median(comb.data[comb.data$Pclass == '3',]$Fare,na.rm = T)

#Cabin:
comb.data$Cabin <- NULL

#New variable Age_Group
comb.data <- comb.data %>%
  mutate(Age_Group = case_when(Age<13 ~ 'Below13',
                               Age>=13 && Age<20 ~ 'Teenager',
                               Age>=20 && Age<60 ~ 'Adult',
                               Age>=60 ~ 'Senior'))


#New variables Family_size
comb.data$count <- comb.data$SibSp + comb.data$Parch + 1
comb.data$Family_size[comb.data$count == 1] <- 'Single' 
comb.data$Family_size[comb.data$count > 1 && comb.data$count <6] <- 'Small' 
comb.data$Family_size[comb.data$count >=6 ] <- 'Big' 
comb.data <- comb.data %>%
  mutate(Survived = case_when(Survived==1 ~ "Yes", 
                              Survived==0 ~ "No"))


comb.data$SibSp[comb.data$PassengerId==280] = 0
comb.data$Parch[comb.data$PassengerId==280] = 2
comb.data$SibSp[comb.data$PassengerId==1284] = 1
comb.data$Parch[comb.data$PassengerId==1284] = 1

#Title
comb.data$Title <- gsub('(.*, )|(\\..*)', '', comb.data$Name)
# Titles by Sex
table(comb.data$Sex, comb.data$Title)
# Reassign rare titles
officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')
# Reassign mlle, ms, and mme, and rare
comb.data$Title[comb.data$Title == 'Mlle'] <- 'Miss' 
comb.data$Title[comb.data$Title == 'Ms'] <- 'Miss'
comb.data$Title[comb.data$Title == 'Mme'] <- 'Mrs' 
comb.data$Title[comb.data$Title %in% royalty] <- 'Royalty'
comb.data$Title[comb.data$Title %in% officer] <- 'Officer'
# Titles by Sex
table(comb.data$Sex, comb.data$Title)
#----------------------------------------------------------------------------------------------------------------------------------------------------

#Data Types:
str(comb.data)
comb.data$Sex <- factor(comb.data$Sex)
comb.data$Embarked <- factor(comb.data$Embarked)
comb.data$Age_Group <- factor(comb.data$Age_Group)
comb.data$Family_size <- factor(comb.data$Family_size)
comb.data$count <- NULL
comb.data$Survived <- factor(comb.data$Survived)
comb.data$Ticket <- NULL
comb.data$Name <- NULL
comb.data$Pclass <- factor(comb.data$Pclass)
comb.data$Title <- factor(comb.data$Title)
comb.data$SibSp <- as.integer(comb.data$SibSp)
comb.data$Parch <- as.integer(comb.data$Parch)
#----------------------------------------------------------------------------------------------------------------------------------------------------

#Split back to train and test:

train <-comb.data[1:891,]
test <-comb.data[(892):nrow(comb.data),]
test <- test[ , -which(names(test) %in% c("Survived"))]

#----------------------------------------------------------------------------------------------------------------------------------------------------

#Random Forest:
mod_rf <- randomForest(factor(Survived) ~ Pclass + Sex + Fare + Embarked + Title + 
                         Family_size, data=train, na.action = na.exclude, importance = TRUE, ntree = 1000)
mod_rf
varImpPlot(mod_rf)

prediction <- predict(mod_rf, test)
rf_op <- cbind(test$PassengerId, prediction)
colnames(rf_op) <- c("PassengerId", "Survived")
rf_op <- as.data.frame(rf_op)
rf_op$Survived <- rf_op$Survived-1
write.csv(rf_op,"RFSubmission.csv", row.names=FALSE)
#----------------------------------------------------------------------------------------------------------------------------------------------------

myControl <- trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = T, # IMPORTANT!
  verboseIter = TRUE
)

model <- train(factor(Survived) ~ Pclass + Sex + Fare + Embarked + Title + 
                 Family_size, train,method = "glmnet",trControl = myControl)
summary(model)
# Print maximum ROC statistic
max(model[["results"]][["ROC"]])

prediction1 <- predict(model, test)
glmnet_op <- cbind(test$PassengerId, prediction1)
colnames(glmnet_op) <- c("PassengerId", "Survived")
write.csv(glmnet_op,"GLMNETSubmission.csv", row.names=FALSE)

#----------------------------------------------------------------------------------------------------------------------------------------------------

library(e1071)
mod_svm <- svm(factor(Survived) ~ Pclass + Sex + Fare + Embarked + Title + 
                 Family_size,data=train)
svm_pred <- predict(mod_svm,newdata = test)
solution <- data.frame(PassengerId=test$PassengerId,Survived=svm_pred)
solution$Survived <- ifelse(solution$Survived=='No', 0, 1)
write.csv(solution,"svm_solution2.csv",row.names = F)

#----------------------------------------------------------------------------------------------------------------------------------------------------
