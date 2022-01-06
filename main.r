test <- read.csv("~/Downloads/titanic/test.csv")
train <- read.csv("~/Downloads/titanic/train.csv")

#Exploration
mean(train$Survived)  # .38 survive
table(train$Pclass, train$Survived)
plot(train$Age, train$Survived) # looks like you're slightly more likely to survive if you're younger 
table(train$Sex, train$Survived) # much more women than men survived 
table(train$Sex) # interesting, esp given that there were much more women on board 
plot(train$Parch, train$Survived)
plot(train$SibSp, train$Survived) # a lot more likely to die the more siblings you have on board 
table(train$Embarked, train$Survived) # two people who survived w/o embarked station 
table(train$Embarked, train$Pclass) # port and class highly correlated - maybe only need one of these then
plot(train$Fare, train$Survived)

#Cleaning
train$Age[which(is.na(train$Age))] <- mean(train$Age, na.rm = TRUE)
# cabin is too messy, don't use 
train$Sex <- ifelse(train$Sex == "male", 1, 0) # men are 1, women are 0 
train$Sex <- as.numeric(train$Sex)

# New Feature 
train$ageClass <- train$Age * train$Pclass

# subsetted to cleaned 
train_df <- train[,c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "ageClass")]
cor(train$Survived, train_df) # fare, parch, sibsq, age, sex, and ageClass, pclass 

# cleaning test 
test$Age[which(is.na(test$Age))] <- mean(test$Age, na.rm = TRUE)
# cabin is too messy, don't use 
test$Sex <- ifelse(test$Sex == "male", 1, 0) # men are 1, women are 0 
test$Sex <- as.numeric(test$Sex)
test$ageClass <- test$Age * test$Pclass

test_df <- test[,c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "ageClass")]

```

## Training Model


```{r}
library(caret)
library(Matrix)
library(glmnet)

#Scaling 
#train_df$Age <- scale(train_df$Age)
#train_df$SibSp <- scale(train_df$SibSp)
#train_df$Pclass <- scale(train_df$Pclass)
#train_df$Parch <- scale(train_df$Parch)
#train_df$Fare <- scale(train_df$Fare)

#Making Test/Train
trainIndex <- createDataPartition(train_df$Survived, p = .8, 
                                  list = FALSE, 
                                  times = 1)
titanicTrain <- train_df[ trainIndex,]
titanicTest  <- train_df[-trainIndex,]

#without normalization
glm.fit <- glm(titanicTrain$Survived ~ ., family="binomial", data=titanicTrain)
pred <- predict.glm(glm.fit, type="response")
pred_survive <- ifelse(pred >= .5, 1,0)
table(pred_survive, titanicTrain$Survived) # this is a liiiiiiitle off
mean(pred_survive == titanicTrain$Survived) # .79 right 

#train and test 
glm.fit2 <- glm(titanicTrain$Survived ~ ., family="binomial", data=titanicTrain)
pred <- predict.glm(glm.fit2, newdata = titanicTest, type="response")
pred_survive <- ifelse(pred >= .5, 1,0)
table(pred_survive, titanicTest$Survived)
mean(pred_survive == titanicTest$Survived) 

glm.fit3 <- glm(Survived ~ Age + Sex + Pclass + Parch + Fare + SibSp + ageClass + Age:Sex + SibSp:Pclass + Parch:Pclass, family="binomial", data=titanicTrain)
pred <- predict.glm(glm.fit3, newdata = titanicTest, type="response")
pred_survive <- ifelse(pred >= .5, 1,0)
table(pred_survive, titanicTest$Survived)
mean(pred_survive == titanicTest$Survived) 

# Sex:Pclass OUT
# Fare:Parch OUT 
# Fare:SibSp OUT
# Fare:ageClass OUT
# Age:SibSp OUT

#checking model
glm.fit3 <- glm(Survived ~ Age + Sex + Pclass + Parch + Fare + SibSp + ageClass + Age:Parch + Age:Sex + SibSp:Pclass + Parch:Pclass, family="binomial", data=titanicTrain)
pred <- predict.glm(glm.fit3, newdata = titanicTest, type="response")
pred_survive <- ifelse(pred >= .5, 1,0)
table(pred_survive, titanicTest$Survived)
mean(pred_survive == titanicTest$Survived) 

# Final Model to submit!
glm.fit3 <- glm(Survived ~ Age + Sex + Pclass + Parch + Fare + SibSp + ageClass + Age:Parch + Age:Sex + SibSp:Pclass + Parch:Pclass, family="binomial", data=train_df)
pred <- predict.glm(glm.fit3, newdata = test_df, type="response")
pred_survive <- ifelse(pred >= .5, 1,0)


output <- data.frame("PassengerId" = test$PassengerId, "Survived" = pred_survive)
write.csv(output, "submission1.csv", row.names = FALSE)
