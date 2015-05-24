# Set working directory
setwd("C:/Users/Daniel/Desktop/Data Science/Coursera/Practical Machine Learning/Course Project")

trainDataURL<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainDataURL, destfile = "./Data/pml-training.csv")
testDataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testDataURL, destfile = "./Data/pml-testing.csv")

preTrain <- read.csv("./Data/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
preTest <- read.csv("./Data/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))

set.seed(2015)

library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)



#Based on the Test set
#Remove columns that  won't contribute to the model (data acquisition)
train <- preTrain[,-c(1,2,3,4,5,6,7)]
#Remove columns with NAs
train <- train[,colSums(is.na(train)) == 0]
#Remove columns with little to no variation
train <- train[,-(nearZeroVar(preTrain))]

#Same for Test
test <- preTest[,as.vector(names(train))]

#Add classe
train$classe <- preTrain$classe

#Partition
t <- createDataPartition(y = train$classe, p=0.7, list=FALSE)
tr <- train[t,]
cv <- train[-t,]

ggplot(data=train, aes(x=classe)) + geom_histogram()

mf <- train(classe ~ ., data=tr, method="rpart")

mfrf <- train(classe ~ ., data = tr, method = "rf", 
              prof = TRUE, 
              trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
confusionMatrix(predict(mfrf,cv), cv$classe)