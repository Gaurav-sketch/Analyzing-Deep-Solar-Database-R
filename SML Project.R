#Load the packages 
install.packages("adabag")
#load the packages required.
library(rpart) # for classification tree
library(bestglm) #
library(nnet)  # for logisitc regression
library(randomForest) # for random forest

library(adabag) #Bagging and boosting

#Load the data .
data<-read.csv("E://Spring Semester 2//Statistical Machine Learning//Projects//data_project_deepsolar.csv",header=TRUE)
dim(data)
#Scale the data.
data[,-c(1,2,76,79)]<-scale(data[,-c(1,2,76,79)])

#This is to convert the variable to factors.
data[,76]<-as.factor(data$voting_2016_dem_win)
data[,79]<-as.factor(data$voting_2012_dem_win)

#This is just to check the structure of the data.
str(data)

#Change the levels to binary values.
data$solar_system_count<-factor(data$solar_system_count,levels = c("low","high"),labels = c(0,1))
levels(data$solar_system_count)


#EDA_Explorartory Data Analysis
install.packages("mlbench")
library(mlbench)
library(caret)
correlationMatrix<-cor(data[,-c(1,2,76,79)])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
highlyCorrelated
print(highlyCorrelated)
ndata<-data[,-c(1,2,76,79)][,-highlyCorrelated]
Final_data<-cbind(data[,c(1,2,76,79)],ndata)
Final_data

#To check any variable that is highly insignificant and should be removed. 
fit<-glm(solar_system_count ~ .,data=Final_data,family = "binomial")
summary(fit)
#From logistic regression we got that 5 variables are showing singularity
Final_data<-Final_data[,-c(34:38)]

#The final data used for further analysis.
dim(Final_data)
head(data)
#Some of the variables are not defined because of singularity means that the 
#variables are not linearly independent. If you remove the variables that are giving NA in the above summary, you will obtain the same result for the rest of the variables. This is because the information given by those variables is already contained in the other variables and thus redundant.
#check the prediction of the model

N <- nrow(Final_data)
train <- sample(1:N, size = 0.50*N) # 70% of data used as training data
val <- sample( setdiff(1:N, train), size = 0.25*N ) # 15% of data used as validation data
test <- setdiff(1:N, union(train, val))  # 15% of data used as test data

#This is used for training and model comparison.
dat_train <- Final_data[train,]

#This is used for validation.
dat_val<- Final_data[val,]

#This is employed for testing.
dat_test <- Final_data[test,]

#Implement the logistic algorithm
fit<-glm(solar_system_count ~ .,data=dat_train,family = "binomial")

#Implement the random forest algorithm
fitrf<-randomForest(solar_system_count ~ .,data = dat_train ,importance=TRUE)


#classification tree
fitct <- rpart(solar_system_count ~ ., data = dat_train)

#SVM Method 
fitSvm <- ksvm(solar_system_count ~ ., data = dat_train, type = "C-svc")

#Bagging
fitbag <- bagging(solar_system_count ~ ., data = dat_train)
install.packages("e1071")
library(kernlab)
#Boosting
fitboost <- boosting(solar_system_count ~ ., data = dat_train,coeflearn = "Breiman", boos = TRUE)

#To check for the best tau value 
#Reciever Operating Characteristic (ROC) Curve.
library(ROCR)

#This is to create the prediction object.
check<-predict(fit,type="response",newdata= dat_val)
predObj <- prediction(check, dat_val$solar_system_count)

predObj
fit$
fitted(fit)
#Below is to create the performance object. 
perf <- performance(predObj, "tpr", "fpr")
perf

#This is to plot the True positive rate vs False positive rate.
plot(perf,colorize=T,ylab="Sensitivity",xlab="1-Specificity",main="ROC Curve : Logistic regression")
abline(0,1, col = "darkorange2", lty = 2) # add bisect line

# compute the area under the ROC curve
auc <- performance(predObj, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)

#Sensitivity/True positive rate
sens <- performance(predObj, "sens")

#Specificity/True negative rate
spec <- performance(predObj, "spec")
tau <- sens@x.values[[1]]
sensSpec <- sens@y.values[[1]] + spec@y.values[[1]]

#This is to get the index of the max sensSpec
best <- which.max(sensSpec)

#This is to plot the sensitivity vs specificity.
plot(tau, sensSpec, type = "l",ylab="Sensitivity+specificity",xlab="Threshold(tau)")
points(tau[best], sensSpec[best], pch = 19, col = adjustcolor("darkorange2", 0.5))
abline(h=sensSpec[best],v=tau[best],col = "red", lty = 2)

#This is the selection of best tau(cutoff value)
tau[best]


#ROC Curve Classification Tree
phat<-predict(fitct,newdata= dat_val)
head(phat)
predObj<-prediction(phat[,2],dat_val$solar_system_count)
roc <- performance(predObj, "tpr", "fpr")
plot(roc,ylab="Sensitivity",xlab="1-Specificity",colorize=T,main="ROC Curve : Classification Tree")
abline(0,1, col = "darkorange2", lty = 2)
auc <- performance(predObj, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)

#ROC Curve for random forest
install.packages("pROC")
library("pROC")
par(pty="m")

#ROC Curve for Random forest 
check2<-predict(fitrf, type = "vote", dat_val)
check2
pobj<-prediction(check2[,2],dat_val$solar_system_count)
rocrf <- performance(pobj, "tpr", "fpr")

plot(rocrf,ylab="Sensitivity",xlab="1-Specificity",main="ROC Curve : Random Forest",colorize=T)
abline(0,1, col = "darkorange2", lty = 2)
auc <- performance(pobj, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)



#ROC curve for SVM

ypredscore <- predict(fitSvm, dat_val, type = "decision")
predObj<-prediction(ypredscore,dat_val$solar_system_count)
roc <- performance(predObj, "tpr", "fpr")
plot(roc,ylab="Sensitivity",xlab="1-Specificity",main="ROC Curve : SVM",colorize=T)
abline(0,1, col = "darkorange2", lty = 2)
auc <- performance(predObj, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)

#ROC Curve for Bagging 
test= predict(fitbag ,type="prob",newdata = dat_val)
bagpred = prediction(test$prob[,2], dat_val$solar_system_count)
bagperf = performance(bagpred, "tpr","fpr")
plot(bagperf,ylab="Sensitivity",xlab="1-Specificity",main="ROC Curve : Bagging",colorize=T)
abline(0,1, col = "darkorange2", lty = 2)
auc <- performance(bagpred, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)



#ROC Curve for Boosting
test2= predict(fitboost ,type="prob",newdata = dat_val)
boosting = prediction(test2$prob[,2], dat_val$solar_system_count)
boosting2 = performance(boosting, "tpr","fpr")
plot(boosting2,ylab="Sensitivity",xlab="1-Specificity",main="ROC Curve : Boosting")
abline(0,1, col = "darkorange2", lty = 2)
auc <- performance(boosting, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)


plot(lr)

# compute the area under the ROC curve
auc <- performance(predObj, "auc")
auc@y.values
legend(.6,.2,auc@y.values,title="AUC",cex=1)

#Accuracy on Validation data for logistic regression
pdl<-predict(fit,type = "response",newdata = dat_val)
pdl<- ifelse(pdl > 0.5, 1, 0)
original<-dat_val$solar_system_count
prediction<-pdl
tab_val <- table(original,prediction)
tab_val
# compute accuracy
accl<-sum( diag(tab_val) ) / sum(tab_val)


#Accuracy on validation data for random forest 
pred_val<-predict(fitrf,dat_val)
tab2_val<-table(dat_val$solar_system_count,pred_val)
tab2_val
accrf<-sum( diag(tab2_val) ) / sum(tab2_val)

#Accuracy on validation data for classification tree
predct <- predict(fitct, type = "class", newdata = dat_val) # classification tree
tab1 <- table(dat_val$solar_system_count, predct)
tab1
accct <- sum(diag(tab1))/sum(tab1)

#Accuracy on Validation data for Support vector machines 
predValSvm <- predict(fitSvm, newdata = dat_val)
tabValSvm <- table(dat_val$solar_system_count, predValSvm)
tabValSvm
accSvm <- sum(diag(tabValSvm))/sum(tabValSvm)

acc#Accuracy on Validation data for Bagging.
predTestBag <- predict(fitbag, newdata = dat_val)
predTestBag[c("confusion", "error")]
bgacc<-1-predTestBag$error

#Accuracy on Validation data for Boosting.
predTestBoost <- predict(fitboost, newdata = dat_val)
predTestBoost[c("confusion", "error")]
boacc<-1-predTestBoost$error

#accuracy
acc <- c(logistic = accl, ran_forest = accrf,classification=accct,SVM =accSvm ,bagging=bgacc,boosting=boacc)
best<-which.max(acc)
best

#Testing the classification of test data using best Random forest model
predTestrf <- predict(fitrf, type = "class", dat_test)
tabTestrf <- table(dat_test$solar_system_count, predTestrf)
accBest <- sum(diag(tabTestrf))/sum(tabTestrf)




# Below is to use the method that did best on the validation data 
# to predict the test data
best <- names( which.max(acc) )
switch(best,
       logistic = {
         predTestl <- predict(fit,type = "response",newdata = dat_test)
         predTest<- ifelse(predTestl > 0.5, 1, 0)
         tabTestl <- table(dat_test$solar_system_count, predTest)
         accBest <- sum(diag(tabTestl))/sum(tabTestl)
         
       },
       ran_forest = {
         predTestrf <- predict(fitrf, type = "class", dat_test)
         tabTestrf <- table(dat_test$solar_system_count, predTestrf)
         accBest <- sum(diag(tabTestrf))/sum(tabTestrf)
         
       },
       classification={   
         predct <- predict(fitct, type = "class", newdata = dat_test) # classification tree
         tab1 <- table(dat_test$solar_system_count, predct)
         accBest<- sum(diag(tab1))/sum(tab1)
       },
       SVM={
         predtestSvm <- predict(fitSvm, newdata = dat_test)
         tabtestSvm <- table(dat_test$solar_system_count, predtestSvm)
         accBest<- sum(diag(tabtestSvm))/sum(tabtestSvm)
       },
       Bagging={
         predTestBag <- predict(fitbag, newdata = dat_test)
         predTestBag[c("confusion", "error")]
         bgacc<-1-predTestBag$error
       },
       Boosting={
         predTestBoost <- predict(fitboost, newdata = dat_test)
         predTestBoost[c("confusion", "error")]
         accBest<-1-predTestBoost$error
       }
)      
best
accBest


#replicate the process 100 number of times to find the best classifier
R <-100

#This holds the validation test result of all methods
#This includes the best method name and its accuracy
out <- matrix(NA, R, 8)

#Store the test results for all methods.
testR<-matrix(NA, R, 6)
colnames(out) <- c("val_l", "val_rf","val_ct","Val_svm" ,"val_bag","val_bost","best","test")
out <- as.data.frame(out)

for ( r in 1:R ) {
  
  # split the data into training ,validation and test sets
  N <- nrow(Final_data)
  train <- sample(1:N, size = 0.70*N) # 70% of data used as training data
  val <- sample( setdiff(1:N, train), size = 0.15*N ) # 15% of data used as validation data
  test <- setdiff(1:N, union(train, val))  # 15% of data used as test data
  
  #This is used for training and model comparison.
  dat_train <- Final_data[train,]
  
  #This is used for validation.
  dat_val<- Final_data[val,]
  
  #This is employed for testing.
  dat_test <- Final_data[test,]
  
  #Logistic regression.
  fit<-glm(solar_system_count ~ .,data=dat_train,family = "binomial")
  

  #Using random forest to classification of images.
  #install.packages("randomForest")
  #Implement the random forest algorithm
  fitrf<-randomForest(solar_system_count ~ .,data = dat_train ,importance=TRUE)
  
  
  #classification tree
  fitct <- rpart(solar_system_count ~ ., data = dat_train)

  #SVM Method 
  fitSvm <- ksvm(solar_system_count ~ ., data = dat_train)
  
  #Bagging
  fitbag <- bagging(solar_system_count ~ ., data = dat_train)
  
  #Boosting
  fitboost <- boosting(solar_system_count ~ ., data = dat_train,coeflearn = "Breiman", boos = FALSE)
  
  #Accuracy on Validation data for logistic regression
  pdl<-predict(fit,type = "response",newdata = dat_val)
  pdl<- ifelse(pdl > 0.5, 1, 0)
  tab_val <- table(dat_val$solar_system_count , pdl)
  # compute accuracy
  accl<-sum( diag(tab_val) ) / sum(tab_val)
  
  
  #Accuracy on validation data for random forest 
  pred_val<-predict(fitrf,dat_val)
  tab2_val<-table(dat_val$solar_system_count,pred_val)
  accrf<-sum( diag(tab2_val) ) / sum(tab2_val)
  
  #Accuracy on validation data for classification tree
  predct <- predict(fitct, type = "class", newdata = dat_val) # classification tree
  tab1 <- table(dat_val$solar_system_count, predct)
  accct <- sum(diag(tab1))/sum(tab1)
  
  #Accuracy on Validation data for Support vector machines 
  predValSvm <- predict(fitSvm, newdata = dat_val)
  tabValSvm <- table(dat_val$solar_system_count, predValSvm)
  accSvm <- sum(diag(tabValSvm))/sum(tabValSvm)
  
  #Accuracy on Validation data for Bagging method 
  predTestBag <- predict(fitbag, newdata = dat_val)
  predTestBag[c("confusion", "error")]
  bgacc<-1-predTestBag$error
  
  #Accuracy on Validation data for Boosting method 
  predTestBoost <- predict(fitboost, newdata = dat_val)
  predTestBoost[c("confusion", "error")]
  boacc<-1-predTestBoost$error
  
  #accuracy
  acc <- c(logistic = accl, ran_forest = accrf,classification=accct,SVM =accSvm)
  #This is to store all the accuracy for validation data
  out[r,1] <- accl
  out[r,2] <- accrf
  out[r,3]<- accct
  out[r,4]<- accSvm
  out[r,5]<- bgacc
  out[r,6]<- boacc
  
  # Below is to use the method that did best on the validation data 
  # to predict the test data
  best <- names( which.max(acc) )
  switch(best,
         logistic = {
           predTestl <- predict(fit,type = "response",newdata = dat_test)
           predTest<- ifelse(predTestl > 0.5, 1, 0)
           tabTestl <- table(dat_test$solar_system_count, predTest)
           accBest <- sum(diag(tabTestl))/sum(tabTestl)
           
         },
         ran_forest = {
           predTestrf <- predict(fitrf, type = "class", dat_test)
           tabTestrf <- table(dat_test$solar_system_count, predTestrf)
           accBest <- sum(diag(tabTestrf))/sum(tabTestrf)
           
         },
         classification={   
           predct <- predict(fitct, type = "class", newdata = dat_test) # classification tree
           tab1 <- table(dat_test$solar_system_count, predct)
           accBest<- sum(diag(tab1))/sum(tab1)
         },
         SVM={
          predtestSvm <- predict(fitSvm, newdata = dat_test)
          tabtestSvm <- table(dat_test$solar_system_count, predtestSvm)
          accBest<- sum(diag(tabtestSvm))/sum(tabtestSvm)
         },
         Bagging={
           predTestBag <- predict(fitbag, newdata = dat_val)
           predTestBag[c("confusion", "error")]
           bgacc<-1-predTestBag$error
         },
         Boosting={
           predTestBoost <- predict(fitboost, newdata = dat_val)
           predTestBoost[c("confusion", "error")]
           accBest<-1-predTestBoost$error
         }
  )
  
  #This is to record the accuracy for the test data for logistic regression
  predTestl <- predict(fit,type = "response",newdata = dat_test)
  predTest<- ifelse(predTestl > 0.5, 1, 0)
  tabTestl <- table(dat_test$solar_system_count, predTest)
  ltest<- sum(diag(tabTestl))/sum(tabTestl)
  
  #Random forest
  predTestrf <- predict(fitrf, type = "class", dat_test)
  tabTestrf <- table(dat_test$solar_system_count, predTestrf)
  lrf<-sum(diag(tabTestrf))/sum(tabTestrf)

  #Classification tree
  predct <- predict(fitct, type = "class", newdata = dat_test) # classification tree
  tab1 <- table(dat_test$solar_system_count, predct)
  lct<-sum(diag(tab1))/sum(tab1)

  #Support vector machines
  predtestSvm <- predict(fitSvm, newdata = dat_test)
  tabtestSvm <- table(dat_test$solar_system_count, predtestSvm)
  lsvm<-sum(diag(tabtestSvm))/sum(tabtestSvm)
  
  #Bagging
  predTestBag <- predict(fitbag, newdata = dat_test)
  predTestBag[c("confusion", "error")]
  lbga<-1-predTestBag$error
  
  #Boosting
  predTestBoost <- predict(fitboost, newdata = dat_test)
  predTestBoost[c("confusion", "error")]
  lboos<-1-predTestBoost$error

  
  
  
  #Store the best method and its test data accuracy.
  out[r,7] <- best
  out[r,8] <- accBest
  
  #Store the test data accuracy for all the methods.
  testR[r,1]<-ltest
  testR[r,2]<-lrf
  testR[r,3]<-lct
  testR[r,4]<-lsvm
  testR[r,5]<-lbga
  testR[r,6]<-lboos
  print(r)
}

out
testR
colnames(testR) <- c("Logistic_test", "RF_test","CT_Test","SVM_Test" ,"Bagging_test","Boosting_Test")
# check out the error rate summary statistics
table(out[,7])
tapply(out[,8], out[,7], summary)
boxplot(out$test ~ out$best)
stripchart(out$test ~ out$best, add = TRUE, vertical = TRUE,
           method = "jitter", pch = 19, col = adjustcolor("magenta3", 0.2))

#Mean for the accuracy of different methods.
avg <- apply(testR,2,mean) 
head(avg)
colnames(as.matrix(avg)) <- c("Logistic_test", "RF_test","CT_Test","SVM_Test" ,"Bagging_test","Boosting_Test")
avg
testR


#This is to plot the test accuracy for different methods.
matplot(testR, type = "l", lty = c(2,3,4,5,6,7), col = c("red", "blue","orange","green","black","yellow"),
      xlab = "Replications", ylab = "Accuracy")

#Estimated mean line.
abline(h = avg, col = c("red", "blue","orange","green","black","yellow"),lwd=1)

#This is to add legend.
legend("bottomleft", fill = c("red", "blue","orange","green","black","yellow"),
       legend = c("Logistic", "Random Forest","classification Tree","SVM","Bagging","Boosting"), bty = "n")

out1[,1]<-rep(c(out[,1]),20)
out1 <- as.data.frame(out1)
out1 <- matrix(NA, 100, 8)
out1[,2]<-rep(c(out[,2]),20)
out1[,3]<-rep(c(out[,3]),20)
out1[,4]<-rep(c(out[,4]),20)
out1[,5]<-rep(c(out[,5]),20)
out1[,6]<-rep(c(out[,6]),20)
out1[,7]<-rep(c(out[,7]),20)
out1[,8]<-rep(c(out[,8]),20)
out1
out
test
testR1<-matrix(NA, 100, 6)
testR1[,1]<-rep(c(testR[,1]),20)
testR1[,2]<-rep(c(testR[,2]),20)
testR1[,3]<-rep(c(testR[,3]),20)
testR1[,4]<-rep(c(testR[,4]),20)
testR1[,5]<-rep(c(testR[,5]),20)
testR1[,6]<-rep(c(testR[,6]),20)
testR1

plot(as.party(fitct), cex = 0.1)
install.packages("partykit")
library("partykit")


library(partykit)
par( mfrow = c(2,3) )
for ( j in sample(1:length(fitbag$trees), 6) ) {
  plot(fitbag$trees[[j]], main = paste("Example ", j) )
  text(fitbag$trees[[j]], use.n = TRUE, xpd = TRUE, col = "deepskyblue3")
}



