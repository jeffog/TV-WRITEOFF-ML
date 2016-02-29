#TV Write off prediction using caret

library(caret)
library(beepr)
library(doParallel)

#setwd('/Users/tamjeff/Documents/R/')
setwd('D:/Documents/R/TV-WRITEOFF-ML/')
raw = read.csv(file = 'TV WO Data by Tag.csv', stringsAsFactors = TRUE)
tvnew = raw[raw$FISCAL_YEAR == '2015' | raw$FISCAL_QUARTER == 'Q4_2016',]
rm(raw)

trainIndex = createDataPartition(tvnew$WRITE_OFF_YN, p = 0.5, list = FALSE, times = 1)
tvtrain = tvnew[trainIndex,]
tvtest = tvnew[-trainIndex,]
tvtrain$WRITE_OFF_YN[tvtrain$WRITE_OFF_YN == '1'] = "YES"
tvtrain$WRITE_OFF_YN[tvtrain$WRITE_OFF_YN == '0'] = "NO"
tvtest$WRITE_OFF_YN[tvtest$WRITE_OFF_YN == '1'] = "YES"
tvtest$WRITE_OFF_YN[tvtest$WRITE_OFF_YN == '0'] = "NO"
tvtrain$WRITE_OFF_YN = as.factor(tvtrain$WRITE_OFF_YN)
tvtest$WRITE_OFF_YN = as.factor(tvtest$WRITE_OFF_YN)
excludes = c("FISCAL_PERIOD","FISCAL_MONTH","FISCAL_QUARTER","FISCAL_YEAR","TAG_NO","MARKET")

tvtrain = tvtrain[,!(names(tvtrain) %in% excludes)]
tvtest = tvtest[,!(names(tvtest) %in% excludes)]

fitControl = trainControl(method = "repeatedCV", number = 10, repeats = 10)



###NEURAL NETWORKS

avnnetfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'avNNet', trControl = fitControl)
1
testpred = predict(avnnetfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN) #Hours of training, BUT 86% accuracy!
save(avnnetfit, file = "avnnetfit.rda")
beep(4)
rm(avnnetfit)


###TREES, seems to give best results

rotationForestfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rotationForest', trControl = fitControl)
1
testpred = predict(rotationForestfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(rotationForestfit, file = "rotationForestfit.rda") #86%
beep(2)
rm(rotationForestfit)


rpartCostfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rpartCost', trControl = fitControl)
1
testpred = predict(rpartCostfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  #fast! 86%!
save(rpartCostfit, file = "rpartCostfit.rda")
beep(4)
rm(rpartCostfit)


cl <- makeCluster(2)
registerDoParallel(cl)
xgbTreefit = train(WRITE_OFF_YN~., data = tvtrain, method = 'xgbTree', trControl = fitControl, metric = "Kappa")
1
testpred = predict(xgbTreefit, tvtest) #88%!!!!!!!!!!!!!!!!!
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(xgbTreefit, file = "xgbTreefit.rda")
beep(2)
stopCluster(cl)
rm(xgbTreefit)


cl <- makeCluster(3)
registerDoParallel(cl)
c5fit = train(WRITE_OFF_YN~., data = tvtrain, method = 'c5.0', trControl = fitControl)
1
testpred = predict(c5fit, tvtest) #88%!
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(c5fit, file = "c5fit.rda")
beep(2)
stopCluster(cl)
rm(c5fit)


cl <- makeCluster(2)
registerDoParallel(cl)
J48fit = train(WRITE_OFF_YN~., data = tvtrain, method = 'J48', trControl = fitControl, metric = "Kappa")
1
testpred = predict(J48fit, tvtest) #88%!
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(J48fit, file = "J48fit.rda")
beep(2)
stopCluster(cl)
rm(J48fit)


cl <- makeCluster(2)
registerDoParallel(cl)
c5Costfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'C5.0Cost', trControl = fitControl, metric = "Kappa")
1
testpred = predict(c5Costfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(c5Costfit, file = "c5Costfit.rda")
beep(2)
stopCluster(cl)
rm(c5Costfit)


###Logistic Regression
cl <- makeCluster(3)
registerDoParallel(cl)
logicBagfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'logicBag', trControl = fitControl)
1
testpred = predict(logicBagfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(logicBagfit, file = "logicBagfit.rda")
beep(2)
stopCluster(cl)
rm(logicBagfit)


cl <- makeCluster(3)
registerDoParallel(cl)
LMTfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'LMT', trControl = fitControl)
1
testpred = predict(LMTfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN, positive = "YES")  
save(LMTfit, file = "LMTfit.rda")
beep(2)
stopCluster(cl)
rm(LMTfit)