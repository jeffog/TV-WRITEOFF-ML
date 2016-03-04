#TV Write off prediction using caret

library(caret)
library(beepr)
library(doParallel)

#setwd('/Users/tamjeff/Documents/R/')
#setwd('D:/Documents/R/TV-WRITEOFF-ML/')
setwd('C:/Users/jgaisano/Documents/TV WRITEOFF ML')


raw = read.csv(file = 'TV WO Data by Tag.csv', stringsAsFactors = TRUE)
tvnew = raw[raw$FISCAL_YEAR == '2015' | raw$FISCAL_YEAR == '2016',]
tvnew$WRITE_OFF_YN = as.factor(tvnew$WRITE_OFF_YN)
rm(raw)

trainIndex = createDataPartition(tvnew$WRITE_OFF_YN, p = 0.5, list = FALSE, times = 1)
tvtrain = tvnew[trainIndex,]
tvtest = tvnew[-trainIndex,]

excludes = c("FISCAL_PERIOD","FISCAL_MONTH","FISCAL_QUARTER","FISCAL_YEAR","TAG_NO","MARKET")
tvtrain = tvtrain[,!(names(tvtrain) %in% excludes)]
tvtest = tvtest[,!(names(tvtest) %in% excludes)]

fitControl = trainControl(method = "repeatedCV", number = 10, repeats = 10,
                          adaptive = list(min = 10,
                                          alpha = 0.05,
                                          method = "gls",
                                          complete = TRUE)) #adaptive resampling helps find best parameters



###NEURAL NETWORKS
cl <- makeCluster(2)
registerDoParallel(cl)
avNNetfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'avNNet', trControl = fitControl, metric = "Kappa")
1
testpred = predict(avNNetfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)
save(avNNetfit, file = "avNNetfit.rda")
beep(4)
stopCluster(cl)
png(filename = "avNNetfit.png")
plot(varImp(avNNetfit), top = 20)
dev.off()
rm(avNNetfit)


cl <- makeCluster(2)
registerDoParallel(cl)
nnetfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'nnet', trControl = fitControl, metric = "Kappa")
1
testpred = predict(nnetfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)
save(nnetfit, file = "nnetfit.rda")
beep(4)
stopCluster(cl)
png(filename = "nnetfit.png")
plot(varImp(nnetfit), top = 20)
dev.off()
rm(nnetfit)


###TREES
cl <- makeCluster(2)
registerDoParallel(cl)
rotationForestfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rotationForest', trControl = fitControl)
1
testpred = predict(rotationForestfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
stopCluster(cl)
save(rotationForestfit, file = "rotationForestfit.rda") 
png(filename = "rotationForestfit.png")
plot(varImp(rotationForestfit), top = 20)
dev.off()
beep(2)
rm(rotationForestfit)


cl <- makeCluster(2)
registerDoParallel(cl)
rpartCostfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rpartCost', trControl = fitControl)
1
testpred = predict(rpartCostfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)
stopCluster(cl)
save(rpartCostfit, file = "rpartCostfit.rda")
png(filename = "rpartCostfit.png")
plot(varImp(rpartCostfit), top = 20)
dev.off()
beep(4)
rm(rpartCostfit)


cl <- makeCluster(2)
registerDoParallel(cl)
xgbTreefit = train(WRITE_OFF_YN~., data = tvtrain, method = 'xgbTree', trControl = fitControl, metric = "Kappa")
1
testpred = predict(xgbTreefit, tvtest) #89% Acc, 39% Kappa
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(xgbTreefit, file = "xgbTreefit2.rda")
beep(2)
stopCluster(cl)
png(filename = "xgbTreefit.png")
plot(varImp(xgbTreefit), top = 20)
dev.off()
rm(xgbTreefit)


cl <- makeCluster(2)
registerDoParallel(cl)
c5fit = train(WRITE_OFF_YN~., data = tvtrain, method = 'C5.0', trControl = fitControl)
1
testpred = predict(c5fit, tvtest) #89%
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(c5fit, file = "c5fit.rda")
beep(2)
stopCluster(cl)
png(filename = "c5fit.png")
plot(varImp(c5fit), top = 20)
dev.off()
rm(c5fit)


cl <- makeCluster(2)
registerDoParallel(cl)
J48fit = train(WRITE_OFF_YN~., data = tvtrain, method = 'J48', trControl = fitControl, metric = "Kappa")
1
testpred = predict(J48fit, tvtest) 
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(J48fit, file = "J48fit.rda")
beep(2)
stopCluster(cl)
png(filename = "J48fit.png")
plot(varImp(J48fit), top = 20)
dev.off()
rm(J48fit)


cl <- makeCluster(2)
registerDoParallel(cl)
c5Costfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'C5.0Cost', trControl = fitControl, metric = "Kappa")
1
testpred = predict(c5Costfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(c5Costfit, file = "c5Costfit.rda")
png(filename = "c5Costfit.png")
plot(varImp(c5Costfit), top = 20)
dev.off()
beep(2)
stopCluster(cl)
rm(c5Costfit)



cl <- makeCluster(2)
registerDoParallel(cl)
adafit = train(WRITE_OFF_YN~., data = tvtrain, method = 'ada', trControl = fitControl, metric = "Kappa")
1
testpred = predict(adafit, tvtest) 
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(adafit, file = "adafit.rda")
beep(2)
stopCluster(cl)
png(filename = "adafit.png")
plot(varImp(adafit), top = 20)
dev.off()
rm(adafit)


cl <- makeCluster(2)
registerDoParallel(cl)
AdaBoostM1fit = train(WRITE_OFF_YN~., data = tvtrain, method = 'AdaBoost.M1', trControl = fitControl, metric = "Kappa")
1
testpred = predict(AdaBoostM1fit, tvtest) 
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(adafit, file = "AdaBoostM1fit.rda")
beep(2)
stopCluster(cl)
png(filename = "AdaBoostM1fit.png")
plot(varImp(AdaBoostM1fit), top = 20)
dev.off()
rm(AdaBoostM1fit)



###Logistic Regression, no luck
cl <- makeCluster(2)
registerDoParallel(cl)
logicBagfit = train(WRITE_OFF_YN~., data = na.omit(tvtrain), method = 'logicBag', trControl = fitControl)
1
testpred = predict(logicBagfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(logicBagfit, file = "logicBagfit.rda")
beep(2)
stopCluster(cl)
png(filename = "logicBagfit.png")
plot(varImp(logicBagfit), top = 20)
dev.off()
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


###SVM, no luck
cl <- makeCluster(2)
registerDoParallel(cl)
lssvmPolyfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'lssvmPoly', trControl = fitControl, metric = "Kappa")
1
testpred = predict(lssvmPolyfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)
save(lssvmPolyfit, file = "lssvmPolyfit.rda")
beep(2)
stopCluster(cl)
png(filename = "lssvmPolyfit.png")
plot(varImp(lssvmPolyfit), top = 20)
dev.off()
rm(lssvmLinearfit)
