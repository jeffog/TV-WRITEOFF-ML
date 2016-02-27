library(caret)
library(beepr)

setwd('/Users/tamjeff/Documents/R/')
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

avnnetfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'avNNet', trControl = fitControl)
1
testpred = predict(avnnetfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN) #Hours of training, BUT 86% accuracy!
save(avnnetfit, file = "avnnetfit.rda")
beep(4)
rm(avnnetfit)


pcaNNetfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'pcaNNet', trControl = fitControl)
1
testpred = predict(pcaNNetfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN) #6 hrs, 85%
save(pcaNNetfit, file = "pcaNNetfit.rda")
beep(4)
rm(pcaNNetfit)

#load("pcaNNetfit.rda")
#load("avnnetfit.rda")
#testpred1 = predict(pcaNNetfit, tvtest)
#testpred2 = predict(avnnetfit, tvtest)
#confusionMatrix(testpred1, na.omit(tvtest)$WRITE_OFF_YN)
#confusionMatrix(testpred2, na.omit(tvtest)$WRITE_OFF_YN)

rpartCostfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rpartCost', trControl = fitControl)
1
testpred = predict(rpartCostfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  #fast! 86%!
save(rpartCostfit, file = "rpartCostfit.rda")
beep(4)
rm(rpartCostfit)

rotationForestfit = train(WRITE_OFF_YN~., data = tvtrain, method = 'rotationForest', trControl = fitControl)
1
testpred = predict(rotationForestfit, tvtest)
confusionMatrix(testpred, na.omit(tvtest)$WRITE_OFF_YN)  
save(rotationForestfit, file = "rotationForestfit.rda")
beep(2)
rm(rotationForestfit)
