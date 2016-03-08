#TV Write-off regression


library(caret)
library(beepr)
library(doParallel)

#setwd('/Users/tamjeff/Documents/R/')
#setwd('D:/Documents/R/TV-WRITEOFF-ML/')
setwd('C:/Users/jgaisano/Documents/TV WRITEOFF ML')

raw = read.csv(file = 'TV write off data by percentage.csv', stringsAsFactors = TRUE)
trainIndex = createDataPartition(raw$Writeoff_Percent_Qty, p = 0.5, list = FALSE, times = 1)
train = raw[trainIndex,]
test = raw[-trainIndex,]

fitControl = trainControl(method = "repeatedCV", number = 10, repeats = 10) 


###Linear Regression
linearmodels = c('blassoAveraged')
#linearmodels = c('blassoAveraged','BstLm','cubist','glmnet','icr','M5Rules','pls','pcr','relaxo','svmLinear','blasso')
linearfitparam = list()
linearresults = list()

for (model in linearmodels){
  cl <- makeCluster(2)
  registerDoParallel(cl)
  fit = train(Writeoff_Percent_Qty~., data = train, method = model, trControl = fitControl)
  1 #install dependencies
  
  linearfitparam[[length(linearfitparam)+1]] = fit
  testpred = predict(fit, test)
  linearresults[[length(linearresults)+1]] = confusionMatrix(testpred, na.omit(test)$Writeoff_Percent_Qty)
  
  save(fit, file = paste0("r",model,"fit.rda"))
  beep(4)
  #stopCluster(cl)
  png(filename = paste0("r",model,"varimp.png"))
  plot(varImp(fit), top = 20)
  dev.off()
  rm(fit)
}


###Trees
treemodels = c()
treefitparam = list()
treeresults = list()


for (model in treemodels){
  cl <- makeCluster(2)
  registerDoParallel(cl)
  fit = train(Writeoff_Percent_Qty~., data = train, method = model, trControl = fitControl)
  1 #install dependencies
  
  treefitparam[[length(treefitparam)+1]] = fit
  testpred = predict(fit, test)
  treeresults[[length(treeresults)+1]] = confusionMatrix(testpred, na.omit(test)$Writeoff_Percent_Qty)
  
  save(fit, file = paste0("r",model,"fit.rda"))
  beep(4)
  #stopCluster(cl)
  png(filename = paste0("r",model,"varimp.png"))
  plot(varImp(fit), top = 20)
  dev.off()
  rm(fit)
}


###Neural Network
nnmodels = c('brnn','elm','avNNet','mlpWeightDecayML','nnet','dnn')
nnfitparam = list()
nnresults = list()


for (model in nnmodels){
  cl <- makeCluster(2)
  registerDoParallel(cl)
  fit = train(Writeoff_Percent_Qty~., data = train, method = model, trControl = fitControl)
  1 #install dependencies
  
  nnfitparam[[length(nnfitparam)+1]] = fit
  testpred = predict(fit, test)
  nnresults[[length(nnresults)+1]] = confusionMatrix(testpred, na.omit(test)$Writeoff_Percent_Qty)
  
  save(fit, file = paste0("r",model,"fit.rda"))
  beep(4)
  #stopCluster(cl)
  png(filename = paste0("r",model,"varimp.png"))
  plot(varImp(fit), top = 20)
  dev.off()
  rm(fit)
}


###Splines

