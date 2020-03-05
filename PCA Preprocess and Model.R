source("Open and Preprocess Sentiment Analysis.R")
### 1.1. PCA FOR IPHONE FULL PIPE ----

# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParamsiPhone <- preProcess(trainingiPhone[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsiPhone)

# use predict to apply pca parameters, create training, exclude dependant
train.pca.iPhone <- predict(preprocessParamsiPhone, trainingiPhone[,-59])

# add the dependent to training
train.pca.iPhone$iphonesentiment <- trainingiPhone$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca.iPhone <- predict(preprocessParamsiPhone, testingiPhone[,-59])

# add the dependent to training
test.pca.iPhone$iphonesentiment <- testingiPhone$iphonesentiment

# inspect results
str(train.pca.iPhone)
str(test.pca.iPhone)

#1.2.Model pca iPhone + smote + log loss ----

# set up paralell processing
cl <- makeCluster(3)
registerDoParallel(cl)

# train control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = T, summaryFunction = mnLogLoss, classProbs = T, sampling = "smote")

DecTreeModelpcaiPhone<- train(iphonesentiment ~ .,
                                data =train.pca.iPhone, 
                                method = "C5.0",
                                metric="logLoss",
                                trControl= ctrl)

# Test 

DecTreeModelpcaiPhone <- predict(DecTreeModelpcaiPhone, newdata = test.pca.iPhone)

print(cmDDecTreeModelpcaiPhone <- confusionMatrix(DecTreeModelpcaiPhone, test.pca.iPhone$iphonesentiment,positive = "POS"))

stopCluster(cl)

### 2.1. PCA FOR samsungFULL PIPE ----

# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParamssamsung <- preProcess(trainingSamsung[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsiPhone)

# use predict to apply pca parameters, create training, exclude dependant
train.pca.samsung <- predict(preprocessParamssamsung, trainingSamsung[,-59])

# add the dependent to training
train.pca.samsung$galaxysentiment <- trainingSamsung$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca.samsung <- predict(preprocessParamssamsung, testingSamsung[,-59])

# add the dependent to training
test.pca.samsung$galaxysentiment <- testingSamsung$galaxysentiment

# inspect results
str(train.pca.samsung)
str(test.pca.samsung)

#2..2.Model pca samsung + smote + log loss ----

# set up paralell processing
cl <- makeCluster(3)
registerDoParallel(cl)

# train control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = T, summaryFunction = mnLogLoss, classProbs = T, sampling = "smote")

DecTreeModelpcasamsung<- train(galaxysentiment ~ .,
                              data =train.pca.samsung, 
                              method = "C5.0",
                              metric="logLoss",
                              trControl= ctrl)

# Test 

DecTreeModelpcasamsung <- predict(DecTreeModelpcasamsung, newdata = test.pca.samsung)

print(cmDDecTreeModelpcasamsung <- confusionMatrix(DecTreeModelpcasamsung, test.pca.samsung$galaxysentiment,positive = "POS"))

