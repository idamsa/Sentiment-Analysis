source("Optional Preprocessing, Feature Engineering.R")

# We found Decision Tree to be best model for samsung and iPhone so we will now try to use the model on the feature
# engenieered models that we built ( COR, NZV and RFE)

# Data Split ----

# 1. COR ---- 
# Partition data iPhone COR: 70% of the data to train the model
inTrainiPhoneCOR <- createDataPartition(y = iPhoneCOR$iphonesentiment, p = 0.70, list = FALSE)
trainingiPhoneCOR <- iPhoneCOR[inTrainiPhoneCOR,]
testingiPhoneCOR <- iPhoneCOR[-inTrainiPhoneCOR,] 

# Partition data Samsung COR: 70% of the data to train the model
inTrainSamsungCOR <- createDataPartition(y = samsungCOR$galaxysentiment, p = 0.70, list = FALSE)
trainingSamsungCOR <- samsungCOR[inTrainSamsungCOR,]
testingSamsungCOR <- samsungCOR[-inTrainSamsungCOR,] 

# 2. NZV ----
# Partition data iPhone NZV: 70% of the data to train the model
inTrainiPhoneNZV <- createDataPartition(y = iphoneNZV$iphonesentiment, p = 0.70, list = FALSE)
trainingiPhoneNZV <- iphoneNZV[inTrainiPhoneNZV,]
testingiPhoneNZV <- iphoneNZV[-inTrainiPhoneNZV,] 

# Partition data Samsung NZV: 70% of the data to train the model
inTrainSamsungNZV <- createDataPartition(y = samsungNZV$galaxysentiment, p = 0.70, list = FALSE)
trainingSamsungNZV <- samsungNZV[inTrainSamsungNZV,]
testingSamsungNZV <- samsungNZV[-inTrainSamsungNZV,] 

# 3. RFE ----
# Partition data iPhone NZV: 70% of the data to train the model
inTrainiPhoneRFE <- createDataPartition(y = iPhoneRFE$iphonesentiment, p = 0.70, list = FALSE)
trainingiPhoneRFE <- iPhoneRFE[inTrainiPhoneRFE,]
testingiPhoneRFE <- iPhoneRFE[-inTrainiPhoneRFE,] 

# Partition data Samsung NZV: 70% of the data to train the model
inTrainSamsungRFE <- createDataPartition(y = samsungRFE$galaxysentiment, p = 0.70, list = FALSE)
trainingSamsungRFE <- samsungRFE[inTrainSamsungRFE,]
testingSamsungRFE <- samsungRFE[-inTrainSamsungRFE,]

# Apply model on all datasets and test them 

# Train Control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = T, summaryFunction = mnLogLoss, classProbs = T)

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# 1. Model for COR Datasets ----

# 1.1. iPhone COR ----

DecTreeModelCORiPhone <- train(iphonesentiment ~ .,
                               data =trainingiPhoneCOR, 
                               method = "C5.0",
                               metric="logLoss",
                               trControl= ctrl)

# Test c50 COR iPhone Accuracy : 0.7213,  Kappa : 0.4315

DecTreePredCORiPhone <- predict(DecTreeModelCORiPhone, newdata = testingiPhoneCOR)

print(cmDecTreeCORiPhone <- confusionMatrix(DecTreePredCORiPhone, testingiPhoneCOR$iphonesentiment))

# saveRDS(DecTreeModelCORiPhone,"DecTreeModelCORiPhone.RDS")

# 1.2. C5.0 COR Samsung  

DecTreeModelCORSamsung <- train(galaxysentiment ~ .,
                                data =trainingSamsungCOR, 
                                method = "C5.0",
                                trControl= ctrl)

# Test c50 COR Samsung   Accuracy : 0.6534,  Kappa : 0.215

DecTreePredCORSamsung <- predict(DecTreeModelCORSamsung, newdata = testingSamsungCOR)

print(cmDecTreeCORSamsung <- confusionMatrix(DecTreePredCORSamsung, testingSamsungCOR$galaxysentiment))

# saveRDS(DecTreeModelCORSamsung,"DecTreeModelCORSamsung.RDS")

# 2. Model for NZV Datasets ----

# 2.1. iPhone NZV ----

DecTreeModelNZViPhone <- train(iphonesentiment ~ .,
                               data =trainingiPhoneNZV, 
                               method = "C5.0",
                               trControl= ctrl)

# Test c50 NZV iPhone Accuracy : 0.7201 , Kappa : 0.4315 

DecTreePredNZViPhone <- predict(DecTreeModelNZViPhone, newdata = testingiPhoneNZV)

print(cmDecTreeNZViPhone <- confusionMatrix(DecTreePredNZViPhone, testingiPhoneNZV$iphonesentiment))

# saveRDS(DecTreeModelNZViPhone,"DecTreeModelNZViPhone.RDS")

# 2.2. C5.0 NZV Samsung  

DecTreeModelNZVSamsung <- train(galaxysentiment ~ .,
                                data =trainingSamsungNZV, 
                                method = "C5.0",
                                trControl= ctrl)

# Test c50 NZV Samsung   Accuracy : 0.6421, Kappa : 0.1662  

DecTreePredNZVSamsung <- predict(DecTreeModelNZVSamsung, newdata = testingSamsungNZV)

print(cmDecTreeNZVSamsung <- confusionMatrix(DecTreePredNZVSamsung, testingSamsungNZV$galaxysentiment))

# saveRDS(DecTreeModelNZVSamsung,"DecTreeModelNZVSamsung.RDS")

# 3. Model for RFE Datasets ----

# 3.1. iPhone RFE ----

DecTreeModelRFEiPhone <- train(iphonesentiment ~ .,
                               data =trainingiPhoneRFE, 
                               method = "C5.0",
                               trControl= ctrl)

# Test c50 RFE iPhone Accuracy : 0.7226  ,  Kappa : 0.4365 

DecTreePredRFEiPhone <- predict(DecTreeModelRFEiPhone, newdata = testingiPhoneRFE)

print(cmDecTreeRFEiPhone <- confusionMatrix(DecTreePredRFEiPhone, testingiPhoneRFE$iphonesentiment))

# saveRDS(DecTreeModelRFEiPhone,"DecTreeModelRFEiPhone.RDS")

# 2.2. C5.0 RFE Samsung  

DecTreeModelRFESamsung <- train(galaxysentiment ~ .,
                                data =trainingSamsungRFE, 
                                method = "C5.0",
                                trControl= ctrl)

# Test c50 RFE Samsung   Accuracy : 0.6518 , Kappa : 0.2093  

DecTreePredRFESamsung <- predict(DecTreeModelRFESamsung, newdata = testingSamsungRFE)

print(cmDecTreeRFESamsung <- confusionMatrix(DecTreePredRFESamsung, testingSamsungRFE$galaxysentiment))

# saveRDS(DecTreeModelRFESamsung,"DecTreeModelRFESamsung.RDS")

stopCluster(cl)

###CONCLUSIONS
# Even with feature engenieering done by correlation, nearzero variance and recursive feature elimination the models
# still suffer from the class imbalance. In order to better this we need to solve the clas imbalace
# for iphone : c50 RFE iPhone Accuracy : 0.7226  ,  Kappa : 0.4365 RFE FEATURED better than OOB
# for samsung : oob samsung Accuracy : 0.6544 ,  Kappa : 0.2164 still better than any of above



