source("Optional Preprocessing, Feature Engineering.R")

# Data Split
# Partition data iPhone OOB: 70% of the data to train the model
inTrainiPhone <- createDataPartition(y = iPhone$iphonesentiment, p = 0.70, list = FALSE)
trainingiPhone <- iPhone[inTrainiPhone,]
testingiPhone <- iPhone[-inTrainiPhone,] 

# Partition data Samsung OOB: 70% of the data to train the model
inTrainSamsung <- createDataPartition(y = samsung$galaxysentiment, p = 0.70, list = FALSE)
trainingSamsung <- samsung[inTrainSamsung,]
testingSamsung <- samsung[-inTrainSamsung,] 


# Out of the Box Models ----
# Train Control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = T)

# set up paralell
detectCores()

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

#_____________________________________________________________
# 1.1. C5.0 OOB iPhone

DecTreeModelOOBiPhone <- train(iphonesentiment ~ .,
                               data =trainingiPhone, 
                               method = "C5.0",
                               trControl= ctrl)

# Test c50 OOB iPhone Accuracy : 0.7195  , Kappa : 0.4283  

DecTreePredOOBiPhone <- predict(DecTreeModelOOBiPhone, newdata = testingiPhone)

print(cmDecTreeOOBiPhone <- confusionMatrix(DecTreePredOOBiPhone, testingiPhone$iphonesentiment))

# saveRDS(DecTreeModelOOBiPhone,"DecTreeModelOOBiPhone.RDS")

# 1.2. C5.0 OOB Samsung Accuracy : 0.6544 ,  Kappa : 0.2164

DecTreeModelOOBSamsung <- train(galaxysentiment ~ .,
                               data =trainingSamsung, 
                               method = "C5.0",
                               trControl= ctrl)

# Test c50 OOB Samsung  Accuracy : 0.6544 ,  Kappa : 0.2164

DecTreePredOOBSamsung <- predict(DecTreeModelOOBSamsung, newdata = testingSamsung)

print(cmDecTreeOOBSamsung <- confusionMatrix(DecTreePredOOBSamsung, testingSamsung$galaxysentiment))

# saveRDS(DecTreeModelOOBSamsung,"DecTreeModelOOBSamsung.RDS")

#_________________________________________________

# 2.1. Random Forest OOB iPhone

rfModelOOBiPhone <- train(iphonesentiment ~ .,
                               data =trainingiPhone, 
                               method = "rf",
                               trControl= ctrl)

# Test Random Forest OOB iPhone  Accuracy : 0.7203  Kappa : 0.4259  

rfPredOOBiPhone <- predict(rfModelOOBiPhone, newdata = testingiPhone)

print(cmRfOOBiPhone <- confusionMatrix(rfPredOOBiPhone, testingiPhone$iphonesentiment)) 

# saveRDS(rfModelOOBiPhone,"rfModelOOBiPhone.RDS")

# 1.2. Random Forest OOB Samsung 

rfOOBSamsung <- train(galaxysentiment ~ .,
                                data =trainingSamsung, 
                                method = "rf",
                                trControl= ctrl)

# Test Random Forest OOB Samsung Accuracy : 0.6554    Kappa : 0.2144  

rfPredOOBSamsung <- predict(rfOOBSamsung, newdata = testingSamsung)

print(cmRfOOBSamsung <- confusionMatrix(rfPredOOBSamsung, testingSamsung$galaxysentiment))

# saveRDS(rfOOBSamsung,"rfModelOOBSamsung.RDS")

#SVM (from the e1071 package)

# 3.1. SVM OOB iPhone

svmModelOOBiPhone <- train(iphonesentiment ~ .,
                          data =trainingiPhone, 
                          method = "svmRadial",
                          trControl= ctrl)

# Test SVM OOB iPhone 
# Accuracy : 0.6077 ,Kappa : 0.0987 for SVM Linear
# Accuracy : 0.7075,  Kappa : 0.3939 for SVM Radial

svmPredOOBiPhone <- predict(svmModelOOBiPhone, newdata = testingiPhone)

print(cmSvmOOBiPhone <- confusionMatrix(svmPredOOBiPhone, testingiPhone$iphonesentiment))

# saveRDS(svmModelOOBiPhone,"svmModelOOBiPhone.RDS")

# 3.2. SVM OOB Samsung 

svmModelOOBSamsung <- train(galaxysentiment ~ ., 
                      data =trainingSamsung, 
                      method = "svmLinear",
                      trControl= ctrl)

# Test SVM OOB Samsung 
# Accuracy : 0.6557, Kappa : 0.2173 for SVM Linear
# Accuracy : 0.6549, Kappa : 0.2163 for SVM Radial

svmPredOOBSamsung <- predict(svmModelOOBSamsung, newdata = testingSamsung)

print(cmSvmOOBSamsung <- confusionMatrix(svmPredOOBSamsung, testingSamsung$galaxysentiment))

# saveRDS(svmModelOOBSamsung,"svmModelOOBSamsung.RDS")

#kknn (from the kknn package)

# 4.1. KKNN OOB iPhone

kknnModelOOBiPhone <- train(iphonesentiment ~ .,
                           data =trainingiPhone, 
                           method = "kknn",
                           trControl= ctrl)

# Test KKNN OOB iPhone  Accuracy : 0.2789, Kappa : 0.1123 

kknnPredOOBiPhone <- predict(kknnModelOOBiPhone, newdata = testingiPhone)

print(cmKknnOOBiPhone <- confusionMatrix(kknnPredOOBiPhone, testingiPhone$iphonesentiment))

# saveRDS(kknnModelOOBiPhone,"kknnModelOOBiPhone.RDS")

# 4.2. KKNN OOB Samsung 

kknnModelOOBSamsung <- train(galaxysentiment ~ .,
                       data =trainingSamsung, 
                       method = "kknn",
                       trControl= ctrl)

# Test KKNN OOB Samsung  Accuracy : 0.1417 , Kappa : 0.0114 

kknnPredOOBSamsung <- predict(kknnModelOOBSamsung, newdata = testingSamsung)

print(cmKknnOOBSamsung <- confusionMatrix(kknnPredOOBSamsung, testingSamsung$galaxysentiment))

# saveRDS(kknnModelOOBSamsung,"kknnModelOOBSamsung.RDS")

stopCluster(cl)

### CONCLUSIONS
# BEST MODEL iPhone : Decision Tree
# BEST MODEL Samsung : Decision Tree
# All the models suffer because of the class imbalnce with the level 5 beeing 
# the one that brings the problems
# In order to better the models we have to work on the class imbalance 