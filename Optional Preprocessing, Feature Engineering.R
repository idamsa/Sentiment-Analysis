source("Open and Preprocess Sentiment Analysis.R")
# Class Imbalance Solution -------------------------------------------

# Check proportions for the 5 classes
# Class 5 highly disproportionate, classes 1,2,3 vey low proportion, 0,4 ok
round(prop.table(table(iPhone$iphonesentiment)), 2)   
# NEG  POS 
# 0.22 0.78 
round(prop.table(table(samsung$galaxysentiment)), 2)
# NEG POS 
# 0.2 0.8  
# Fixed the imbalance with SMOTE IN THE TRAIN CONTROL

# # Feature Selection Cases ----

# 1. Correlation Matrix ----
# Eliminate highly correlated features

# iPhone case
iPhoneCOR <- as.data.frame(sapply(iPhone, as.numeric))
corMatrixiPhone <- cor(iPhoneCOR)
corrplot(corMatrixiPhone, method = "number")
hciPhone <- findCorrelation(corMatrixiPhone , cutoff = 0.9) # extracts the correlations higher than 0.9
hciPhone <- sort(hciPhone)
iPhoneCOR <- iPhoneCOR[,-c(hciPhone)]
iPhoneCOR$iphonesentiment <- factor(iPhoneCOR$iphonesentiment, ordered = T)
iPhoneCOR$iphonesentiment <- plyr::revalue(iPhoneCOR$iphonesentiment, c("1"="NEG", "2" = "POS"))



# Samsung case
samsungCOR <- as.data.frame(sapply(samsung, as.numeric))
corMatrixSamsung <- cor(samsungCOR)
corrplot(corMatrixSamsung, method = "number")
hcSamsung <- findCorrelation(corMatrixSamsung, cutoff = 0.9) # extracts the correlations higher than 0.9
hcSamsung <- sort(hcSamsung)
samsungCOR <- samsungCOR[,-c(hcSamsung)]
samsungCOR$galaxysentiment <- factor(samsungCOR$galaxysentiment , ordered = T)
samsungCOR$galaxysentiment <- plyr::revalue(samsungCOR$galaxysentiment, c("1"="NEG", "2" = "POS"))

# 2. Feature Variance ----
# Eliminate near zero variance features

# iPhone case

nzvMetricsiPhone <- nearZeroVar(iPhone, saveMetrics = TRUE)
nzviPhone <- nearZeroVar(iPhone, saveMetrics = FALSE) 
iphoneNZV <- iPhone[,-nzviPhone]

# Samsung case
# just 1 feature left after eliminating the near 0 var ones
nzvMetricsSamsung <- nearZeroVar(samsung, saveMetrics = TRUE)
nzvSamsung <- nearZeroVar(samsung, saveMetrics = FALSE) 
samsungNZV <- samsung[,-nzvSamsung]


# 3. Recursive Feature Elimination ---- 

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# iPhone case
# Sample the data before using RFE
set.seed(123)
iPhoneSample <- iPhone[sample(1:nrow(iPhone), 1000, replace=FALSE),]
iPhoneSample <- iPhoneSample %>% select(-iphonesentiment,everything()) # movest iphonesnetiment to last column
# Use rfe and omit the response variable (attribute 15 iphonesentiment) 
if (!exists("RFE iPhone")) {
  if (file.exists("RFE iPhone")) {
    rfeResultsiPhone <- readRDS("RFE iPhone")
  }
  else{
rfeResultsiPhone <- rfe(iPhoneSample[,1:14], 
                  iPhoneSample$iphonesentiment, 
                  sizes=(1:14), 
                  rfeControl=ctrl)
  }
}

# Plot results
plot(rfeResultsiPhone, type=c("g", "o"))

# create new data set with rfe recommended features
iPhoneRFE <- iPhone[,predictors(rfeResultsiPhone)]

# add the dependent variable to iphoneRFE
iPhoneRFE$iphonesentiment <- iPhone$iphonesentiment

# review outcome
str(iPhoneRFE)

# save rfe
if (!exists("RFE iPhone")) {
  saveRDS(rfeResultsiPhone, file = "RFE iPhone.rds")
}


# Samsung case
# Sample the data before using RFE
set.seed(123)
samsungSample <- samsung[sample(1:nrow(samsung), 1000, replace=FALSE),]
samsungSample <- samsungSample %>% select(-galaxysentiment,everything()) # movest galaxysentiment to last column
# Use rfe and omit the response variable (attribute 15 galaxysentiment) 
if (!exists("RFE Samsung")) {
  if (file.exists("RFE Samsung")) {
    rfeResultsSamsung <- readRDS("RFE Samsung")
  }
  else{
rfeResultsSamsung <- rfe(samsungSample[,1:14],
                  samsungSample$galaxysentiment, 
                  sizes=(1:14), 
                  rfeControl=ctrl)
  }
}

# Plot results
plot(rfeResultsSamsung, type=c("g", "o"))

# create new data set with rfe recommended features
samsungRFE <- samsung[,predictors(rfeResultsSamsung)]

# add the dependent variable to iphoneRFE
samsungRFE$galaxysentiment <- samsung$galaxysentiment

# review outcome
as.data.frame(unlist(samsungRFE)) # just 1 predictor

# save rfe
if (!exists("RFE Samsung")) {
  saveRDS(rfeResultsSamsung, file = "RFE Samsung.rds")
}

