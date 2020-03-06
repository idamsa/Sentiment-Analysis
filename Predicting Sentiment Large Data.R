# Applly models on new data large matrix

#1. bring the models and the data

fit_iphone <- readRDS( "DecTreeModelCORiPhone.RDS")
fit_samsung <- readRDS("DecTreeModelOOBSamsung.RDS")

data_large <- read.csv("data_large.csv", stringsAsFactors = F)

# Preprocess for the iphone and samsung
# Chose just the highly corelated features for iphone
# just the ones that contain samsung for the samsung

data_large_iphone <- data_large[,which(names(data_large) %in% names(iPhoneCOR))]
data_large_iphone <- as.data.frame(sapply( data_large_iphone, function(x) strtoi(x, base = 0L))) # chenage to int
data_large_iphone <- data_large_iphone[complete.cases(data_large_iphone), ]
data_large_samsung <-  data_large %>% select(grep("galaxy", names(data_large)), grep("samsung", names(data_large)),grep("google", names(data_large)))
data_large_samsung <- as.data.frame(sapply(data_large_samsung, function(x) strtoi(x, base = 0L))) 
data_large_samsung <- data_large_samsung[complete.cases(data_large_samsung), ]

# Predict the sentiment

iphone_sentiment_prediction <- predict(fit_iphone, data_large_iphone) #Predict the sentiment
samsung_sentiment_prediction <- predict(fit_samsung, data_large_samsung)

# add prediction to df

data_large_iphone$sentiment <- iphone_sentiment_prediction
data_large_samsung$sentiment <- samsung_sentiment_prediction

# plot the results

plot_ly(data_large_iphone, x= data_large_iphone$sentiment, type='histogram')%>%  # iPhone results
  layout(title = "Distribution iPhone Predicted Sentiment")


plot_ly(data_large_samsung, x= data_large_samsung$sentiment, type='histogram')%>%  # samsung results
  layout(title = "Distribution Samsung Predicted Sentiment")
