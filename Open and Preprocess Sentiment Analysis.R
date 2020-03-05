# SENTIMENT ANALYSIS IPHONE AND SAMSUNG---------------------------------------------------------------------------------------------
# Predict the sentiment towards iPhone and Samsung using Common Crawl data harvested using AWS EMR
# Objective: Build models for sentiment towards iPhone and Samsung
# LIBRARIES ----

if ("pacman" %in% rownames(installed.packages()) == FALSE) {
  install.packages("pacman")
} else{
  library(pacman)
  rm(list = ls(all = TRUE))
  p_unload(pacman::p_loaded(), character.only = TRUE)
  pacman::p_load(caret,ggplot2,dplyr,lubridate, plotly,readr, doParallel, corrplot, e1071, kknn, C50, ROSE, plyr,DMwR)
}

# Get the location of the current script in order to be perfectly transferable
# The variable "loc" contains full paths to any file with the same name as the
# current script.
# The "iloc", gets the current index inside the vector of potential matches.
# The "myloc" gets the full path of the script.
loc   <- grep("Open and Preprocess Sentiment Analysis.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/Open and Preprocess Sentiment Analysis.R$",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")

# We set the working directory to the path of the script.
setwd(substr(myloc,1,42))


# LOADING DATASETS----

iPhone <- read.csv("iphone_smallmatrix_labeled_8d.csv")
samsung <- read.csv("galaxy_smallmatrix_labeled_8d.csv")

# Visualise, inxpect, preprocess data ----

iPhone$iphonesentiment <- factor(iPhone$iphonesentiment, ordered = T)
samsung$galaxysentiment <- factor(samsung$galaxysentiment , ordered = T)
# scale 0: very negative, 1: negative, 2: somewhat negative, 3: somewhat positive, 4: positive, 5: very positive

# Recode factors AND set just to 2 levels positive and negative

iPhone$iphonesentiment <- plyr::revalue(iPhone$iphonesentiment, c("0"="NEG", "1" = "NEG", "2" = "NEG","3" = "POS", "4" = "POS", "5" = "POS"))
samsung$galaxysentiment <- plyr::revalue(samsung$galaxysentiment, c("0"="NEG", "1" = "NEG", "2" = "NEG","3" = "POS", "4" = "POS", "5" = "POS"))

# iPhone case
# Histogram Sentiment
plot_ly(iPhone, x= ~iPhone$iphonesentiment, type='histogram')%>%
  layout(title = "Distribution Iphone Sentiment")

# Samsung case
# Histogram Sentiment
plot_ly(samsung, x= ~samsung$galaxysentiment, type='histogram')%>%
        layout(title = "Distribution Galaxy Sentiment")

# From the histograms above we can see that both datasets have the same problem : class imbalance
# very positive sentiment beeing the majority class with the other  class much less representated

# Primary Feature selection ----
# For each dataset we must delete the columns that have nothing to do with either samsung or iphone

iPhone <- iPhone %>% select(grep("iphone", names(iPhone)), grep("ios", names(iPhone)))
samsung <- samsung %>% select(grep("galaxy", names(samsung)), grep("samsung", names(samsung)),grep("google", names(samsung)))

# MOVE SENTIMENT LAST COLUMN
iPhone <- iPhone %>% select(-iphonesentiment,everything())
samsung <- samsung %>% select(-galaxysentiment,everything())

