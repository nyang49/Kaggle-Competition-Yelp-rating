train = read.csv("train_Madison.csv")

##Correcting errors in city names
train$city[which (train$city == "De Forest" | train$city == "Deforest")] <- "DeForest"
train$city[which (train$city == "Mc Farland" | train$city == "Mcfarland")] <- "McFarland"
train$city[which (train$city == "Sun Praiie")] = "Sun Prairie"

train$city = as.factor(as.character(train$city))
train$name = as.factor(as.character(train$name))

test = read.csv("test_Madison.csv")

test$city[which (test$city == "De Forest" | test$city == "Deforest")] <- "DeForest"
test$city[which (test$city == "Mc Farland" | test$city == "Mcfarland")] <- "McFarland"
test$city[which (test$city == "Sun Praiie")] = "Sun Prairie"

test$city = as.factor(as.character(test$city))
test$name = as.factor(as.character(test$name))


##Selecting variables with 'quanteda' package - word extraction (by Nan Yang and Yupei Lin)
library(quanteda)
library(caret)

TF = function(rownum){
  tf = rownum/sum(rownum)
  return(tf)
}
IDF = function(docnum){
  idf = log10((length(docnum))/(length(which(docnum > 0))))
  return(idf)
}

TF_IDF = function(tf,idf){
  tfidf = tf*idf
  return(tfidf)
}


# ================================ Train Data =====================================
# Data wrangling for train data (created by Nan Yang and Yupei Lin, edited by Sukyoung Cho and Crystal Liu)

#FIRST:tokenize
train$text = as.character(train$text)
train_token = tokens(train$text, what = "word", remove_numbers = TRUE, remove_punct = TRUE,remove_symbols = TRUE, remove_hyphens = TRUE)
train_token = tokens_tolower(train_token)
train_token = tokens_select(train_token, stopwords(), selection = "remove")
train_token = tokens_wordstem(train_token, language = "english")
train_token = tokens_ngrams(train_token, n = 1:2) #phrase extraction

# choose top most frequent 4000 words
train_token_dfm = dfm(train_token, tolower = FALSE)
counts = topfeatures(train_token_dfm, n = 4000, decreasing = T, scheme = c("count","docfreq"), groups = NULL)
top4000 = names(counts)
train_token_dfm = dfm_select(train_token_dfm, pattern = top4000)
train_token_matrix = as.matrix(train_token_dfm)
dim(train_token_matrix)
train_token_df = as.data.frame(train_token_matrix)
dim(train_token_df)

# use TF-IDF
train_token_tf = apply(train_token_matrix, 1, TF)
train_token_idf = apply(train_token_matrix, 2, IDF)
train_token_tfidf = apply(train_token_tf, 2, TF_IDF, idf = train_token_idf)
train_token_df = as.data.frame(t(train_token_tfidf))
dim(train_token_df)

# Combine with stars and some original dataset variables
train_token_df$stars = train$star
train_token_df$city = train$city
train_token_df$lognword = log(train$nword)

##Building a MLR Model

yelp.lm = lm(stars ~ ., data = train_token_df)
summary(yelp.lm)


## Prediction Submission
# ===================================== Test Data ======================================
# do the same thing for test for submission:

# Tokenize
test$text = as.character(test$text)
test_token = tokens(test$text, what = "word", remove_numbers = TRUE, remove_punct = TRUE,remove_symbols = TRUE, remove_hyphens = TRUE)
test_token = tokens_tolower(test_token)
test_token = tokens_select(test_token, stopwords(), selection = "remove")
test_token = tokens_wordstem(test_token, language = "english")
test_token = tokens_ngrams(test_token, n = 1:2)

#choose top 4000 words
test_token_dfm = dfm(test_token, tolower = FALSE)
test_token_dfm = dfm_select(test_token_dfm, pattern = top4000)
test_token_matrix = as.matrix(test_token_dfm)
dim(test_token_matrix)
test_token_df = data.frame(test_token_matrix)
dim(test_token_df)

# Use TF-IDF

test_token_tf = apply(test_token_matrix, 1, TF)
test_token_idf = apply(test_token_matrix, 2, IDF)
test_token_tfidf = apply(test_token_tf, 2, TF_IDF, idf = test_token_idf)
test_token_df = as.data.frame(t(test_token_tfidf))

test_token_df$city = test$city
test_token_df$lognword = log(test$nword)


##Writing .csv file

result = data.frame(Id=test$Id, Expected=predict(yelp.lm, newdata = test_token_df))
#The rating is only between [1:5]
result$Expected[result$Expected > 5] = 5
result$Expected[result$Expected < 1] = 1
write.csv(result,"prediction_tfidf_4000.csv", row.names = FALSE)
