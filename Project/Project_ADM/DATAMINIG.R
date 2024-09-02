# Install and load required packages
install.packages(c("readr", "tidyverse", "tm", "SnowballC", "textTinyR", "wordcloud", "ggplot2", "nnet", "randomForest", "e1071", "caret", "pROC", "PRROC"))

library(readr)
library(tidyverse)
library(tm)
library(SnowballC)
library(textTinyR)
library(wordcloud)
library(ggplot2)
library(nnet)
library(randomForest)
library(e1071)
library(caret)
library(pROC)
library(PRROC)

# Read the dataset
df <- read_csv("twitter_training.csv", col_names = c("id", "country", "Label", "Text"))

# Preprocessing text data
preprocess <- function(text) {
  # Convert to lowercase
  text <- tolower(text)
  # Remove punctuation
  text <- removePunctuation(text)
  # Tokenize
  words <- unlist(strsplit(text, "\\s+"))
  # Remove stopwords
  words <- words[!words %in% stopwords("en")]
  # Stemming
  words <- wordStem(words)
  # Concatenate tokens
  text <- paste(words, collapse = " ")
  return(text)
}

# Apply preprocessing to text column
df$Preprocessed_Text <- sapply(df$Text, preprocess)

# Convert Label column to factor
df$Label <- as.factor(df$Label)

# Calculate the length of each tweet
df$Tweet_Length <- nchar(df$Text)

# Word cloud for positive sentiment tweets
positive_tweets <- df[df$Label == "Positive", "Preprocessed_Text"]
wordcloud(words = unlist(positive_tweets), min.freq = 50, random.order = FALSE,
          colors = brewer.pal(8, "Set2"), main = "Word Cloud for Positive Sentiment Tweets")

# Word cloud for negative sentiment tweets
negative_tweets <- df[df$Label == "Negative", "Preprocessed_Text"]
wordcloud(words = unlist(negative_tweets), min.freq = 50, random.order = FALSE,
          colors = brewer.pal(8, "Set2"), main = "Word Cloud for Negative Sentiment Tweets")

# Word cloud for irrelevant sentiment tweets
irrelevant_tweets <- df[df$Label == "Irrelevant", "Preprocessed_Text"]
wordcloud(words = unlist(irrelevant_tweets), min.freq = 50, random.order = FALSE,
          colors = brewer.pal(8, "Set2"), main = "Word Cloud for Irrelevant Sentiment Tweets")

# Word cloud for neutral sentiment tweets
neutral_tweets <- df[df$Label == "Neutral", "Preprocessed_Text"]
wordcloud(words = unlist(neutral_tweets), min.freq = 50, random.order = FALSE,
          colors = brewer.pal(8, "Set2"), main = "Word Cloud for Neutral Sentiment Tweets")

# Create a bar plot for the distribution of tweets per branch and type
ggplot(data = df, aes(x = country, y = id, fill = Label)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Country", y = "Number of Tweets", title = "Distribution of Tweets by Country and Sentiment") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

# Create a histogram of tweet lengths, faceted by sentiment
ggplot(data = df, aes(x = Tweet_Length)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  labs(x = "Tweet Length", y = "Frequency", title = "Distribution of Tweet Lengths by Sentiment") +
  facet_wrap(~ Label, scales = "free") +
  theme_minimal() +
  theme(legend.position = "none")

# Create a bar plot for the distribution of sentiment labels
ggplot(data = df, aes(x = Label, fill = Label)) +
  geom_bar() +
  labs(x = "Sentiment Label", y = "Count", title = "Distribution of Sentiment Labels") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Set the seed for reproducibility
set.seed(123)

# Sample a subset of the data (e.g., 20% of the original data)
subset_size <- 0.2
sampled_indices <- sample(1:nrow(df), size = floor(subset_size * nrow(df)))
subset_df <- df[sampled_indices, ]

# Split the sampled data into training and testing sets
train_index <- createDataPartition(subset_df$Label, p = 0.8, list = FALSE)
train_data <- subset_df[train_index, ]
test_data <- subset_df[-train_index, ]

# Convert Label column to factor
train_data$Label <- as.factor(train_data$Label)

# Train SVM model
svm_model <- svm(Label ~ Preprocessed_Text, data = train_data, kernel = "linear")

# Train Random Forest model
rf_model <- randomForest(Label ~ Preprocessed_Text, data = train_data, ntree = 100)

# Use the test data that was split from the training dataset
test_data <- train_data[-train_index, ]  # Assuming train_index was previously defined

# Make predictions using SVM
svm_predictions <- predict(svm_model, newdata = test_data)

# Make predictions using Random Forest
rf_predictions <- predict(rf_model, newdata = test_data)

# Remove 30 rows from test_data
test_data_svm <- test_data[-(1:30), ]

# Evaluate SVM model
svm_conf_matrix <- confusionMatrix(data = svm_predictions, reference = test_data_svm$Label)
svm_accuracy <- svm_conf_matrix$overall['Accuracy']
svm_precision <- svm_conf_matrix$byClass['Precision']
svm_recall <- svm_conf_matrix$byClass['Recall']
svm_f1 <- svm_conf_matrix$byClass['F1']

# Evaluate Random Forest model
rf_conf_matrix <- confusionMatrix(data = rf_predictions, reference = test_data$Label)
rf_accuracy <- rf_conf_matrix$overall['Accuracy']
rf_precision <- rf_conf_matrix$byClass['Precision']
rf_recall <- rf_conf_matrix$byClass['Recall']
rf_f1 <- rf_conf_matrix$byClass['F1']

# Confusion matrix
print("SVM Confusion Matrix:")
print(svm_conf_matrix$table)
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix$table)

# ROC curves
binary_outcome <- ifelse(test_data_svm$Label == "Positive", 1, 0)
svm_roc <- roc(binary_outcome, as.numeric(svm_predictions == "Positive"))
plot(svm_roc, col = "blue")
# Convert Label to binary outcome with two levels
binary_outcome <- ifelse(test_data$Label == "Positive", 1, 0)

# Compute ROC curve
rf_roc <- roc(binary_outcome, as.numeric(rf_predictions))

plot(rf_roc, col = "red")

# Precision-recall curves
svm_pr <- pr.curve(scores.class0 = as.numeric(svm_predictions == "Positive"), weights.class0 = ifelse(test_data$Label == "Positive", 1, 0))
rf_pr <- pr.curve(scores.class0 = as.numeric(rf_predictions == "Positive"), weights.class0 = ifelse(test_data$Label == "Positive", 1, 0))

# Plot ROC curves
plot(svm_roc, col = "blue", main = "ROC Curves", legacy.axes = TRUE, print.auc = TRUE)
plot(rf_roc, col = "red", add = TRUE, print.auc = TRUE, print.thres = c(0.5, 0.1), print.thres.pattern = "%.1f")
legend("bottomright", legend = c("SVM", "Random Forest"), col = c("blue", "red"))

# Evaluate SVM model
svm_conf_matrix <- confusionMatrix(data = svm_predictions, reference = test_data_svm$Label)
svm_accuracy <- svm_conf_matrix$overall['Accuracy']
svm_precision <- svm_conf_matrix$byClass[4, 'Pos Pred Value']
svm_recall <- svm_conf_matrix$byClass[4, 'Sensitivity']
svm_f1 <- (2 * svm_precision * svm_recall) / (svm_precision + svm_recall)

# Evaluate Random Forest model
rf_conf_matrix <- confusionMatrix(data = rf_predictions, reference = test_data$Label)
rf_accuracy <- rf_conf_matrix$overall['Accuracy']
rf_precision <- rf_conf_matrix$byClass[4, 'Pos Pred Value']
rf_recall <- rf_conf_matrix$byClass[4, 'Sensitivity']
rf_f1 <- (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)

# Create a data frame to store evaluation metrics
evaluation_metrics <- data.frame(
  Model = c("SVM", "Random Forest"),
  Accuracy = c(svm_accuracy, rf_accuracy),
  Precision = c(svm_precision, rf_precision),
  Recall = c(svm_recall, rf_recall),
  F1_Score = c(svm_f1, rf_f1)
)

# Print the table
print(evaluation_metrics)
