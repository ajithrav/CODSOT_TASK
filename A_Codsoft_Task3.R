

# Load necessary libraries
library(ggplot2)
library(caret)
library(class)

# Load Iris dataset from Kaggle path
iris_data <- read.csv("/kaggle/input/iris-flower-dataset/IRIS.csv")

# Ensure there are at least two classes in the dataset
if (length(unique(iris_data$species)) < 2) {
  stop("The dataset must have at least two classes for classification.")
}

# Convert the 'species' column to factor
iris_data$species <- factor(iris_data$species)

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(iris_data$species, p = 0.8, list = FALSE)

# Check if the partitioning is valid
if (length(unique(iris_data$species[train_index])) < 2) {
  stop("Not enough classes for classification in the training set.")
}

train_data <- iris_data[train_index, ]
test_data <- iris_data[-train_index, ]

# Train the k-Nearest Neighbors (KNN) model
knn_model <- knn(train = train_data[, 1:4], test = test_data[, 1:4], cl = train_data$species, k = 3)

# Ensure factor levels match
test_data$species <- factor(test_data$species, levels = levels(train_data$species))

# Evaluate the model performance
conf_matrix <- confusionMatrix(knn_model, test_data$species)
print(conf_matrix)

# Prediction Example
new_data <- data.frame(sepal_length = 5.0, sepal_width = 3.0, petal_length = 1.5, petal_width = 0.2)
predicted_class <- knn(train_data[, 1:4], new_data, cl = train_data$species, k = 3)

# Ensure factor levels match for the new_data
new_data$species <- factor(predicted_class, levels = levels(train_data$species))
print(paste("Predicted Class:", predicted_class))