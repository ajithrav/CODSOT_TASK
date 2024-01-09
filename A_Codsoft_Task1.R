# Install and load ggplot2 package
# install.packages("ggplot2")
library(ggplot2)

# Read the Titanic dataset
titanic_data <- read.csv("/kaggle/input/test-file/tested.csv")  # Replace with the actual path

# Remove duplicate rows
dup_set <- unique(titanic_data)

# Check if 'Survived' column is present
if ("Survived" %in% colnames(dup_set)) {
  
  # Create a bar chart for the count of Survived and Unsurvived persons
  ggplot(dup_set, aes(x = factor(Survived))) +
    geom_bar(stat = "count", fill = "steelblue") +
    labs(title = "Count of Survived and Unsurvived Persons",
         x = "Survived",
         y = "Count")
  
  # Display the percentage of Survived and Unsurvived persons
  total_passengers <- nrow(dup_set)
  percentage_data <- data.frame(
    Category = c("Survived", "Unsurvived"),
    Count = c(sum(dup_set$Survived == 1), sum(dup_set$Survived == 0)),
    Percentage = c(sum(dup_set$Survived == 1) / total_passengers * 100,
                   sum(dup_set$Survived == 0) / total_passengers * 100)
  )
  
  # Create a bar chart for the percentage of Survived and Unsurvived persons with labels
  ggplot(percentage_data, aes(x = Category, y = Percentage, fill = Category)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_text(aes(label = paste0(round(Percentage, 1), "%")),
              position = position_dodge(width = 0.9),
              vjust = -0.5) +
    labs(title = "Percentage of Survived and Unsurvived Persons",
         x = "Category",
         y = "Percentage")
  
} else {
  print("No 'Survived' column found in the dataset.")
}


