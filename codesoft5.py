import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data['Class'].value_counts())  # Check class distribution

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))

# Check for missing values
print(data.isnull().sum())

# Handle missing values (if any)
# For example, drop columns with too many missing values:
data_cleaned = data.dropna(axis=1)

# Feature engineering example: Creating a new feature 'hour' from the 'Time' column
data['hour'] = data['Time'] // 3600  # Convert seconds to hours

# Load sales dataset
data_sales = pd.read_csv('sales_data.csv')

# Display first few rows of the dataset and check its structure
print(data_sales.head())
print(data_sales.info())

# Assuming multiple features in the dataset
selected_features_sales = ['Advertising_Expenditure', 'Target_Audience', 'Platform_Selection']
X_sales = data_sales[selected_features_sales]  # Features
y_sales = data_sales['Sales']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

# Normalize/Scale features
scaler_sales = StandardScaler()
X_train_scaled_sales = scaler_sales.fit_transform(X_train_sales)
X_test_scaled_sales = scaler_sales.transform(X_test_sales)

# Initialize and train the linear regression model
model_sales = RandomForestRegressor(random_state=42)
model_sales.fit(X_train_scaled_sales, y_train_sales)

# Make predictions on the test set
y_pred_sales = model_sales.predict(X_test_scaled_sales)

# Evaluate the linear regression model
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Random Forest Regression Metrics:")
print(f"Mean Squared Error: {mse_sales}")
print(f"R-squared: {r2_sales}")

# Hyperparameter tuning for RandomForestRegressor
param_grid_sales = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15]
}

grid_search_sales = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_sales, cv=3)
grid_search_sales.fit(X_train_scaled_sales, y_train_sales)
best_params_sales = grid_search_sales.best_params_

# Initialize and train the RandomForestRegressor with best parameters
best_rf_model_sales = RandomForestRegressor(**best_params_sales, random_state=42)
best_rf_model_sales.fit(X_train_scaled_sales, y_train_sales)

# Make predictions on the test set using RandomForestRegressor
y_pred_rf_sales = best_rf_model_sales.predict(X_test_scaled_sales)

# Evaluate the RandomForestRegressor model
mse_rf_sales = mean_squared_error(y_test_sales, y_pred_rf_sales)
r2_rf_sales = r2_score(y_test_sales, y_pred_rf_sales)

print(f"Random Forest Regression Metrics:")
print(f"Mean Squared Error: {mse_rf_sales}")
print(f"R-squared: {r2_rf_sales}")

# Predict future sales with new data using the RandomForestRegressor model
new_data_sales = pd.DataFrame({
    'Advertising_Expenditure': [1000, 1500, 2000],
    'Target_Audience': [0.8, 0.6, 0.7],
    'Platform_Selection': [1, 2, 3]
})  # Example new data

new_data_scaled_sales = scaler_sales.transform(new_data_sales)
predicted_sales_rf_sales = best_rf_model_sales.predict(new_data_scaled_sales)
print("Predicted sales for new data using RandomForestRegressor:")
print(predicted_sales_rf_sales)
