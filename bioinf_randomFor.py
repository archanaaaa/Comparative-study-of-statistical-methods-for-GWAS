from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

# Load the FHS data into a Pandas DataFrame
fhs_data = pd.read_csv(r'C:\Projects\GWAS\frmgham2.csv')

# print(fhs_data.shape) ----> (11627, 39)

## PRE-PROCESSING DATA

# Remove individuals with missing data
fhs_data.dropna(axis=0, inplace=True)

# Exclude SNPs with a minor allele frequency less than 5%
maf_threshold = 0.05
minor_allele_frequency = fhs_data.mean(axis=0) / 2
common_snps = minor_allele_frequency[minor_allele_frequency >= maf_threshold].index
fhs_data = fhs_data[common_snps]

# Perform imputation to fill in missing genotypes
imputer = KNNImputer(n_neighbors=5)
fhs_data = pd.DataFrame(imputer.fit_transform(fhs_data), columns=fhs_data.columns)

# print(fhs_data.shape) ---> (2236, 33)

## To identify the most important genetic variants associated with both systolic blood pressure (SBP) and diastolic blood pressure (DBP)

# Split data into features (X) and target variable (y)
X = fhs_data.drop('SYSBP', axis=1)
y = fhs_data['SYSBP']

# create a matrix with SBP and DBP as targets
targets = np.column_stack((fhs_data['SYSBP'], fhs_data['DIABP']))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a range of hyperparameters to test
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [5, 10, 15, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# Create the random forest model
rf = RandomForestRegressor()

# Use grid search to find the optimal hyperparameters
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters and their corresponding performance
print("Best hyperparameters: ", grid_search.best_params_)
print("Best performance: ", grid_search.best_score_)

# Make predictions on the test set using the best estimator
y_pred = grid_search.best_estimator_.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R2 score: ", r2)

# Get the most important features
important_features = X_train.columns[grid_search.best_estimator_.feature_importances_ > 0]

print("no of features selected:", len(important_features))
print("selected features: ", important_features)
