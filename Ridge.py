from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV


# Load the FHS data into a Pandas DataFrame
fhs_data = pd.read_csv('./csv/frmgham2.csv')

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
X = fhs_data.drop('DIABP', axis=1)
y = fhs_data['DIABP']

# create a matrix with SBP and DBP as targets
targets = np.column_stack((fhs_data['SYSBP'], fhs_data['DIABP']))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

# Scale features using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a range of alpha values to test
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'max_iter': [1000, 5000, 10000],
              'tol': [1e-3, 1e-4, 1e-5]}

# Create the Ridge model
ridge = Ridge()

# Use grid search to find the optimal hyperparameters
grid_search = GridSearchCV(ridge, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters and their corresponding performance
print("Best hyperparameters: ", grid_search.best_params_)
print("Best performance: ", grid_search.best_score_)

# Make predictions on the test set
y_pred = grid_search.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R2 score: ", r2)

# Get the most important features
threshold = 0.01

# access the coefficients after fitting the Ridge model using the coef_ attribute
important_features = X_train.columns[abs(grid_search.best_estimator_.coef_) > threshold]

print("Number of important features: ", len(important_features))
print("Important features: ", important_features)
