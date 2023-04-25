import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the FHS data into a Pandas DataFrame
fhs_data = pd.read_csv('./csv/frmgham2.csv')

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

# Split data into features (X) and target variable (y)
X = fhs_data.drop('DIABP', axis=1)
y = fhs_data['DIABP']

# create a matrix with SBP and DBP as targets
targets = np.column_stack((fhs_data['SYSBP'], fhs_data['DIABP']))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Lasso regression model
lasso = Lasso()

# Define the parameter grid to search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-3, 1e-4, 1e-5]
}

# Create a grid search object
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use the best estimator to make predictions on the test set
lasso_best = grid_search.best_estimator_
y_pred = lasso_best.predict(X_test_scaled)

# Evaluate model on test set
score = lasso_best.score(X_test_scaled, y_test)
print(f'R-squared score: {score:.3f}')

# Calculate MSEL
mse = ((y_pred - y_test) ** 2).mean()
print(f'Mean Squared Error: {mse:.3f}')

# The features with non-zero coefficients are the selected features
# get selected features
lasso_mask = lasso_best.coef_ != 0

# get important features and their coefficients
important_features = X_train.columns[lasso_mask]
feature_coefficients = lasso_best.coef_[lasso_mask]

print("no of features selected:", len(important_features))
print("selected features: ", important_features)
