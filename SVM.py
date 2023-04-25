import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE

# Load the FHS data into a Pandas DataFrame
fhs_data = pd.read_csv('./csv/frmgham2.csv')

# fhs_data_shape:
n_cols = fhs_data.shape[1]

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
X = fhs_data.drop('SYSBP', axis=1)
y = fhs_data['SYSBP']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define SVM model and RFE feature selection
svm = SVR(kernel='linear')
rfe = RFE(svm)

# Define parameter grid for n_features_to_select
arr = list(range(n_cols//2, n_cols))
param_grid = {'n_features_to_select': arr}

# Define grid search with cross-validation
grid_search = GridSearchCV(rfe, param_grid, cv=5)

# Fit grid search to training data
grid_search.fit(X_train_scaled, y_train)

# Get best model and its selected features
best_model = grid_search.best_estimator_
selected_features = X_train.columns[best_model.support_]

# Evaluate model on test set
score = best_model.score(X_test_scaled, y_test)
print(f'R-squared score: {score:.3f}')

# Predict on test set
y_pred = best_model.predict(X_test_scaled)

# Calculate MSEL
mse = ((y_pred - y_test) ** 2).mean()

print(f'Mean Squared Error: {mse:.3f}')
print("Number of features selected: ",len(selected_features))
print("Selected features:", selected_features)
