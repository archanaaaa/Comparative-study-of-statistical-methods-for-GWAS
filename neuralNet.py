from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

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
X = fhs_data.drop('DIABP', axis=1)
y = fhs_data['DIABP']

# create a matrix with SBP and DBP as targets
targets = np.column_stack((fhs_data['SYSBP'], fhs_data['DIABP']))

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train set into train and validation sets
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale features using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=2)

mse = model.evaluate(X_test_scaled, y_test)
print("Mean squared error: ", mse)



