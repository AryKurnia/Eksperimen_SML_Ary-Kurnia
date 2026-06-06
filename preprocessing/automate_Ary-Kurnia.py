# Import Library yang Dibutuhkan
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Load Dataset
train = pd.read_csv('mobile-price-classification_raw/train.csv')
test = pd.read_csv('mobile-price-classification_raw/test.csv')

# TRAIN
# Handle missing values
if train.isnull().sum().sum() > 0:
    df_train = train.dropna()
else:
    df_train = train.copy()

# Hapus duplikat
df_train = df_train.drop_duplicates()

# Deteksi dan hapus outlier (IQR)
numeric_features = df_train.select_dtypes(include=['number']).columns
categorical_features = df_train.select_dtypes(include=['object']).columns

Q1 = df_train[numeric_features].quantile(0.25)
Q3 = df_train[numeric_features].quantile(0.75)
IQR = Q3 - Q1

condition = ~((df_train[numeric_features] < (Q1 - 1.5 * IQR)) | (df_train[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_train = pd.concat([
    df_train.loc[condition, numeric_features],
    df_train.loc[condition, categorical_features]
], axis=1)

# Pisahkan fitur dan target SETELAH outlier di-filter
X_train = df_train.drop(columns=['price_range'])
y_train = df_train['price_range']

# Standardisasi — fit di X_train
numeric_features = X_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])


# TEST
# Handle missing values
if test.isnull().sum().sum() > 0:
    df_test = test.dropna()
else:
    df_test = test.copy()

# Hapus duplikat
df_test = df_test.drop_duplicates()

# Test sudah tidak punya target, langsung pakai sebagai fitur
X_test = df_test.copy()

# Standardisasi — transform saja, pakai scaler dari train
X_test[numeric_features] = scaler.transform(X_test[numeric_features])


# SIMPAN
# memastikan folder tujuan
os.makedirs('./mobile-price-classification_preprocessing', exist_ok=True)

X_train.to_csv('mobile-price-classification_preprocessing/X_train_processed.csv', index=False)
y_train.to_csv('mobile-price-classification_preprocessing/y_train_processed.csv', index=False)
X_test.to_csv('mobile-price-classification_preprocessing/X_test_processed.csv', index=False)