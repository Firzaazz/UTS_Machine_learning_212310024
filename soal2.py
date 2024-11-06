import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('dataset.csv')

X = dataset.drop(columns=['price_range']) 
y = dataset['price_range']

# Statistik deskriptif sebelum pengisian missing values
print("Statistik Deskriptif Sebelum Pengisian Missing Values:")
print(X.describe())


imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


print("\nStatistik Deskriptif Setelah Pengisian Missing Values:")
print(X_imputed.describe())

# Standarisasi data prediktor
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Statistik deskriptif setelah standarisasi
print("\nStatistik Deskriptif Setelah Standarisasi:")
print(X_scaled.describe())

