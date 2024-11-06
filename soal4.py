import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset = pd.read_csv('dataset.csv')


X = dataset.drop(columns=['price_range'])

# Mengisi missing values dengan SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standarisasi data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# model K-Means
kmeans = KMeans(n_clusters=4, random_state=42) 
kmeans.fit(X_scaled)

# Evaluasi model clustering dengan silhouette score
labels = kmeans.labels_
sil_score = silhouette_score(X_scaled, labels)

print("Silhouette Score:", sil_score)
