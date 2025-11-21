import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import time

#############################################
filename = '../data/preproc_dataset_knn.csv'
df = pd.read_csv(filename, delimiter=',')
#############################################

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = -np.inf
best_k = 0

for k in range(1, 21): 
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring='r2')
    mean_score = scores.mean()
    print(f"k={k}, mean R2={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nBest num: {best_k} R2: {best_score:.4f}")
