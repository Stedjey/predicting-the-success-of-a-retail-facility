import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from catboost import CatBoostRegressor
import optuna

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from joblib import Parallel, delayed
import joblib

# загружаем данные из csv файла
train_df = pd.read_csv('train.csv', index_col=False)
train_df.set_index('id', inplace=True)
test_df = pd.read_csv('test.csv')
features = pd.read_csv('features.csv')




# ----------------------------------------------------------
# Преобразование датасетов

# Проведем кластеризацию на train_df
kmeans = KMeans(n_clusters=10, random_state=42) #количество кластеров 
train_coordinates = train_df[['lat', 'lon']].values
kmeans.fit(train_coordinates)

# Найдем ближайшие кластеры к точкам из test
test_coordinates = test_df[['lat', 'lon']].values
closest_cluster, _ = pairwise_distances_argmin_min(test_coordinates, kmeans.cluster_centers_)

# Создадим DataFrame для хранения новых признаков
new_features = pd.DataFrame(columns=features.columns)

# Для каждой точки из train_df найдем ближайший кластер и добавим его признаки
for i, (lat, lon) in enumerate(train_coordinates):
    cluster_idx = kmeans.predict([[lat, lon]])
    new_features = new_features.append(features.iloc[cluster_idx[0]], ignore_index=True)

# Переименуем вторые lat и lon добавленные из features
new_features.rename(columns={'lat': 'lat_features', 'lon': 'lon_features'}, inplace=True)

# Объединим новые признаки с train_df
final_train_df = pd.concat([train_df, new_features], axis=1)

# Результирующий датасет будет иметь размерность (3084 rows × 368 columns)

# Создадим DataFrame для хранения новых признаков для test_df
new_test_features = pd.DataFrame(columns=features.columns)

# Для каждой точки из test_df найдем ближайший кластер и добавим его признаки
for i, (lat, lon) in enumerate(test_coordinates):
    cluster_idx = closest_cluster[i]
    new_test_features = new_test_features.append(features.iloc[cluster_idx], ignore_index=True)

# Переименуем вторые lat и lon добавленные из features
new_test_features.rename(columns={'lat': 'lat_features', 'lon': 'lon_features'}, inplace=True)

# Объединим новые признаки с test_df
final_test_df = pd.concat([test_df, new_test_features], axis=1)

# Результирующий датасет будет иметь размерность (1029 rows × 368 columns)

# Сохранение final_train_df в формате CSV
final_train_df.to_csv('train_df_with_features.csv', index=False)

# Сохранение final_test_df в формате CSV
final_test_df.to_csv('test_df_with_features.csv', index=False)