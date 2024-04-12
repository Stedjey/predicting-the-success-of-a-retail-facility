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
sample_submission = pd.read_csv('submission_sample.csv')




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




# --------------------------------------------------------------------------------

x = final_train_df.drop(['score'], axis=1)
y = final_train_df.score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# ---------------------------------------------------------pip freeze > requirements.txt-----------------------
def value_of_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('R2 Score: ', r2)


# --------------------------------------------------------------------------------
    

    
# Глобальные переменные для хранения лучших параметров и модели
best_params = None
best_model = None
best_mae = float('inf')

def objective(trial):
    global best_params, best_model, best_mae

    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.2, 3)
    }

    model = CatBoostRegressor(**params, silent=True)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)

    # Если текущее значение MAE лучше предыдущего лучшего, обновляем лучшие параметры и модель
    if mae < best_mae:
        best_params = params
        best_model = model
        best_mae = mae

    return mae

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Возвращаем модель с лучшими параметрами
print("Лучшие параметры:", best_params)
print("Лучшее значение MAE:", best_mae)
print("Лучшая модель:", best_model)
# --------------------------------------------------------------------------------

predictions = best_model.predict(x_test)
value_of_metrics(y_test, predictions)

# --------------------------------------------------------------------------------
# Save the model as a pickle in a file
joblib.dump(best_model, 'catboost_model.pkl')