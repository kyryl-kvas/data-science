# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# =========================
# 0. Завантаження даних та відбір колонок
# =========================
# Завантажте дані з файлу "train.csv" з датасету House Prices – Advanced Regression Techniques
df = pd.read_csv('train.csv')

# Відбір лише необхідних колонок (11 незалежних + цільова змінна)
cols_to_use = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
               'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'GarageCars', 'GarageArea']
df = df[cols_to_use]

# =========================
# 1. Попередній аналіз даних
# =========================
print("Перші рядки даних:")
print(df.head())

print("\nІнформація про дані:")
print(df.info())

print("\nСтатистичний опис числових змінних:")
print(df.describe())

# Побудова попарних залежностей для числових змінних
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# sns.pairplot(df[numeric_cols])
# plt.show()

# =========================
# 2. Обробка пропущених значень
# =========================
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# =========================
# 3. Масштабування даних
# =========================
skewness = df[numeric_cols].apply(lambda x: skew(x.dropna()))
normal_feats = skewness[abs(skewness) < 0.5].index.tolist()
non_normal_feats = skewness[abs(skewness) >= 0.5].index.tolist()

print("\nЗмінні з нормальним розподілом (буде StandardScaler):", normal_feats)
print("Інші змінні (буде MinMaxScaler):", non_normal_feats)

print("\nОпис числових даних до масштабування:")
print(df[numeric_cols].describe())

df_scaled = df.copy()
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()
df_scaled[normal_feats] = scaler_std.fit_transform(df[normal_feats])
df_scaled[non_normal_feats] = scaler_minmax.fit_transform(df[non_normal_feats])

print("\nОпис числових даних після масштабування:")
print(df_scaled[numeric_cols].describe())

# =========================
# 4. Кодування категоріальних змінних
# =========================
# У цьому прикладі категоріальних змінних немає, тому просто копіюємо дані.
df_encoded = df_scaled.copy()
print("\nТипи даних після кодування (якщо були категоріальні):")
print(df_encoded.dtypes)

# =========================
# 5. Кореляційний аналіз
# =========================
corr_matrix = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Матриця кореляції")
plt.show()

target = 'SalePrice'
corr_target = corr_matrix[target].abs().sort_values(ascending=False)
print("\nКореляція змінних з 'SalePrice':")
print(corr_target)

cols_to_drop = set()
corr_thresh = 0.8
cols = corr_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
         col_i = cols[i]
         col_j = cols[j]
         if col_i == target or col_j == target:
             continue
         if abs(corr_matrix.loc[col_i, col_j]) > corr_thresh:
             if corr_target[col_i] >= corr_target[col_j]:
                 cols_to_drop.add(col_j)
             else:
                 cols_to_drop.add(col_i)

print("\nСтовпці, які буде видалено через високу кореляцію:", cols_to_drop)
df_final = df_encoded.drop(columns=list(cols_to_drop))
print("Розмір даних після видалення:", df_final.shape)

# =========================
# 6. Автоматизація препроцесингу та Feature Engineering
# =========================
# Оновлений кастомний трансформер з feature engineering, який створює нові колонки:
# - TotalFinishedArea = TotalBsmtSF + GrLivArea
# - GarageEfficiency = GarageArea / GarageCars (якщо GarageCars > 0, інакше 0)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        # Створення нової колонки TotalFinishedArea
        if 'TotalBsmtSF' in X_transformed.columns and 'GrLivArea' in X_transformed.columns:
            X_transformed['TotalFinishedArea'] = X_transformed['TotalBsmtSF'] + X_transformed['GrLivArea']
        # Створення нової колонки GarageEfficiency
        if 'GarageArea' in X_transformed.columns and 'GarageCars' in X_transformed.columns:
            X_transformed['GarageEfficiency'] = X_transformed.apply(
                lambda row: row['GarageArea'] / row['GarageCars'] if row['GarageCars'] > 0 else 0,
                axis=1)
        return X_transformed

def preprocess_data(df):
    """
    Функція для автоматичного препроцесингу даних:
      - Відбір лише обраних колонок
      - Заповнення пропущених значень
      - Feature engineering: створення нових колонок
      - Масштабування числових даних (StandardScaler для нормально розподілених, MinMaxScaler для інших)
    """
    df_processed = df.copy()
    
    cols_to_use = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                   'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
                   'GarageCars', 'GarageArea']
    df_processed = df_processed[cols_to_use]
    
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    # Виконуємо feature engineering: додаємо нові колонки
    fe = FeatureEngineer()
    df_processed = fe.transform(df_processed)
    
    # Масштабування числових даних (окрім цільової 'SalePrice')
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    target = 'SalePrice'
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    skewness = df_processed[numeric_cols].apply(lambda x: skew(x.dropna()))
    normal_feats = skewness[abs(skewness) < 0.5].index.tolist()
    non_normal_feats = skewness[abs(skewness) >= 0.5].index.tolist()
    
    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()
    df_processed[normal_feats] = scaler_std.fit_transform(df_processed[normal_feats])
    df_processed[non_normal_feats] = scaler_minmax.fit_transform(df_processed[non_normal_feats])
    
    return df_processed

# Демонстрація роботи функції preprocess_data з feature engineering
df_preprocessed = preprocess_data(pd.read_csv('train.csv'))
print("\nПопередній перегляд оброблених даних з feature engineering:")
print(df_preprocessed.head())

# =========================
# Побудова Pipeline з feature engineering
# =========================
# Оновлюємо списки колонок для pipeline з врахуванням нових ознак
numeric_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                    'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'GarageCars', 'GarageArea', 'TotalFinishedArea', 'GarageEfficiency']
categorical_features = []  # У цьому прикладі категоріальних колонок немає

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

full_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

df_raw = pd.read_csv('train.csv')
df_raw = df_raw[cols_to_use]  # використання початкового набору колонок
df_pipeline_processed = full_pipeline.fit_transform(df_raw)
print("\nФорма даних після обробки через pipeline з feature engineering:", df_pipeline_processed.shape)
