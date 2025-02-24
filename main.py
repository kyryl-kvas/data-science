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

df = pd.read_csv('train.csv')

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
sns.pairplot(df[numeric_cols])
plt.show()

# Оскільки в обраному наборі колонок немає текстових змінних, вивід категорій не потрібен.
# Якщо є категоріальні колонки, їх можна обробити, як наведено нижче:
# categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# for col in categorical_cols:
#     print(f"Унікальні категорії у стовпці '{col}': {df[col].unique()}")

# =========================
# 2. Обробка пропущених значень
# =========================
# Для числових змінних заповнюємо пропуски середнім значенням.
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# =========================
# 3. Масштабування даних
# =========================
# Аналіз скошеності для визначення нормального розподілу
skewness = df[numeric_cols].apply(lambda x: skew(x.dropna()))
# Вважаємо, що змінна з абсолютним значенням скошеності менше 0.5 є нормальною
normal_feats = skewness[abs(skewness) < 0.5].index.tolist()
non_normal_feats = skewness[abs(skewness) >= 0.5].index.tolist()

print("\nЗмінні з нормальним розподілом (буде StandardScaler):", normal_feats)
print("Інші змінні (буде MinMaxScaler):", non_normal_feats)

print("\nОпис числових даних до масштабування:")
print(df[numeric_cols].describe())

# Масштабування – створимо копію даних для трансформації
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
# У цьому підході у вибраному наборі колонок немає категоріальних змінних.
# Якщо вони з’являться, можна використовувати pd.get_dummies, наприклад:
# df_encoded = pd.get_dummies(df_scaled, columns=categorical_cols)
df_encoded = df_scaled.copy()  # оскільки всі колонки числові

print("\nТипи даних після кодування (якщо були категоріальні):")
print(df_encoded.dtypes)

# =========================
# 5. Кореляційний аналіз
# =========================
# Побудова матриці кореляції
corr_matrix = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Матриця кореляції")
plt.show()

# Визначаємо змінні, найбільш корельовані з цільовою змінною 'SalePrice'
target = 'SalePrice'
corr_target = corr_matrix[target].abs().sort_values(ascending=False)
print("\nКореляція змінних з 'SalePrice':")
print(corr_target)

# Видалення змінних з високою взаємною кореляцією (corr > 0.8)
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
             # Видаляємо ту змінну, яка має меншу кореляцію з цільовою
             if corr_target[col_i] >= corr_target[col_j]:
                 cols_to_drop.add(col_j)
             else:
                 cols_to_drop.add(col_i)

print("\nСтовпці, які буде видалено через високу кореляцію:", cols_to_drop)
df_final = df_encoded.drop(columns=list(cols_to_drop))
print("Розмір даних після видалення:", df_final.shape)

# =========================
# 6. Автоматизація препроцесингу
# =========================
def preprocess_data(df):
    """
    Функція для автоматичного препроцесингу даних:
      - Заповнення пропущених значень
      - Відбір лише обраних колонок
      - Кодування категоріальних змінних (якщо є)
      - Масштабування числових даних (з використанням StandardScaler для нормально розподілених змінних 
        та MinMaxScaler для інших)
    """
    df_processed = df.copy()
    
    # Відбір колонок
    cols_to_use = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                   'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
                   'GarageCars', 'GarageArea']
    df_processed = df_processed[cols_to_use]
    
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Заповнення пропусків для числових змінних
    for col in numeric_cols:
        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    # Якщо з'являться категоріальні змінні, їх можна обробити наступним чином:
    # categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    # for col in categorical_cols:
    #     df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    #     df_processed = pd.get_dummies(df_processed, columns=[col])
    
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

# Демонструємо роботу функції на сирих даних
df_preprocessed = preprocess_data(pd.read_csv('train.csv'))
print("\nПопередній перегляд оброблених даних:")
print(df_preprocessed.head())

# =========================
# Побудова Pipeline для автоматизації препроцесингу
# =========================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        # Приклад feature engineering: створення нового параметру, якщо є відповідні колонки
        if 'YrSold' in X_transformed.columns and 'YearBuilt' in X_transformed.columns:
            X_transformed['HouseAge'] = X_transformed['YrSold'] - X_transformed['YearBuilt']
        return X_transformed

# Визначаємо списки колонок на основі обраного набору
numeric_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                    'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'GarageCars', 'GarageArea']
categorical_features = []  # В даному прикладі немає текстових колонок

# Pipeline для числових змінних: заповнення пропусків і масштабування
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline для категоріальних змінних (залишено для демонстрації)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Об’єднуємо трансформації
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Повний Pipeline
full_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

# Демонстрація роботи pipeline на сирих даних
df_raw = pd.read_csv('train.csv')
# Відбір лише необхідних колонок
df_raw = df_raw[cols_to_use]
df_pipeline_processed = full_pipeline.fit_transform(df_raw)
print("\nФорма даних після обробки через pipeline:", df_pipeline_processed.shape)
