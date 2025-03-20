# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# ======================== Крок 1. Завантаження даних та попередній аналіз ========================
print("==== Крок 1: Завантаження даних та попередній аналіз ====")
df = pd.read_csv('train.csv')
print("Розмір датасету:", df.shape)

# ======================== Крок 2. Вибір залежної змінної та 12 незалежних змінних ========================
print("\n==== Крок 2: Вибір змінних ====")
target = 'SalePrice'
selected_features = [
    "OverallQual",   # загальна якість
    "GrLivArea",     # житлова площа над рівнем землі
    "TotalBsmtSF",   # загальна площа підвалу
    "FullBath",      # кількість повних ванних кімнат
    "YearBuilt",     # рік побудови
    "YearRemodAdd",  # рік реконструкції
    "GarageCars",    # кількість машин у гаражі
    "GarageArea",    # площа гаражу
    "1stFlrSF",      # площа першого поверху
    "TotRmsAbvGrd",  # загальна кількість кімнат над рівнем землі
    "LotArea",       # площа ділянки
    "BsmtFullBath"   # кількість повних ванних кімнат у підвалі
]
print("Обрані незалежні змінні:", selected_features)

X = df[selected_features]
y = df[target]

# ======================== Крок 3. Додавання константи ========================
print("\n==== Крок 3: Додавання константи ====")
X = sm.add_constant(X)
print("Стовпець константи додано. Переглянемо перші рядки:")
print(X.head())

# ======================== Крок 4. Розбиття вибірки на навчальну та тестову ========================
print("\n==== Крок 4: Розбиття вибірки ====")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Навчальна вибірка:", X_train.shape)
print("Тестова вибірка:", X_test.shape)

# ======================== Крок 5. Побудова базової регресійної моделі (Модель 1) ========================
print("\n==== Крок 5: Побудова базової моделі (Модель 1) ====")
model1 = sm.OLS(y_train, X_train).fit()
print(model1.summary())

# Виведення рівняння регресії
coefficients = model1.params
eq_terms = ["{:.4f}*{}".format(coeff, var) for var, coeff in coefficients.items() if var != 'const']
equation = "SalePrice = {:.4f} + ".format(coefficients['const']) + " + ".join(eq_terms)
print("\nРівняння регресії (Модель 1):")
print(equation)

# Коефіцієнт множинної кореляції та F-статистика
R = np.sqrt(model1.rsquared)
print("\nКоефіцієнт множинної кореляції (R):", R)
print("F-статистика:", model1.fvalue)

# ======================== Крок 6. Перевірка значущості коефіцієнтів ========================
print("\n==== Крок 6: Перевірка значущості коефіцієнтів (Модель 1) ====")
print(model1.summary2().tables[1])

# ======================== Крок 7. Прогноз та інтервали довіри для тестової вибірки (Модель 1) ========================
print("\n==== Крок 7: Прогноз та довірчі інтервали для тестової вибірки (Модель 1) ====")
pred_test = model1.get_prediction(X_test)
pred_test_df = pred_test.summary_frame(alpha=0.05)
n_test = X_test.shape[0]
in_ci_test = ((y_test >= pred_test_df['obs_ci_lower']) & (y_test <= pred_test_df['obs_ci_upper'])).sum()
out_ci_test = n_test - in_ci_test
print("Кількість спостережень (тест):", n_test)
print("Попадання у довірчий інтервал:", in_ci_test, "(", in_ci_test/n_test, ")")
print("Непопадання:", out_ci_test, "(", out_ci_test/n_test, ")")

# ======================== Крок 8. Прогноз та інтервали довіри для навчальної вибірки (Модель 1) ========================
print("\n==== Крок 8: Прогноз та довірчі інтервали для навчальної вибірки (Модель 1) ====")
pred_train = model1.get_prediction(X_train)
pred_train_df = pred_train.summary_frame(alpha=0.05)
n_train = X_train.shape[0]
in_ci_train = ((y_train >= pred_train_df['obs_ci_lower']) & (y_train <= pred_train_df['obs_ci_upper'])).sum()
out_ci_train = n_train - in_ci_train
print("Кількість спостережень (навч):", n_train)
print("Попадання у довірчий інтервал:", in_ci_train, "(", in_ci_train/n_train, ")")
print("Непопадання:", out_ci_train, "(", out_ci_train/n_train, ")")

# ======================== Крок 9. Аналіз мультиколінеарності ========================
print("\n==== Крок 9: Аналіз мультиколінеарності ====")
corr_matrix = X_train.drop('const', axis=1).corr()
print("Кореляційна матриця незалежних змінних:")
print(corr_matrix)

# Визначення пар змінних із високою кореляцією (|кореляція| > 0.8)
threshold = 0.8
high_corr_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            high_corr_pairs.append((cols[i], cols[j], corr_value))
print("\nПари змінних з високою кореляцією (|кореляція| > {}):".format(threshold))
for pair in high_corr_pairs:
    print(pair)

# ======================== Крок 10. Побудова оптимізованої регресійної моделі (Модель 2) ========================
print("\n==== Крок 10: Побудова оптимізованої моделі (Модель 2) ====")
# Для спрощення та усунення мультиколінеарності обираємо підмножину змінних:
selected_features_model2 = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt"]
X_model2 = df[selected_features_model2]
X_model2 = sm.add_constant(X_model2)
# Використовуємо ті самі індекси для розбиття
X2_train = X_model2.loc[X_train.index]
X2_test = X_model2.loc[X_test.index]

model2 = sm.OLS(y_train, X2_train).fit()
print(model2.summary())

# ======================== Крок 11. Перевірка значущості коефіцієнтів Моделі 2 ========================
print("\n==== Крок 11: Перевірка значущості коефіцієнтів (Модель 2) ====")
print(model2.summary2().tables[1])

# ======================== Крок 12. Прогноз та інтервали довіри для тестової вибірки (Модель 2) ========================
print("\n==== Крок 12: Прогноз та довірчі інтервали для тестової вибірки (Модель 2) ====")
pred_test2 = model2.get_prediction(X2_test)
pred_test2_df = pred_test2.summary_frame(alpha=0.05)
n_test2 = X2_test.shape[0]
in_ci_test2 = ((y_test >= pred_test2_df['obs_ci_lower']) & (y_test <= pred_test2_df['obs_ci_upper'])).sum()
out_ci_test2 = n_test2 - in_ci_test2
print("Кількість спостережень (тест, модель 2):", n_test2)
print("Попадання у довірчий інтервал:", in_ci_test2, "(", in_ci_test2/n_test2, ")")
print("Непопадання:", out_ci_test2, "(", out_ci_test2/n_test2, ")")

# ======================== Крок 13. Прогноз та інтервали довіри для навчальної вибірки (Модель 2) ========================
print("\n==== Крок 13: Прогноз та довірчі інтервали для навчальної вибірки (Модель 2) ====")
pred_train2 = model2.get_prediction(X2_train)
pred_train2_df = pred_train2.summary_frame(alpha=0.05)
n_train2 = X2_train.shape[0]
in_ci_train2 = ((y_train >= pred_train2_df['obs_ci_lower']) & (y_train <= pred_train2_df['obs_ci_upper'])).sum()
out_ci_train2 = n_train2 - in_ci_train2
print("Кількість спостережень (навч, модель 2):", n_train2)
print("Попадання у довірчий інтервал:", in_ci_train2, "(", in_ci_train2/n_train2, ")")
print("Непопадання:", out_ci_train2, "(", out_ci_train2/n_train2, ")")

y_train_pred1 = model1.predict(X_train)
y_test_pred1 = model1.predict(X_test)

# Розрахунок метрик для Моделі 1
mse_train1 = mean_squared_error(y_train, y_train_pred1)
mse_test1 = mean_squared_error(y_test, y_test_pred1)
rmse_train1 = np.sqrt(mse_train1)
rmse_test1 = np.sqrt(mse_test1)
mae_train1 = mean_absolute_error(y_train, y_train_pred1)
mae_test1 = mean_absolute_error(y_test, y_test_pred1)

print("Модель 1:")
print("Train MSE:", mse_train1)
print("Test MSE:", mse_test1)
print("Train RMSE:", rmse_train1)
print("Test RMSE:", rmse_test1)
print("Train MAE:", mae_train1)
print("Test MAE:", mae_test1)

# Прогнозування для Моделі 2
y_train_pred2 = model2.predict(X2_train)
y_test_pred2 = model2.predict(X2_test)

# Розрахунок метрик для Моделі 2
mse_train2 = mean_squared_error(y_train, y_train_pred2)
mse_test2 = mean_squared_error(y_test, y_test_pred2)
rmse_train2 = np.sqrt(mse_train2)
rmse_test2 = np.sqrt(mse_test2)
mae_train2 = mean_absolute_error(y_train, y_train_pred2)
mae_test2 = mean_absolute_error(y_test, y_test_pred2)

print("\nМодель 2:")
print("Train MSE:", mse_train2)
print("Test MSE:", mse_test2)
print("Train RMSE:", rmse_train2)
print("Test RMSE:", rmse_test2)
print("Train MAE:", mae_train2)
print("Test MAE:", mae_test2)
