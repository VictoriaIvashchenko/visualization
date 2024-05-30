import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Завантаження даних
df = pd.read_csv('logdata.csv')

# Перетворення формату дати та часу
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Видалення приставки "ms"
df['TimeTaken'] = df['TimeTaken'].str.replace('ms', '').astype(float)

# Заміна значень у стовпчику LogLevel
log_level_mapping = {
    'INFO': 1,
    'DEBUG': 2,
    'WARNING': 3,
    'ERROR': 4,
    'FATAL': 5
}

df['LogLevel'] = df['LogLevel'].replace(log_level_mapping)

# Графічне зображення залежностей між рівнями логів (LogLevel) та сервісами (Service)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Service', hue='LogLevel')
plt.title('Залежність між рівнями логів та сервісами')
plt.xlabel('Сервіс')
plt.ylabel('Кількість')
plt.legend(title='Рівень Логів')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_grouped = df.groupby('LogLevel')['TimeTaken'].mean().reset_index()
print(df_grouped)

# Додамо одиничний стовпчик для константи в регресійній моделі
df_grouped['intercept'] = 1

# Виконаємо регресію
model = sm.OLS(df_grouped['TimeTaken'], df_grouped[['intercept', 'LogLevel']])
results = model.fit()
print(results.summary())

# Побудова графіку регресії
plt.figure(figsize=(10, 6))
sns.regplot(x='LogLevel', y='TimeTaken', data=df_grouped, ci=None)
plt.title('Парна лінійна регресія між TimeTaken та LogLevel')
plt.xlabel('Рівень Логів')
plt.ylabel('Середній час виконання (TimeTaken)')
plt.tight_layout()
plt.show()

# Прогноз середнього значення з довірчим інтервалом
predictions = results.get_prediction(df_grouped[['intercept', 'LogLevel']])
summary_frame = predictions.summary_frame(alpha=0.05)
df_grouped['mean'] = summary_frame['mean']
df_grouped['mean_ci_lower'] = summary_frame['mean_ci_lower']
df_grouped['mean_ci_upper'] = summary_frame['mean_ci_upper']

# Побудова графіку регресії з довірчим інтервалом
plt.figure(figsize=(10, 6))
sns.regplot(x='LogLevel', y='TimeTaken', data=df_grouped, ci=None, line_kws={"color":"red"})
plt.fill_between(df_grouped['LogLevel'], df_grouped['mean_ci_lower'], df_grouped['mean_ci_upper'], color='gray', alpha=0.2, label='95% Довірчий Інтервал')
plt.title('Парна лінійна регресія між TimeTaken та LogLevel')
plt.xlabel('Рівень Логів')
plt.ylabel('Середній час виконання (TimeTaken)')
plt.legend()
plt.tight_layout()
plt.show()

# Коефіцієнт детермінації (R²)
r_squared = results.rsquared
print(f"Коефіцієнт детермінації (R²): {r_squared}")

# Коефіцієнт кореляції (r)
correlation_matrix = df[['LogLevel', 'TimeTaken']].corr()
correlation = correlation_matrix.loc['LogLevel', 'TimeTaken']
print(f"Коефіцієнт кореляції (r): {correlation}")

# Перевірка статистичної значущості коефіцієнтів
# Для цього ми використовуємо t-тест для коефіцієнтів регресії
t_values = results.tvalues
p_values = results.pvalues
print(f"t-значення коефіцієнтів: \n{t_values}")
print(f"p-значення коефіцієнтів: \n{p_values}")

# Перевірка статистичної значущості моделі (F-тест)
f_p_value = results.f_pvalue
print(f"p-значення для F-тесту: {f_p_value}")

# Перевірка статистичної значущості коефіцієнта кореляції
r, p_value_corr = stats.pearsonr(df['LogLevel'], df['TimeTaken'])
print(f"p-значення коефіцієнта кореляції: {p_value_corr}")

# Інтерпретація результатів
if f_p_value < 0.05:
    print("Регресійна модель є статистично значущою (значення F-тесту).")
else:
    print("Регресійна модель не є статистично значущою (значення F-тесту).")

if p_value_corr < 0.05:
    print("Коефіцієнт кореляції є статистично значущим.")
else:
    print("Коефіцієнт кореляції не є статистично значущим.")
