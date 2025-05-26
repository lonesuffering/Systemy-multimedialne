import pandas as pd

# Загрузка данных из CSV
df = pd.read_csv('AB_NYC_2019.csv')

# Создание сводной таблицы
pivot_table = pd.pivot_table(
    df,
    values='price',  # Агрегируемое числовое поле
    index='neighbourhood_group',  # Группировка по районам
    columns='room_type',  # Разделение по типам жилья
    aggfunc='mean'  # Вычисление среднего значения
)

# Сохранение сводной таблицы в Excel (опционально)
pivot_table.to_excel('pivot_table.xlsx')

print(pivot_table)