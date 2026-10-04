
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from catboost import CatBoostRegressor

# Константы путей к сохраненным файлам
MODEL_FILE = 'base_catboost_model.cbm'
SCALER_FILE = 'scaler.pkl'

# Список признаков  ( как в модели)
feature_cols = [
    'Плотность, кг/м3', 'Mодуль упругости, ГПа', 'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', 'Поверхностная плотность, г/м2',
    'Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа',
    'Потребление смолы, г/м2', 'Шаг нашивки', 'Плотность нашивки'
]


# 1. ПРОВЕРКА НАЛИЧИЯ И ЗАГРУЗКА ФАЙЛОВ МОДЕЛИ

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    # Если файлов нет, приложение предупредит и закроется
    root_error = tk.Tk()
    root_error.withdraw()
    messagebox.showerror(
        "Ошибка запуска",
        f"Не найдены обязательные файлы модели или скейлера!\n\n"
        f"Убедитесь, что файлы '{MODEL_FILE}' и '{SCALER_FILE}' "
        f"находятся в одной папке со скриптом приложения."
    )
    exit()

# Загружаем скейлер
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

# Загружаем базовую модель CatBoost
model = CatBoostRegressor()
model.load_model(MODEL_FILE)


# 2. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (GUI)

class BaseCompositeApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Базовая модель расчета параметров ПКМ")
        self.geometry("760x560")
        self.resizable(False, False)

        # Словарь текстовых переменных полей ввода
        self.inputs = {}
        self.create_widgets()

    def create_widgets(self):
        # Главная рамка
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Верхний информационный блок
        title_label = ttk.Label(
            main_frame,
            text="Определение соотношения матрица-наполнитель (Базовая модель)",
            font=("Helvetica", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")

        # ЛЕВАЯ СТОРОНА: Поля для 11 признаков
        input_frame = ttk.LabelFrame(main_frame, text=" Входные характеристики композита ", padding="10")
        input_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 15))

        for idx, col_name in enumerate(feature_cols):
            label = ttk.Label(input_frame, text=f"{col_name}:", font=("Helvetica", 9))
            label.grid(row=idx, column=0, sticky="w", pady=4, padx=5)

            var = tk.StringVar()
            var.set("0.0") # Значение по умолчанию для полей ввода

            entry = ttk.Entry(input_frame, textvariable=var, width=15, justify="right")
            entry.grid(row=idx, column=1, sticky="e", pady=4, padx=5)

            self.inputs[col_name] = var

        # ПРАВАЯ СТОРОНА: Управление и вывод
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew")

        desc_text = (
            "Данная версия приложения использует модель CatBoost "
            "(без автоматического поиска гиперпараметров).\n\n"
            "Заполните числовые показатели слева и нажмите кнопку ниже."
        )
        desc_label = ttk.Label(right_frame, text=desc_text, wraplength=280, justify="left", font=("Helvetica", 10))
        desc_label.pack(anchor="w", pady=(10, 25))

        # Кнопка выполнения расчета
        self.calc_btn = ttk.Button(right_frame, text="Выполнить расчет", command=self.calculate)
        self.calc_btn.pack(fill=tk.X, ipady=12, pady=(0, 40))

        # Рамка вывода ответа
        result_frame = ttk.LabelFrame(right_frame, text=" Рекомендуемое значение ", padding="15")
        result_frame.pack(fill=tk.X)

        self.result_value = ttk.Label(
            result_frame,
            text="---",
            font=("Helvetica", 22, "bold"),
            foreground="#2e7d32" # Зеленый цвет для индикации успешного расчета базовой модели
        )
        self.result_value.pack(pady=5)

        target_label = ttk.Label(result_frame, text="Соотношение матрица-наполнитель", font=("Helvetica", 9, "italic"))
        target_label.pack()

    def calculate(self):
        """Обработка нажатия кнопки расчета."""
        raw_values = []

        # 1. Извлечение и валидация
        for col_name in feature_cols:
            val_str = self.inputs[col_name].get().replace(',', '.')
            try:
                val_float = float(val_str)
                raw_values.append(val_float)
            except ValueError:
                messagebox.showerror(
                    "Ошибка данных",
                    f"В поле '{col_name}' обнаружено некорректное значение. Допускаются только числа."
                )
                return

        # 2. Масштабирование через загруженный StandardScaler
        input_array = np.array(raw_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # 3. Предсказание базовой моделью CatBoost
        try:
            prediction = model.predict(input_scaled)[0]

            # Физический сдерживатель (пропорция не может быть ниже нуля)
            if prediction < 0:
                prediction = 0.0

            # Отображаем результат
            self.result_value.config(text=f"{prediction:.4f}")

        except Exception as e:
            messagebox.showerror("Критическая ошибка", f"Модель не смогла рассчитать параметр:\n{str(e)}")


# 3. ТОЧКА ВХОДА В GUI

if __name__ == "__main__":
    app = BaseCompositeApp()
    app.mainloop()
