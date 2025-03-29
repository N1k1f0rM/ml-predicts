# ML-From-Scratch 🧠⚙️

**Чистый Python фреймворк для машинного обучения**  
Платформа с модульной архитектурой, реализующая алгоритмы "с нуля".  
Вдохновлено scikit-learn, но под капотом — только стандартная библиотека Python, Pandas и NumPy.

![Project Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)

---

## 🧻 Что уже реализовано?

### 🔨 **Ядро системы (core/)**
Базовые абстракции для единого API:
- `BaseClassifier`: Интерфейс для всех классификаторов
- `BaseRegressor`: Шаблон для регрессионных моделей
- `BaseOptimizer`: Контракт методов оптимизации
- `BaseMetric`: Абстракция для метрик качества
- `BaseTransformer`: Каркас для преобразователей данных

## 🏎️ Что в процессе

### 📈 **Метрики (metrics/)**
**Классификация:**
- Accuracy (доля верных предсказаний)
- Precision, Recall, F1-Score
- Confusion Matrix (визуализация через ASCII-таблицы)

**Регрессия:**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- RMSLE (Root Mean Squared Log Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE ()
- WAPE (Weightned Absolute Percentage Error)
- R² Score (Determination)

### 🧠 **Модели (models/)**
**Линейные алгоритмы:**
- Линейная регрессия
- Линейная регрессия + L1 + L2 + ElasticNet
- Логистическая регрессия

**Деревья:**
- Дерево решений (CART-алгоритм)
  - Критерии разделения: Энтропия, Джини
  - Ограничение глубины и минимального числа образцов
- Случайный лес (бэггинг над деревьями)

**SVM:**
- Линейный SVM (реализация через SGD)
- Ядровой трюк (полиномиальное/RBF ядро)

**Ансамбли:**
- AdaBoost (адаптивное бустирование)

### ⚙️ **Оптимизаторы (optimizers/)**
- GD (просто градиентный спуск)
- SGD (стохастический градиентный спуск)
- Momentum GD
- Nesterov GD
- AdaGrad
- Adam (адаптивная оценка моментов)
- RMSprop (экспоненциальное затухание)

---

## 🛠️ Примеры кода

**Обучение модели:**
```python
from models.linear import RidgeRegression
from optimizers import SGD

model = RidgeRegression(
    optimizer=SGD()
)
model.fit(X_train, y_train)
```

## 🔍 Детали реализаций

### 🧪 Тестирование
- Юнит-тесты для всех компонентов (pytest)
- Сравнение с эталонными реализациями (sklearn)
- Проверка численной стабильности (edge-cases)

---

## 📌 Что планируется в будущем?
- Препроцессинг (Нормализация, кодирование признаков итд)
- Кластеризация (Kmeans, DBSCAN, HDBSCAN, Aglomerative, Spectral)
- PCA: снижение размерности
- Gradient Boosting
- Blending, Stacking
- Визуализация (t-sne, mds, isomap)

---
