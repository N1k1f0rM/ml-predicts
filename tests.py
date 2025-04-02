import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from models.linear.linear_regression import LinReg

# X = np.random.randn(200, 20)
# y = np.random.randn(200)


diabetes = load_diabetes()
X = diabetes["data"][:350]
X_test = diabetes["data"][351:]
y = diabetes["target"][:350]
y_test = diabetes["target"][351:]

lr = LinReg(solver="close")
lr.fit(X, y)
lr_pred = lr.predict(X_test)
print(mean_squared_error(y_test, lr_pred))

lrr = LinearRegression()
lrr.fit(X, y)
lrr_pred = lrr.predict(X_test)
print(mean_squared_error(y_test, lrr_pred))


# print(lrr)
# print(len(lr.loss_history), lr.loss_history[-1])