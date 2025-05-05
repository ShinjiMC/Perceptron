from sklearn.linear_model import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

print("Entrenando con scikit-learn para función AND")
clf_and = Perceptron(max_iter=100, eta0=0.1, fit_intercept=True, warm_start=False)
clf_and.fit(X, y_and)
print("Pesos (AND):", clf_and.coef_, " Bias:", clf_and.intercept_)

print("\nEntrenando con scikit-learn para función OR")
clf_or = Perceptron(max_iter=100, eta0=0.1, fit_intercept=True, warm_start=False)
clf_or.fit(X, y_or)
print("Pesos (OR):", clf_or.coef_, " Bias:", clf_or.intercept_)
