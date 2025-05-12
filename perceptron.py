from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

def cargar_data(file):
    with open(file, 'r') as f:
        w = list(map(float, f.read().split()))
    return np.array(w[:-1]), w[-1]

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


w_and, sesgo_and = cargar_data('./build/AND_log.txt')
w_or, sesgo_or = cargar_data('./build/OR_log.txt')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# AND
axs[0].plot([0, 1], [-sesgo_and / w_and[1], (-sesgo_and - w_and[0]) / w_and[1]], label='Neurona C++', color='red')
axs[0].plot([0, 1], [-clf_and.intercept_ / clf_and.coef_[0, 1], (-clf_and.intercept_ - clf_and.coef_[0, 0]) / clf_and.coef_[0, 1]], label='Python', color='blue')
axs[0].set_title('Función AND')
axs[0].legend()
axs[0].grid(True)

# OR
axs[1].plot([0, 1], [-sesgo_or / w_or[1], (-sesgo_or - w_or[0]) / w_or[1]], label='Neurona C++', color='red')
axs[1].plot([0, 1], [-clf_or.intercept_ / clf_or.coef_[0, 1], (-clf_or.intercept_ - clf_or.coef_[0, 0]) / clf_or.coef_[0, 1]], label='Python', color='blue')
axs[1].set_title('Función OR')
axs[1].legend()
axs[1].grid(True)

plt.show()