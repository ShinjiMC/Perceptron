# Perceptron

By Braulio Nayap Maldonado Casilla

## Introducción

El **perceptrón** es uno de los modelos más simples y fundamentales dentro del campo del aprendizaje automático y las redes neuronales artificiales. Fue propuesto por **Frank Rosenblatt** en 1958 como un algoritmo para la clasificación binaria inspirado en el funcionamiento de las neuronas biológicas. [1]

El perceptrón es una neurona artificial que toma un conjunto de entradas:

$$
\mathbf{x} = (x_1, x_2, \dots, x_n)
$$

con pesos asociados:

$$
\mathbf{w} = (w_1, w_2, \dots, w_n)
$$

y un término de sesgo (bias) $b$, para ser utilizados por una función de activación y producir una salida binaria $y \in \{0, 1\}$.

El valor neto de entrada se calcula como:

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b
$$

La salida del perceptrón se define como:

$$
y = f(z) =
\begin{cases}
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0
\end{cases}
$$

## Implementación en C++

### Ejecución

### Salida

![Ejecución](.docs/perceptron_AND_1.png)
![Ejecución](.docs/perceptron_AND_2.png)

## Implementación en Python

### Ejecución

### Salida

![Ejecución](.docs/python_perceptron.png)

## Conclusiones

## Author

- **ShinjiMC** - [GitHub Profile](https://github.com/ShinjiMC)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
