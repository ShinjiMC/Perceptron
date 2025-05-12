cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# Ejecutable de entrenamiento
add_executable(perceptron_train
    main_train.cpp
    src/perceptron.cpp
    src/neuron.cpp
    src/dataset.cpp
)

# Ejecutable de prueba
add_executable(perceptron_test
    main_test.cpp
    src/perceptron.cpp
    src/neuron.cpp
)
