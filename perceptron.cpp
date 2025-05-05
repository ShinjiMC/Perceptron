#include "perceptron.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

Perceptron::Perceptron(int n_inputs, float lr)
{
    weights.resize(n_inputs, 0);
    bias = 0;
    learning_rate = lr;
}

void Perceptron::set_mode(const string &m)
{
    mode = m;
}

void Perceptron::set_show_weight_update(bool val)
{
    show_weight_update = val;
}

void Perceptron::set_show_predictions(bool val)
{
    show_predictions = val;
}

int Perceptron::activation(float z)
{
    return z >= 0 ? 1 : 0;
}

int Perceptron::predict(const vector<int> &inputs)
{
    float z = bias;
    for (size_t i = 0; i < weights.size(); i++)
        z += weights[i] * inputs[i];
    return activation(z);
}

void Perceptron::train(const vector<vector<int>> &X, const vector<int> &y)
{
    bool converged = false;
    int epoch = 0;
    while (!converged)
    {
        cout << "Pesos antes del Epoch " << epoch + 1 << ": ";
        for (float w : weights)
            cout << w << " ";
        cout << "| Bias: " << bias << endl;

        vector<float> old_weights = weights;
        float old_bias = bias;

        for (size_t i = 0; i < X.size(); i++)
        {
            int prediction = predict(X[i]);
            int error = y[i] - prediction;

            if (error != 0 && show_weight_update)
            {
                cout << "Fallo en entrada: (" << X[i][0] << ", " << X[i][1] << ")" << endl;
                for (size_t j = 0; j < weights.size(); j++)
                {
                    float delta = learning_rate * error * X[i][j];
                    cout << "w" << j << " = " << weights[j] << " + " << learning_rate << " * " << error << " * " << X[i][j] << " = ";
                    weights[j] += delta;
                    cout << weights[j] << endl;
                }
                float delta_b = learning_rate * error;
                cout << "bias = " << bias << " + " << learning_rate << " * " << error << " = ";
                bias += delta_b;
                cout << bias << endl;
            }
        }

        cout << "----" << endl;
        if (show_predictions)
        {
            for (size_t i = 0; i < X.size(); i++)
            {
                int prediction = predict(X[i]);
                cout << "Entrada: (" << X[i][0] << ", " << X[i][1] << ") ";
                cout << "Salida esperada: " << y[i] << ", Predicha: " << prediction << endl;
            }
            cout << endl;
        }
        converged = true;
        for (size_t i = 0; i < weights.size(); i++)
            if (abs(weights[i] - old_weights[i]) > 0.001)
                converged = false;
        if (abs(bias - old_bias) > 0.001)
            converged = false;
        epoch++;
    }
}

void Perceptron::save_weights(const string &filename)
{
    ofstream file(filename);
    for (float w : weights)
        file << w << " ";
    file << bias;
    file.close();
}

bool Perceptron::load_weights(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
        return false;
    for (size_t i = 0; i < weights.size(); i++)
        file >> weights[i];
    file >> bias;
    file.close();
    return true;
}

void Perceptron::print_weights()
{
    cout << "Pesos: ";
    for (float w : weights)
        cout << w << " ";
    cout << "| Bias: " << bias << endl;
}
