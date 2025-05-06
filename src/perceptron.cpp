#include "perceptron.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

int Perceptron::activation(float z)
{
    return z >= 0 ? 1 : 0;
}

void Perceptron::print_epoch_header(int epoch)
{
    cout << "Pesos antes del Epoch " << epoch + 1 << ": ";
    print_weights();
}

void Perceptron::update_weights(const vector<int> &input, int expected)
{
    int prediction = predict(input);
    int error = expected - prediction;
    if (error != 0 && show_weight_update)
    {
        cout << "Fallo en entrada: (" << input[0] << ", " << input[1] << ")" << endl;
        for (size_t j = 0; j < weights.size(); j++)
        {
            float delta = learning_rate * error * input[j];
            cout << "w" << j << " = " << weights[j] << " + " << learning_rate
                 << " * " << error << " * " << input[j] << " = ";
            weights[j] += delta;
            cout << weights[j] << endl;
        }
        float delta_b = learning_rate * error;
        cout << "bias = " << bias << " + " << learning_rate << " * " << error << " = ";
        bias += delta_b;
        cout << bias << endl;
    }
}

bool Perceptron::has_converged(const vector<float> &old_weights, float old_bias)
{
    for (size_t i = 0; i < weights.size(); i++)
        if (abs(weights[i] - old_weights[i]) > 0.001)
            return false;
    return abs(bias - old_bias) <= 0.001;
}

void Perceptron::print_predictions(const vector<vector<int>> &X, const vector<int> &y)
{
    for (size_t i = 0; i < X.size(); i++)
    {
        int prediction = predict(X[i]);
        cout << "Entrada: (" << X[i][0] << ", " << X[i][1] << ") ";
        cout << "Salida esperada: " << y[i] << ", Predicha: " << prediction << endl;
    }
    cout << endl;
}

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
        print_epoch_header(epoch);
        vector<float> old_weights = weights;
        float old_bias = bias;
        for (size_t i = 0; i < X.size(); i++)
            update_weights(X[i], y[i]);
        cout << "----" << endl;
        if (show_predictions)
            print_predictions(X, y);
        converged = has_converged(old_weights, old_bias);
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
