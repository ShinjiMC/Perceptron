#pragma once
#include "neuron.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class Perceptron
{
private:
    Neuron n;
    float lr;

public:
    Perceptron(int n_inputs, float lr = 0.1, std::function<float(float)> act = nullptr);
    void print_weights();
    void train(const std::vector<std::vector<float>> &X, const std::vector<float> &y);
    float predict(const std::vector<float> &inputs);
    void save_weights(const std::string &filename);
    bool load_weights(const std::string &filename);
};