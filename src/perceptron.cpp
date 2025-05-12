#include "perceptron.hpp"

Perceptron::Perceptron(int n_inputs, float lr_input, std::function<float(float)> act)
{
    if (n_inputs <= 0)
        throw std::invalid_argument("N inputs must be positive: " + std::to_string(n_inputs));
    if (lr <= 0)
        throw std::invalid_argument("Learning rate must be positive: " + std::to_string(lr_input));
    lr = lr_input;
    n = Neuron(n_inputs, act);
}

float Perceptron::predict(const std::vector<float> &inputs)
{
    if (inputs.size() != n.get_weights().size())
        throw std::invalid_argument("Size of inputs /= n inputs");
    return n.forward(inputs);
}

void Perceptron::print_weights()
{
    std::cout << "Pesos: ";
    for (auto x : n.get_weights())
        std::cout << x << " ";
    std::cout << "| Sesgo: " << n.get_sesgo() << std::endl;
}

void Perceptron::train(const std::vector<std::vector<float>> &X, const std::vector<float> &y)
{
    if (X.size() != y.size())
        throw std::invalid_argument("Size of X /= size of y");
    if (X.empty())
        throw std::invalid_argument("X is empty");
    if (X[0].size() != n.get_weights().size())
        throw std::invalid_argument("Size of X[0] /= n inputs");

    int epoch = 0;
    bool converged = false;

    std::cout << "Epoch \t| Pesos \t| Sesgo | Error\n";

    while (!converged)
    {
        converged = true;
        int total_errors = 0;

        for (size_t i = 0; i < X.size(); i++)
        {
            auto prediction = n.forward(X[i]);
            auto err = y[i] - prediction;
            if (err != 0)
            {
                n.update_weights(X[i], err, lr);
                converged = false;
                total_errors++;
            }
        }

        // Imprimir pesos, sesgo y error
        std::cout << epoch << " \t| ";
        for (auto w : n.get_weights())
            std::cout << w << " ";
        std::cout << " \t| " << n.get_sesgo() << " \t| " << total_errors << " |\n";

        epoch++;
    }
}

void Perceptron::save_weights(const std::string &filename)
{
    std::ofstream file(filename);
    for (float x : n.get_weights())
        file << x << " ";
    file << n.get_sesgo() << std::endl;
    file.close();
}

bool Perceptron::load_weights(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return false;
    auto sesgo_backup = n.get_sesgo();
    auto w_backup = n.get_weights();
    for (size_t i = 0; i < w_backup.size(); i++)
        file >> w_backup[i];
    file >> sesgo_backup;
    file.close();
    n.set_weights(w_backup);
    n.set_sesgo(sesgo_backup);
    return true;
}
