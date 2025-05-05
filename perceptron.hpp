#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>
#include <string>

class Perceptron
{
private:
    std::vector<float> weights;
    float bias;
    float learning_rate;
    std::string mode;
    int activation(float z);
    bool show_weight_update = true;
    bool show_predictions = true;

public:
    Perceptron(int n_inputs, float lr = 0.1);
    void set_mode(const std::string &m);
    void set_show_weight_update(bool val);
    void set_show_predictions(bool val);

    int predict(const std::vector<int> &inputs);
    void train(const std::vector<std::vector<int>> &X, const std::vector<int> &y);
    void save_weights(const std::string &filename);
    bool load_weights(const std::string &filename);
    void print_weights();
};

#endif