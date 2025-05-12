#include <iostream>
#include "perceptron.hpp"
#include "dataset.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Use: ./main [dataset.txt]" << std::endl;
        return 1;
    }

    std::string dataset_file = argv[1];
    Dataset dataset(dataset_file);
    std::vector<std::vector<float>> X = dataset.get_X();
    std::vector<float> y = dataset.get_y();
    std::string path = dataset_file;
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");
    std::string base = path.substr(last_slash + 1, last_dot - last_slash - 1);
    if (X.empty() || y.empty())
    {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return 1;
    }
    Perceptron p(X[0].size(), 0.1);
    std::cout << "Loaded dataset. Training..." << std::endl;
    p.train(X, y);
    std::cout << "Training finished, saving data..." << std::endl;
    std::string filename = base + "_log.txt";
    p.save_weights(filename);
    std::cout << "Data saved in " << filename << std::endl;
    p.print_weights();
    for (auto &input : X)
        std::cout << "(" << input[0] << ", " << input[1] << ") => "
                  << static_cast<int>(p.predict(input)) << std::endl;
    return 0;
}
