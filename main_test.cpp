#include <iostream>
#include "perceptron.hpp"
#include "dataset.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "Use: ./main_test x1 x2 weights.txt" << std::endl;
        return 1;
    }

    float x1 = std::stof(argv[1]);
    float x2 = std::stof(argv[2]);
    std::string filename = argv[3];

    Perceptron p(2, 0.1);
    if (!p.load_weights(filename))
    {
        std::cerr << "Error: Not load weights for training." << std::endl;
        return 1;
    }
    std::cout << "File found, loaded data..." << std::endl;
    p.print_weights();
    std::cout << "Input: (" << x1 << ", " << x2 << ") => Output: " << p.predict({x1, x2}) << std::endl;
    return 0;
}
