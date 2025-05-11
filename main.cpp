#include <iostream>
#include "perceptron.hpp"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Uso: ./main x1 x2 AND|OR [archivo.txt]" << std::endl;
        return 1;
    }

    int x1 = atoi(argv[1]);
    int x2 = atoi(argv[2]);
    std::string mode = argv[3];
    std::string filename = (argc == 5) ? argv[4] : mode + "_log.txt";
    std::vector<std::vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<float> y = (mode == "AND") ? std::vector<float>{0, 0, 0, 1} : std::vector<float>{0, 1, 1, 1};
    Perceptron p(2, 0.1);
    bool trained = p.load_weights(filename);
    if (!trained)
    {
        std::cout << "Not found file, training..." << std::endl;
        p.train(X, y);
        std::cout << "Training finished, saving data..." << std::endl;
        p.save_weights(filename);
        std::cout << "Data saved in " << filename << std::endl;
        p.print_weights();
        for (auto &input : X)
        {
            std::cout << "(" << input[0] << ", " << input[1] << ") => "
                      << static_cast<int>(p.predict(input)) << std::endl;
        }
    }
    else
    {
        std::cout << "File found, loaded data..." << std::endl;
        p.print_weights();
    }

    float output = p.predict({static_cast<float>(x1), static_cast<float>(x2)});
    std::cout << "Input: (" << x1 << ", " << x2 << ") => Output: " << output << std::endl;
    return 0;
}
