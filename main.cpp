#include <iostream>
#include "perceptron.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cout << "Uso: ./main x1 x2 AND|OR [archivo.txt]" << endl;
        return 1;
    }

    int x1 = atoi(argv[1]);
    int x2 = atoi(argv[2]);
    string mode = argv[3];
    string filename = (argc == 5) ? argv[4] : mode + "_log.txt";
    vector<vector<int>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<int> y = (mode == "AND") ? vector<int>{0, 0, 0, 1} : vector<int>{0, 1, 1, 1};

    Perceptron p(2);
    p.set_mode(mode);
    p.set_show_weight_update(true);
    p.set_show_predictions(true);

    bool trained = p.load_weights(filename);
    if (!trained)
    {
        cout << "No se encontró archivo, entrenando..." << endl;
        p.train(X, y);
        p.save_weights(filename);
    }

    p.print_weights();

    int output = p.predict({x1, x2});
    cout << "Entrada: (" << x1 << ", " << x2 << ") => Salida: " << output << endl;

    return 0;
}
