#include <iostream>
#include "../include/Neuron.h"
#include "../include/Matrix.h"
#include "../include/NeuralNetwork.h"

using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Neuron *n = new Neuron(1.5);
    // cout << "Val: " << n->getVal() << endl;
    // cout << "Activated val: " << n->getActivatedVal() << endl;
    // cout << "Derived val: " << n->getDerivedVal() << endl;

    Matrix *m = new Matrix(3, 3, true);
    // m->printMatrix();

    // cout << "---Transposed---" << endl;

    Matrix *mT = m->transpose();
    // mT->printMatrix();

    vector<int> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(3);

    vector<double> inputs;
    inputs.push_back(1.0);
    inputs.push_back(0.0);
    inputs.push_back(1.0);

    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->setInputs(inputs);
    nn->printToConsole();

    return 0;
}