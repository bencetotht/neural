#include <iostream>
#include "../include/Neuron.h"
#include "../include/Matrix.h"

using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Neuron *n = new Neuron(1.5);
    // cout << "Val: " << n->getVal() << endl;
    // cout << "Activated val: " << n->getActivatedVal() << endl;
    // cout << "Derived val: " << n->getDerivedVal() << endl;

    Matrix *m = new Matrix(3, 3, true);
    m->printMatrix();

    cout << "---Transposed---" << endl;

    Matrix *mT = m->transpose();
    mT->printMatrix();

    return 0;
}