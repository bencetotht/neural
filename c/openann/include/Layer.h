//
// Created by Bence Tóth on 03/12/2024.
//

#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

#include "Matrix.h"
#include "Neuron.h"
using namespace std;
class Layer {
public:
    Layer(int size);
    void setVal(int index, double val);

    Matrix *matrixifyVals();
    Matrix *matrixifyActivatedVals();
    Matrix *matrixifyDerivedVals();
private:
    int size;
    vector<Neuron *> neurons;
};
#endif //LAYER_H
