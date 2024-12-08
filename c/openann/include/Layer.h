//
// Created by Bence Tóth on 03/12/2024.
//

#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

#include "Neuron.h"
using namespace std;
class Layer {
public:
    Layer(int size);
    void setVal(int index, double val);
private:
    int size;
    vector<Neuron *> neurons;
};
#endif //LAYER_H
