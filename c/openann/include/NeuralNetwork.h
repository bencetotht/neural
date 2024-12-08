//
// Created by Bence Tóth on 03/12/2024.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <iostream>
#include "Neuron.h"
#include "Matrix.h"
#include "Layer.h"

using namespace std;
class NeuralNetwork {
public:
    NeuralNetwork(vector<int> topology);
    void setInputs(vector<double> inputs);
private:
    int topologySize;
    vector<int> topology;
    vector<Layer *> layers;
    vector<Matrix *> weights;
    vector<double> inputs;
};

#endif //NEURALNETWORK_H
