//
// Created by Bence Tóth on 03/12/2024.
//

#include "../include/NeuralNetwork.h"

using namespace std;

void NeuralNetwork::setInputs(vector<double> inputs) {
    this->inputs = inputs;
    for (int i = 0; i < this->topology[0]; i++) {
        this->layers[0]->setVal(i, inputs[i]);
    }
}


NeuralNetwork::NeuralNetwork(vector<int> topology) {
    this->topology = topology;
    this->topologySize = topology.size();

    // Create layers
    for (int i = 0; i < topologySize; i++) {
        Layer *l = new Layer(topology[i]);
        this->layers.push_back(l);
    }
    // Create weights
    for (int i = 0; i < (topologySize - 1); i++) {
        Matrix *m = new Matrix(topology[i], topology[i+1], true);
        this->weights.push_back(m);
    }
}

void NeuralNetwork::printToConsole() {
    for (int i = 0; i < this->topologySize; i++) {
        cout << "Layer " << i << endl;
        cout << "Neurons: " << this->topology[i] << endl;
        cout << "Weights: " << endl;
        if (i < this->topologySize - 1) {
            this->weights[i]->printMatrix();
        }
    }
}
