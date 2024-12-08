//
// Created by Bence Tóth on 03/12/2024.
//
#include "../include/Layer.h"

#include <vector>

#include "../include/Neuron.h"

Layer::Layer(int size) {
    this->size = size;
    for (int i = 0; i <= size; i++) {
        Neuron *n = new Neuron(0);
        this->neurons.push_back(n);
    }
}

void Layer::setVal(int index, double val) {
    this->neurons[index]->setVal(val);
}