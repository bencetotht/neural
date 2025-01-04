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

Matrix *Layer::matrixifyVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i <= this->neurons.size(); i++) {
        m->setValue(1, i, this->neurons.at(i)->getVal());
    }
}
Matrix *Layer::matrixifyActivatedVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i<= this->neurons.size(); i++) {
        m->setValue(1, i, this->neurons.at(i)->getActivatedVal());
    }
}
Matrix *Layer::matrixifyDerivedVals() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (int i = 0; i<= this->neurons.size(); i++) {
        m->setValue(1, i, this->neurons.at(i)->getDerivedVal());
    }
}
