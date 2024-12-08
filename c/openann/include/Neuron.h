//
// Created by Bence Tóth on 02/12/2024.
//

#ifndef NEURON_H
#define NEURON_H

#include <iostream>
using namespace std;

class Neuron {
public:
    Neuron(double val);

    void setVal(double val);

    // Fast Sigmoid Activation Function
    // f(x) = x / (1 + |x|)
    void activate();

    // Derivative of Fast Sigmoid Activation Function
    // f'(x) = f(x) * (1 - f(x))
    void derive();

    // Getters
    double getVal() { return this->val; }
    double getActivatedVal() { return this->activatedVal; }
    double getDerivedVal() { return this->derivedVal; }
private:
    double val;
    double activatedVal; // val after activation function
    double derivedVal; // derived from activatedVal
    double bias;
};

#endif //NEURON_H