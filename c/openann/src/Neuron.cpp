#include "../include/Neuron.h"

Neuron::Neuron(double val) {
    this->val = val;
    activate();
    derive();
}

void Neuron::activate() {
    // f(x) = x / (1 + |x|)
    this->activatedVal = this->val / ( 1 + abs(this->val));
}

void Neuron::derive() {
    // f'(x) = f(x) * (1 - f(x))
    this->derivedVal = this->activatedVal * (1 - this->activatedVal);
}

void Neuron::setVal(double val) {
    this->val = val;
    activate();
    derive();
}
