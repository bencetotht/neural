//
// Created by Bence Tóth on 03/12/2024.
//

#include "../include/Matrix.h"

#include <iostream>
#include <random>

using namespace std;

void Matrix::printMatrix() {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            cout << this->data.at(i).at(j) << "\t\t";
        }
        cout << endl;
    }
}

double Matrix::generateRandomNumber() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    return dis(gen);
}

Matrix *Matrix::transpose() {
    auto *m = new Matrix(this->numCols, this->numRows, false);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            m->setValue(j, i, this->getValue(i, j));
        }
    }

    return m;
}

Matrix::Matrix(int numRows, int numCols, bool isRandom) {
    this->numRows = numRows;
    this->numCols = numCols;

    for (int i = 0; i < this->numRows; i++) {
        vector<double> colValues;
        for (int j = 0; j < this->numCols; j++) {
            double r = isRandom ? Matrix::generateRandomNumber() : 0.00;
            colValues.push_back(r);
        }
        this->data.push_back(colValues);
    }
}
