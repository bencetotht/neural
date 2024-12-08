//
// Created by Bence Tóth on 03/12/2024.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
using namespace std;

class Matrix {
public:
    Matrix(int numRows, int numCols, bool isRandom);

    Matrix *transpose();
    double generateRandomNumber();

    void printMatrix();

    void setValue(int r, int c, double v) { this->data[r][c] = v; } // row, column, value
    double getValue(int r, int c) {return this->data[r][c]; }; // row, column

    int getNumRows() { return this->numRows; }
    int getNumCols() { return this->numCols; }

private:
    int numRows;
    int numCols;

    vector< vector<double> > data;
};

#endif //MATRIX_H
