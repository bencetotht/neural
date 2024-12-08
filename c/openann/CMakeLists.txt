cmake_minimum_required(VERSION 3.30)
project(openann)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_STANDARD 20)

add_executable(
        openann 
        src/main.cpp
        include/Neuron.h
        src/Neuron.cpp
        include/Layer.h
        src/Layer.cpp
        include/Matrix.h
        src/Matrix.cpp
        include/NeuralNetwork.h
        src/NeuralNetwork.cpp
)
