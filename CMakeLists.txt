cmake_minimum_required(VERSION 3.9)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork main.cpp Utils.cpp Utils.hpp Layer.cpp Layer.hpp Network.cpp Network.hpp)