//
// Created by haque on 23/3/18.
//

#ifndef NEURALNETWORK_LAYER_HPP
#define NEURALNETWORK_LAYER_HPP

#include <vector>
#include <cstdio>
#include <iostream>


using namespace std;

class Layer
{
private :
    int n_nodes = 0;
    int n_inputs = 0;
    const float eta = 0.1;

    vector<vector<float> > W;
    vector<float> X, V;

public :
    void INIT(int, int);     // Constructor For All Initializations.
    float trainLayer(float);                        // Training Hidden Layers.
    float trainLayer(vector<float>);                // Training Output Layers.
    void print_weight();                                                   // Printing Weights Of The Current Layer.
    vector<float> testLayer(const vector<float> &);
    void preProcess(const vector<float> &);
    vector<float> getOutput(const vector<float> &);
};

#endif //NEURALNETWORK_LAYER_HPP
