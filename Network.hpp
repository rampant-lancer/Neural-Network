//
// Created by haque on 23/3/18.
//

#ifndef NEURALNETWORK_NETWORK_HPP
#define NEURALNETWORK_NETWORK_HPP

#include <vector>
#include "Layer.hpp"

using namespace std;

class Network
{
private :
    int n_train = 0;
    int n_test = 0;
    int n_features = 0;
    int n_classes = 0;
    int n_total = 0;
    int n_layers = 0;

    vector<vector<float> > x_train, x_test;
    vector<int> y_train, y_test;
    vector<float> X, Y;

    vector<Layer> net;

public :
    Network(const vector<int> &, int, int);
    void load_data();
    void feed_data(int, bool);
    void train();
    void test();
    void print_layer_weight();
};

#endif //NEURALNETWORK_NETWORK_HPP
