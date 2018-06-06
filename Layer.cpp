//
// Created by haque on 23/3/18.
//

#include "Layer.hpp"
#include "Utils.hpp"

void Layer::INIT(int _n_inputs, int _n_nodes) {

    n_nodes = _n_nodes;
    n_inputs = _n_inputs;

    X.resize(n_inputs);
    V.resize(n_nodes);

    W.resize(n_nodes, vector<float>(n_inputs));

    for(int row=0;row<n_nodes;++row){
        for(int col=0;col<n_inputs;++col){
            this->W[row][col] = Utils::genRandom(-1.0, 1.0);
        }
    }
}

void Layer::preProcess(const vector<float> &x_from_prev_layer) {
    // Setting Up Inputs.
    X[0] = 1.0;
    for(int i=1;i<n_inputs;++i)
        X[i] = x_from_prev_layer[i-1];

    // Calling Util's matVectMult To Calculate Output From
    // The Current Layer.

    Utils::matVectMult(X,W,V);
}

float Layer::trainLayer(vector<float> Y) {         // For Output Layer.
    float tError = 0, cError = 1;
    for(int k=0;k<n_nodes;++k){
        cError = (Y[k]-V[k])*V[k]*(1-V[k]);
        for(int j=0;j<n_inputs;++j){
            tError += cError*W[k][j];
            W[k][j] += eta*cError*X[j];
        }
    }
    return tError;
}

float Layer::trainLayer(float totalError) {        // For Hidden Layers.
    float cError = 1, tError = totalError;
    for(int j = 0;j<n_nodes;++j){
        cError = V[j]*(1-V[j])*totalError;
        for(int i=0;i<n_inputs;++i){
            tError += cError*W[j][i];
            W[j][i] += eta*cError*X[i];
        }
    }
    return tError;
}


vector<float> Layer::testLayer(const vector<float> &x_from_prev_layer) {
    preProcess(x_from_prev_layer);
    return V;
}

void Layer::print_weight() {
    printf("\n========== Printing Weights ==========\n");
    for(int row=0;row<n_nodes;++row){
        for(int col=0;col<n_inputs;++col){
            printf("%f\t",W[row][col]);
        }
        printf("\n");
    }
}

vector<float> Layer::getOutput(const vector<float> &x_from_prev_layer) {
    preProcess(x_from_prev_layer);
    return V;
}