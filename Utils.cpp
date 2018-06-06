//
// Created by haque on 23/3/18.
//

#include "Utils.hpp"

float Utils::genRandom(float LO, float HI) {
    return (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
}

float Utils::sigmoid(float x) {
    return 1/(1+exp(-x));
}

void Utils::matVectMult(const vector<float> &input,const vector<vector<float> > &weights, vector<float> &output) {

    float n_rows = weights.size(), n_cols = weights[0].size();

    float temp = 0;
    for(int i=0;i<n_rows;++i){
        temp = 0;
        for(int j=0;j<n_cols;++j){
            temp += weights[i][j]*input[j];
        }
        output[i] = Utils::sigmoid(temp);
    }
}