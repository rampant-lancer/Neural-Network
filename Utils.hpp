//
// Created by haque on 23/3/18.
//

#ifndef NEURALNETWORK_UTILS_HPP
#define NEURALNETWORK_UTILS_HPP

#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

class Utils
{
public :
    static float sigmoid(float );
    static float genRandom(float, float);
    static void matVectMult(const vector<float> &, const vector<vector<float> > &, vector<float> &);
};

#endif //NEURALNETWORK_UTILS_HPP
