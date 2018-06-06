//
// Created by haque on 23/3/18.
//

#include "Network.hpp"

Network::Network(const vector<int> &_layers, int _n_train, int _n_test) {
    n_layers = _layers.size() - 1;
    n_classes = _layers[n_layers];
    n_features = _layers[0];
    n_train = _n_train;
    n_test = _n_test;
    n_total = n_train + n_test;


    x_train.resize(n_train, vector<float>(n_features));
    x_test.resize(n_test, vector<float>(n_features));

    y_train.resize(n_train);
    y_test.resize(n_test);

    X.resize(n_features);
    Y.resize(n_classes);

    net.resize(n_layers);

    int inp, opt;
    for(int i=0;i<n_layers;++i){
        inp = _layers[i] + 1, opt = _layers[i+1];
        net[i].INIT(inp, opt);

    }

}

void Network::load_data() {
    for(int row=0;row<n_train;++row){
        for(int col=0;col<n_features;++col){
            scanf("%f",&x_train[row][col]);
        }
        scanf("%d",&y_train[row]);
    }

    for(int row=0;row<n_test;++row){
        for(int col=0;col<n_features;++col){
            scanf("%f",&x_test[row][col]);
        }
        scanf("%d",&y_test[row]);
    }

}

void Network::feed_data(int idx, bool isTraining) {
    if(isTraining){
        for(int i=0;i<n_features;++i)
            X[i] = x_train[idx][i];
        for(int i=0;i<n_classes;++i)
            Y[i] = 0;
        Y[y_train[idx]] = 1;
    }
    else{
        for(int i=0;i<n_features;++i)
            X[i] = x_test[idx][i];
        for(int i=0;i<n_classes;++i)
            Y[i] = 0;
        Y[y_test[idx]] = 1;
    }
}

void Network::train() {
    vector<float> temp;
    float tError = 0;
    for(int i=0;i<n_train;++i){

        feed_data(i, true);
        temp = X;
        for(int i=0;i<n_layers;++i)
            temp = net[i].getOutput(temp);                       // This Returns V.

        tError = 0;
        for(int i=1;i<=n_layers;++i){
            if(i==1)
                tError += net[n_layers - i].trainLayer(Y);
            else
                tError += net[n_layers - i].trainLayer(tError);
        }
    }
}

void Network::test() {
    vector<float> temp;
    int Error = 0;
    for(int i=0;i<n_test;++i){

        feed_data(i, false);
        temp = X;
        for(int i=0;i<n_layers;++i)
            temp = net[i].testLayer(temp);

        int idx = 0;
        for(int i=1;i<n_classes;++i)
            idx = (temp[i]>temp[idx])?i:idx ;

        if(idx == y_test[i]){
            printf("%d\tCorrect!",i+1);
        }else{
            Error++;
            printf("%d\tWrong!",i+1);
        }
        printf("\n");
    }
    printf("Total number of misclassification  = %d\n",Error);
}

void Network::print_layer_weight() {
    printf("\n========== Weights Are Printed In The Direction Of Output From Input ==========\n");
    for(int i=0;i<n_layers;++i)
        net[i].print_weight();
}