#include "Network.hpp"

int main() {
    freopen("Data/in.txt","r",stdin);
    freopen("Data/out.txt","w",stdout);

    int epoch = 0;
    int _n_layers = 0, _n_train = 0, _n_test = 0, temp = 0;
    vector<int> _layers ;
    scanf("%d",&epoch);
    scanf("%d%d%d",&_n_layers,&_n_train,&_n_test);
    for(int i=0;i<_n_layers;++i){
        scanf("%d",&temp);
        _layers.push_back(temp);
    }
    Network NN = Network(_layers, _n_train, _n_test);
    NN.load_data();
    printf("\n========== Initial Weights ==========\n");
    NN.print_layer_weight();
    for(int i=0;i<epoch;++i)
        NN.train();
    printf("\n========== Final Weights ==========\n");
    NN.print_layer_weight();
    NN.test();
    return 0;
}