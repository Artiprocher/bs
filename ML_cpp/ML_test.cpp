#include <bits/stdc++.h>
#include "ML_Tensor.h"
#include "ML_Vector.h"
#include "ML_Linear_Model.h"
#include "ML_Neural_Network.h"
#define rep(i,a,b) for(int i=a;i<=b;i++)

NeuralNetwork net({2,2,1},{0,0,0});

int main() {
    std::ios::sync_with_stdio(false);
    net.eta=0.1;
    net.show();
    rep(i,1,100000){
        Vector x={Rand(),Rand()},y={x[0]+x[1]};
        net.train(x,y);
    }
    net.show();
    double x1,x2;
    while(std::cin>>x1>>x2){
        std::cout<<net.predict({x1,x2})<<std::endl;
    }
    return 0;
}
/*

*///
