#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

namespace net2{
    ParameterList PL;
    const int N=10;
    ExpandParallel< PaddingLayer<28,28,2,2>,    64       > P1;
    Parallel<       Conv2DLayer<32,32,5,5,2,2>, 64       > C1;
    DenseLayer<                                 14*14*64 > L1(sigmoid,sigmoid_diff);
    DropoutLayer<                               14*14*64 > D1(0.3);
    Parallel<       PaddingLayer<14,14,2,2>,    64       > P2;
    ExpandParallel< DenseLayer<18*18*64>,       2        > E2;
    Parallel<       Conv2DLayer<18,18,5,5,2,2>, 128      > C2;
    DenseLayer<                                 7*7*128  > L2(sigmoid,sigmoid_diff);
    DropoutLayer<                               7*7*128  > D2(0.3);
    DenseLayer<                                 1        > OU(sigmoid,sigmoid_diff);
    auto D2_OU=full_connect(D2,OU);
    auto loss=softmax_crossEntropy;
    Optimazer::Adam OP;
    void init(){
        P1.get_parameters(PL);
        C1.get_parameters(PL);
        L1.get_parameters(PL);
        D1.get_parameters(PL);
        P2.get_parameters(PL);
        E2.get_parameters(PL);
        C2.get_parameters(PL);
        L2.get_parameters(PL);
        D2.get_parameters(PL);
        OU.get_parameters(PL);
        D2_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        P1.clear();
        C1.clear();
        L1.clear();
        D1.clear();
        P2.clear();
        E2.clear();
        C2.clear();
        L2.clear();
        D2.clear();
        OU.clear();
        Vector2Array(x,P1.in_val);
        P1.forward_solve();
        push_forward(P1,C1);
        C1.forward_solve();
        push_forward(C1,L1);
        L1.forward_solve();
        push_forward(L1,D1);
        D1.forward_solve();
        push_forward(D1,P2);
        P2.forward_solve();
        push_forward(P2,E2);
        E2.forward_solve();
        push_forward(E2,C2);
        C2.forward_solve();
        push_forward(C2,L2);
        L2.forward_solve();
        push_forward(L2,D2);
        D2.forward_solve();
        push_forward(D2,OU,D2_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),OU.in_diff);
        OU.backward_solve();
        push_backward(D2,OU,D2_OU);
        D2.backward_solve();
        push_backward(L2,D2);
        L2.backward_solve();
        push_backward(C2,L2);
        C2.backward_solve();
        push_backward(E2,C2);
        E2.backward_solve();
        push_backward(P2,E2);
        P2.backward_solve();
        push_backward(D1,P2);
        D1.backward_solve();
        push_backward(L1,D1);
        L1.backward_solve();
        push_backward(C1,L1);
        C1.backward_solve();
        push_backward(P1,C1);
        P1.backward_solve();
        OP.iterate(PL);
    }
}

#define net net2

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
Vector randVector(int n){
    Vector x(n,0);
    rep(i,0,n-1)x[i]=Rand(0,1.0);
    return x;
}
void judge_discriminator(const DataSet &testx) {
    int all = 1000*2, ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net::predict(testx.data[it]);
        Vector b = net::predict(randVector(784));
        if(a[0]>0.5)ac++;
        if(b[0]<0.5)ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void demo(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("digit/train.csv");
    csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(0, split_position-1, 1, 784, trainx);
    csv_reader.export_onehot_data(0, split_position-1, 0, trainy);
    csv_reader.export_number_data(split_position, 42000-1, 1, 784, testx);
    csv_reader.export_onehot_data(split_position, 42000-1, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // train
    cout << "Training model" << endl;
    net::init();
    judge_discriminator(testx);
    while(1){
        ll epoch = 100;
        rep(it, 1, epoch) {
            int idx = randint(0, split_position - 1);
            net::train(trainx.data[idx], (Vector){1});
            net::train(randVector(784), (Vector){0});
        }
        judge_discriminator(testx);
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
