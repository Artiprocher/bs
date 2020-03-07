#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

function<Vector(Vector,Vector)> generater_loss=[](Vector y,Vector y_){
    each_index(i,y)y[i]=-1/y_[i];
    return y;
};

class Discriminator{
public:
    ParameterList PL;
    ExpandParallel< PaddingLayer<28,28,2,2>,    16       > P1;
    Parallel<       Conv2DLayer<32,32,5,5,2,2>, 16       > C1;
    DenseLayer<                                 14*14*16 > L1=DenseLayer<14*14*16>(sigmoid,sigmoid_diff);
    DropoutLayer<                               14*14*16 > D1=DropoutLayer<14*14*16>(0.3);
    Parallel<       PaddingLayer<14,14,2,2>,    16       > P2;
    ExpandParallel< DenseLayer<18*18*16>,       2        > E2;
    Parallel<       Conv2DLayer<18,18,5,5,2,2>, 32       > C2;
    DenseLayer<                                 7*7*32   > L2=DenseLayer<7*7*32>(sigmoid,sigmoid_diff);
    DropoutLayer<                               7*7*32   > D2=DropoutLayer<7*7*32>(0.3);
    DenseLayer<                                 1        > OU=DenseLayer<1>(sigmoid,sigmoid_diff);
    ComplateEdge<7*7*32,1> D2_OU;
    function<Vector(Vector,Vector)> loss=crossEntropy;
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
        cout<<"number of Discriminator parameters: "<<PL.w.size()<<endl;
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
    }
    void update(){
        OP.iterate(PL);
    }
}D;

class Generater{
public:
    ParameterList PL;
    DenseLayer<                            7*7*32   > IN;
    Parallel< PaddingLayer<7,7,3,3,1,1>,   32       > P1;
    Parallel< Conv2DLayer<19,19,6,6>,      32       > C1;
    DenseLayer<                            14*14*32 > D1=DenseLayer<14*14*32>(sigmoid,sigmoid_diff);
    DenseLayer<                            14*14*16 > L1;
    Parallel< PaddingLayer<14,14,3,3,1,1>, 16       > P2;
    Parallel< Conv2DLayer<33,33,6,6>,      16       > C2;
    DenseLayer<                            28*28*16 > D2=DenseLayer<28*28*16>(LeakyReLU,LeakyReLU_diff);
    DenseLayer<                            28*28*1  > L2=DenseLayer<28*28*1>(sigmoid,sigmoid_diff);
    Optimazer::Adam OP;
    void init(){
        for(int i=0;i<32;i++)C1[i].use_bias=0;
        for(int i=0;i<16;i++)C2[i].use_bias=0;
        D2.use_bias=0;
        L2.use_bias=0;
        IN.get_parameters(PL);
        P1.get_parameters(PL);
        C1.get_parameters(PL);
        D1.get_parameters(PL);
        L1.get_parameters(PL);
        P2.get_parameters(PL);
        C2.get_parameters(PL);
        D2.get_parameters(PL);
        L2.get_parameters(PL);
        OP.init(PL);
        cout<<"number of Generater parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        P1.clear();
        C1.clear();
        D1.clear();
        L1.clear();
        P2.clear();
        C2.clear();
        D2.clear();
        L2.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,P1);
        P1.forward_solve();
        push_forward(P1,C1);
        C1.forward_solve();
        push_forward(C1,D1);
        D1.forward_solve();
        for(int i=0;i<16;i++){
            for(int j=0;j<14*14;j++)L1.in_val[i*(14*14)+j]=D1.out_val[i*(14*14)+j]+D1.out_val[i*(14*14)+j+14*14*16];
        }
        L1.forward_solve();
        push_forward(L1,P2);
        P2.forward_solve();
        push_forward(P2,C2);
        C2.forward_solve();
        push_forward(C2,D2);
        D2.forward_solve();
        for(int i=0;i<28*28;i++){
            for(int j=0;j<16;j++)L2.in_val[i]+=D2.out_val[j*28*28+i];
        }
        L2.forward_solve();
        return L2.out_val.Array2Vector();
    }
    void train(){
        //L2.in_diff*=-1.0;
        L2.backward_solve();
        for(int i=0;i<28*28;i++){
            for(int j=0;j<16;j++)D2.in_diff[j*28*28+i]+=L2.out_diff[i];
        }
        D2.backward_solve();
        push_backward(C2,D2);
        C2.backward_solve();
        push_backward(P2,C2);
        P2.backward_solve();
        push_backward(L1,P2);
        L1.backward_solve();
        for(int i=0;i<16;i++){
            for(int j=0;j<14*14;j++){
                D1.in_diff[i*(14*14)+j]+=L1.out_diff[i*(14*14)+j];
                D1.in_diff[i*(14*14)+j+14*14*16]+=L1.out_diff[i*(14*14)+j];
            }
        }
        D1.backward_solve();
        push_backward(C1,D1);
        C1.backward_solve();
        push_backward(P1,C1);
        P1.backward_solve();
        push_backward(IN,P1);
        IN.backward_solve();
    }
    void update(){
        OP.iterate(PL);
    }
}G;

CSV_Reader csv_reader;
DataSet trainx;

Vector randVector(int n){
    Vector x(n,0);
    rep(i,0,n-1)x[i]=NormalRand(0,0.1);
    return x;
}

namespace GAN{
    void pre_train_G(int T){
        rep(i,1,T){
            Vector seed=randVector(7*7*32);
            Vector x=G.predict(seed);
            Vector e=mse(trainx.data[0],x);
            Vector2Array(e,G.L2.in_diff);
            G.train();
            G.update();
        }
    }
    void train_G(){
        Vector seed=randVector(7*7*32);
        Vector x=G.predict(seed);
        D.loss=generater_loss;
        D.train(x,(Vector){0});
        D.loss=crossEntropy;
        push_backward(G.L2,D.P1);
        G.train();
        G.update();
    }
    void train_D(const Vector &x){
        if(randint(1,2)==1){
            D.train(x,(Vector){1});
            D.update();
        }else{
            Vector seed=randVector(7*7*32);
            Vector x=G.predict(seed);
            D.train(x,(Vector){0});
            D.update();
        }
    }
    Vector generate_image(){
        Vector seed=randVector(7*7*32);
        Vector x=G.predict(seed);
        return x;
    }
}

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        //cout<<fixed<<setprecision(2)<<a[i]<<" ";
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge() {
    int all = 1000, ac = 0;
    rep(it, 0, all - 1) {
        Vector a = D.predict(trainx.data[it]);
        Vector b = D.predict(GAN::generate_image());
        if(a[0]>0.5)ac++;
        if(b[0]<0.5)ac++;
    }
    show_image(GAN::generate_image());
    cout << (ac * 1.0 / all / 2.0) << endl;
}
void demo(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("digit/train_zero.csv");
    csv_reader.shuffle();
    csv_reader.export_number_data(0, 4132-1, 1, 784, trainx);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    // train
    cout << "Training model" << endl;
    G.init();
    D.init();
    /*rep(i,1,5000){
        int idx = randint(0, 4132 - 1);
        GAN::train_D(trainx.data[idx]);
    }
    judge();
    rep(i,1,10){
        GAN::pre_train_G(100);
        show_image(GAN::generate_image());
    }
    judge();*/
    int T=90;
    while(T--){
        ll epoch = 10000;
        rep(it, 1, epoch) {
            if(randint(1,100)<=T){
                int idx = randint(0, 4132 - 1);
                GAN::train_D(trainx.data[idx]);
            }else{
                GAN::train_G();
            }
        }
        judge();
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
