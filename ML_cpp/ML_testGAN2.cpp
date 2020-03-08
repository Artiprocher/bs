#define DEBUG
#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

const int seed_size=100;

/*class Discriminator{
public:
    ParameterList PL;
    static const int N=10;
    DenseLayer<784> IN;
    ExpandParallel< ConvLayer<28,28,5,5>,N > C1;
    Parallel< MaxPoolLayer<24,24,2,2>,N > S2;
    Parallel< DenseLayer<144>,N > D3;
    ComplateEdge<144*N,1> D3_OU;
    DenseLayer<1> OU=DenseLayer<1>(sigmoid,sigmoid_diff);
    Optimazer::Adam OP;
    void init(){
        for(int i=0;i<N;i++)D3[i]=DenseLayer<144>(sigmoid,sigmoid_diff);
        IN.get_parameters(PL);
        C1.get_parameters(PL);
        S2.get_parameters(PL);
        D3.get_parameters(PL);
        D3_OU.get_parameters(PL);
        OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of Discriminator parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        C1.clear();
        S2.clear();
        D3.clear();
        OU.clear();
        each_index(i,x)IN.out_val[i]=x[i];
        push_forward(IN,C1);
        C1.forward_solve();
        push_forward(C1,S2);
        S2.forward_solve();
        push_forward(S2,D3);
        D3.forward_solve();
        push_forward(D3,OU,D3_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y,const function<Vector(Vector,Vector)> &loss){
        assert(x.size()==784 && y.size()==1);
        Vector y_=predict(x);
        Vector d=loss(y,y_);
        Vector2Array(d,OU.in_diff);
        OU.backward_solve();
        push_backward(D3,OU,D3_OU);
        D3.backward_solve();
        push_backward(S2,D3);
        S2.backward_solve();
        push_backward(C1,S2);
        C1.backward_solve();
        push_backward(IN,C1);
        IN.backward_solve();
    }
    void update(){
        OP.iterate(PL);
    }
}D;*/

function<Vector(Vector,Vector)> generater_loss=crossEntropy;
function<Vector(Vector,Vector)> discriminator_loss=crossEntropy;

class Discriminator{
public:
    ParameterList PL;
    DenseLayer<28*28> IN;
    DenseLayer<128> H1=DenseLayer<128>(LeakyReLU,LeakyReLU_diff);
    DenseLayer<1> OU=DenseLayer<1>(sigmoid,sigmoid_diff);
    ComplateEdge<28*28,128> IN_H1=full_connect(IN,H1);
    ComplateEdge<128,1> H1_OU=full_connect(H1,OU);
    Optimazer::GradientDescent OP=Optimazer::GradientDescent(0.001);
    void init(){
        IN.get_parameters(PL);
        H1.get_parameters(PL);
        OU.get_parameters(PL);
        IN_H1.get_parameters(PL);
        H1_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of Discriminator parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        H1.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,H1,IN_H1);
        H1.forward_solve();
        push_forward(H1,OU,H1_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(const Vector &x,const Vector &y,const function<Vector(Vector,Vector)> &loss){
        assert(x.size()==784 && y.size()==1);
        Vector y_=predict(x);
        Vector d=loss(y,y_);
        Vector2Array(d,OU.in_diff);
        OU.backward_solve();
        push_backward(H1,OU,H1_OU);
        H1.backward_solve();
        push_backward(IN,H1,IN_H1);
        IN.backward_solve();
    }
    void update(){
        OP.iterate(PL);
    }
}D;

class Generater{
public:
    ParameterList PL;
    DenseLayer<100> IN;
    DenseLayer<128> H1=DenseLayer<128>(LeakyReLU,LeakyReLU_diff);
    DropoutLayer<128> Dr=DropoutLayer<128>(0.2);
    DenseLayer<28*28> OU=DenseLayer<28*28>(Tanh,Tanh_diff);
    ComplateEdge<100,128> IN_H1=full_connect(IN,H1);
    ComplateEdge<128,28*28> Dr_OU=full_connect(Dr,OU);
    Optimazer::GradientDescent OP=Optimazer::GradientDescent(0.001);
    void init(){
        IN.get_parameters(PL);
        H1.get_parameters(PL);
        Dr.get_parameters(PL);
        OU.get_parameters(PL);
        IN_H1.get_parameters(PL);
        Dr_OU.get_parameters(PL);
        OP.init(PL);
        cout<<"number of Generater parameters: "<<PL.w.size()<<endl;
    }
    Vector predict(const Vector &x){
        IN.clear();
        H1.clear();
        Dr.clear();
        OU.clear();
        Vector2Array(x,IN.in_val);
        IN.forward_solve();
        push_forward(IN,H1,IN_H1);
        H1.forward_solve();
        push_forward(H1,Dr);
        Dr.forward_solve();
        push_forward(Dr,OU,Dr_OU);
        OU.forward_solve();
        return OU.out_val.Array2Vector();
    }
    void train(){
        OU.backward_solve();
        push_backward(Dr,OU,Dr_OU);
        Dr.backward_solve();
        push_backward(H1,Dr);
        H1.backward_solve();
        push_backward(IN,H1,IN_H1);
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
    void train_G(){
        Vector seed=randVector(seed_size);
        Vector x=G.predict(seed);
        D.train(x,(Vector){0.9},generater_loss);
        push_backward(G.OU,D.IN);
        G.train();
        G.update();
    }
    void train_D(const Vector &x){
        if(randint(1,2)==1){
            D.train(x,(Vector){0.9},discriminator_loss);
            D.update();
        }else{
            Vector seed=randVector(seed_size);
            Vector x=G.predict(seed);
            D.train(x,(Vector){0},discriminator_loss);
            D.update();
        }
    }
    Vector generate_image(){
        Vector seed=randVector(seed_size);
        Vector x=G.predict(seed);
        return x;
    }
}

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.0?"*":" ");
        //cout<<fixed<<setprecision(2)<<a[i]<<" ";
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void save_image(const Vector &a,const char *file_name){
    ofstream f;
    f.open(file_name,ios::out);
    for(auto i:a)f<<fixed<<setprecision(5)<<i<<endl;
    f.close();
}
double judge() {
    int all = 1000, ac = 0;
    rep(it, 0, all - 1) {
        Vector a = D.predict(trainx.data[it]);
        Vector b = D.predict(GAN::generate_image());
        if(a[0]>0.5)ac++;
        if(b[0]<0.5)ac++;
    }
    show_image(GAN::generate_image());
    cout << (ac * 1.0 / all / 2.0) << endl;
    return (ac * 1.0 / all / 2.0);
}
void demo(){
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("digit/train_zero.csv");
    csv_reader.shuffle();
    csv_reader.export_number_data(0, 4132-1, 1, 784, trainx);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] = trainx.data[i]*(2.0/255)-1.0;
    // train
    cout << "Training model" << endl;
    G.init();
    D.init();
    double ac=judge();
    int T=300;
    while(T--){
        ll epoch = 10000;
        ll p=ac<0.9?70:30;
        rep(it, 1, epoch) {
            if(randint(1,100)<=p){
                int idx = randint(0, 4132 - 1);
                GAN::train_D(trainx.data[idx]);
            }else{
                GAN::train_G();
            }
        }
        ac=judge();
        save_image(GAN::generate_image(),"vis/data.txt");
        system("python \"c:/git/bs/ML_cpp/vis/ploter.py\"");
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
