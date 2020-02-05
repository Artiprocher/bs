#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i, a, b) for (int i = (a); i <= (int)(b); i++)
using namespace std;
typedef long long ll;

std::function<double(double)> constant = [](double x) { return x; };
std::function<double(double)> constant_diff = [](double x) { return 1.0; };
std::function<double(double)> sigmoid = [](double x) {
    return 1.0 / (1.0 + exp(-x));
};
std::function<double(double)> sigmoid_diff = [](double x) {
    double temp = exp(-x);
    return temp / ((1.0 + temp) * (1.0 + temp));
};

class Layer{
public:
    virtual void show(ostream &o) const {o << " Empty Layer" << endl;}
};

template <const int N>
class ActiveLayer:public Layer{
public:
    /*神经元个数*/
    const int n=N;
    /*神经元的传入数据、传出数据、导数值、阈值*/
    double in_val[N],out_val[N],diff_val[N],c[N];
    /*激活函数与激活函数导数*/
    function<double(double)> f,f_;
    /*是否需要启用阈值 1:需要 0:不需要*/
    int threshold=0;
    /*输出神经元相关信息*/
    virtual void show(ostream &o){o<<" I am a ActiveLayer."<<endl;}
    /*重置权重*/
    void reset_weight(double l=0.5,double r=0.5){
        for(int i=0;i<N;i++)c[i]=Rand(l,r);
    }
    /*构造函数*/
    ActiveLayer<N>(){}
    ActiveLayer<N>(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){
        reset_weight();
        threshold=1;
    }
    /*清理*/
    void clear(){
        fill(in_val,in_val+N,0.0);
        fill(diff_val,diff_val+N,0.0);
    }
    /*更新权重*/
    void update_w(double eta){
        if(threshold==1){
            for(int i=0;i<N;i++)c[i]-=eta*diff_val[i];
        }
    }
};

template <const int N,const int M>
class ComplateEdge{
public:
    double w[N][M],dw[N][M];
    /*随机初始化权重*/
    void init(double l,double r){
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                w[i][j]=Rand(l,r);
            }
        }
    }
};

template <const int N,const int M>
ComplateEdge<N,M> connect(ActiveLayer<N> &A,ActiveLayer<M> &B){
    ComplateEdge<N,M> E;
    E.init(-0.5,0.5);
    return E;
}
template <const int N,const int M>
void push_forward(ActiveLayer<N> &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            B.in_val[j]+=A.out_val[i]*E.w[i][j];
        }
    }
    if(B.threshold==1){
        for(int j=0;j<M;j++)B.in_val[j]+=B.c[j];
    }
    for(int j=0;j<M;j++)B.out_val[j]=B.f(B.in_val[j]);
}
template <const int N,const int M>
void push_backward(ActiveLayer<N> &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E,double eta){
    for(int j=0;j<M;j++){
        B.diff_val[j]*=B.f_(B.in_val[j]);
        for(int i=0;i<N;i++){
            A.diff_val[i]+=B.diff_val[j]*E.w[i][j];
            E.w[i][j]-=eta*B.diff_val[j]*A.out_val[i];
        }
    }
}

namespace net{
    double eta=0.5;
    ActiveLayer<784> input_layer;
    ActiveLayer<100> hidden_layer(sigmoid,sigmoid_diff);
    ActiveLayer<10> output_layer(sigmoid,sigmoid_diff);
    auto E1=connect(input_layer,hidden_layer);
    auto E2=connect(hidden_layer,output_layer);
    void init(){
        ;
    }
    Vector predict(const Vector &x){
        /*清理*/
        input_layer.clear();
        hidden_layer.clear();
        output_layer.clear();
        /*正向传值*/
        each_index(i,x)input_layer.out_val[i]=x[i];
        push_forward(input_layer,hidden_layer,E1);
        push_forward(hidden_layer,output_layer,E2);
        /*导出结果*/
        static Vector y(output_layer.n,0);
        for(int i=0;i<output_layer.n;i++){
            y[i]=output_layer.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        predict(x);
        /*逆向传值*/
        for(int i=0;i<output_layer.n;i++){
            output_layer.diff_val[i]=output_layer.out_val[i]-y[i];
        }
        push_backward(hidden_layer,output_layer,E2,eta);
        push_backward(input_layer,hidden_layer,E1,eta);
        /*更新权重*/
        hidden_layer.update_w(eta);
        output_layer.update_w(eta);
    }
};

CSV_Reader csv_reader;
DataSet trainx,trainy,testx,testy;

void show_image(const Vector &a){
    rep(i,0,783){
        cout<<(a[i]>0.5?"*":" ");
        if((i+1)%28==0)cout<<endl;
    }
    cout<<endl;
}
void judge(const DataSet &testx, const DataSet &testy) {
    int all = testx.data.size(), ac = 0;
    rep(it, 0, all - 1) {
        Vector a = net::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}

int main() {
    // read data
    cout << "Reading data" << endl;
    csv_reader.open("train.csv");
    csv_reader.shuffle();
    int split_position = 30000;
    csv_reader.export_number_data(1, split_position, 1, 784, trainx);
    csv_reader.export_onehot_data(1, split_position, 0, trainy);
    csv_reader.export_number_data(split_position + 1, 42000, 1, 784, testx);
    csv_reader.export_onehot_data(split_position + 1, 42000, 0, testy);
    csv_reader.close();
    rep(i, 0, trainx.data.size() - 1) trainx.data[i] *= 1.0 / 255;
    rep(i, 0, testx.data.size() - 1) testx.data[i] *= 1.0 / 255;
    // train
    cout << "Training model" << endl;
    net::init();
    judge(testx, testy);
    ll epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net::train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge(testx, testy);
            }
            goal++;
        }
    }
    return 0;
}
/*

*///
