#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i, a, b) for (int i = (a); i <= (int)(b); i++)
using namespace std;
typedef long long ll;

/*激活函数*/
std::function<double(double)> constant = [](double x) { return x; };
std::function<double(double)> constant_diff = [](double x) { return 1.0; };
std::function<double(double)> sigmoid = [](double x) {
    return 1.0 / (1.0 + exp(-x));
};
std::function<double(double)> sigmoid_diff = [](double x) {
    double temp = exp(-x);
    return temp / ((1.0 + temp) * (1.0 + temp));
};

/*损失函数(导数)*/
std::function<Vector(Vector,Vector)> mse=[](Vector y,Vector y_){
    each_index(i,y)y[i]=y_[i]-y[i];
    return y;
};
std::function<Vector(Vector,Vector)> crossEntropy=[](Vector y,Vector y_){
    each_index(i,y)y[i]=-y[i]/y_[i]+(1-y[i])/(1-y_[i]);
    return y;
};

class Layer{
public:
    virtual void show(ostream &o) const {o << " Empty Layer" << endl;}
};

template <const int N>
class ActiveLayer:public Layer{
public:
    /*神经元个数*/
    static const int input_size=N,output_size=N;
    /*神经元的传入数据、传出数据、导数值、阈值*/
    double in_val[N],out_val[N],diff_val[N],c[N];
    /*激活函数与激活函数导数*/
    function<double(double)> f,f_;
    /*是否需要启用阈值 1:需要 0:不需要*/
    int threshold=0;
    /*RNN相关 是否启用RNN神经元、前一个时间点的值、权重*/
    int recurrent=0;
    double last_val[N],c_t[N];
    /*输出神经元相关信息*/
    virtual void show(ostream &o){o<<" I am a ActiveLayer."<<endl;}
    /*重置权重*/
    void reset_weight(double l=0.5,double r=0.5){
        for(int i=0;i<N;i++)c[i]=Rand(l,r);
    }
    /*构造函数*/
    ActiveLayer<N>(){reset_weight();}
    ActiveLayer<N>(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){
        threshold=1;
    }
    /*开启RNN神经元*/
    void turn_on_recurrent(){
        recurrent=1;
        for(int i=0;i<N;i++)last_val=Rand(0,1);
        for(int i=0;i<N;i++)c_t[i]=Rand(-0.5,0.5);
    }
    /*清理*/
    void clear(){
        fill(in_val,in_val+N,0.0);
        fill(diff_val,diff_val+N,0.0);
    }
    /*正向传播计算*/
    void forward_solve(){
        if(threshold==1){
            for(int j=0;j<N;j++)in_val[j]+=c[j];
        }
        if(recurrent==1){
            for(int j=0;j<N;j++)in_val[j]+=c_t[j]*last_val[j];
        }
        for(int j=0;j<N;j++)out_val[j]=f(in_val[j]);
    }
    /*更新权重*/
    void update_w(double eta){
        if(threshold==1){
            for(int i=0;i<N;i++)c[i]-=eta*diff_val[i];
        }
        if(recurrent==1){
            for(int i=0;i<N;i++)c_t[i]-=eta*diff_val[i]*last_val[i];
            for(int i=0;i<N;i++)last_val[i]=out_val[i];
        }
    }
};

template <const int N,const int M>
class Matrix{
public:
    double data[N*M];
    void show(){for(int i=0;i<N;i++)for(int j=0;j<M;j++)std::cout<<data[i*M+j]<<",\n"[j+1==M];}
    void clear(){for(int i=0;i<N*M;i++)data[i]=0;}
    double& operator [] (int x){return data[x];}
    double& operator () (int x,int y){return data[x*M+y];}
};

template <const int H_in,const int W_in,const int H_c,const int W_c>
class ConvLayer:public Layer{
public:
    static const int H_out=H_in-H_c+1,W_out=W_in-W_c+1;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    Matrix<H_c,W_c> w;
    Matrix<H_in,W_in> in_val;
    Matrix<H_out,W_out> out_val,diff_val;
    void reset_weight(double l=-0.5,double r=0.5){
        for(int i=0;i<H_c;i++){
            for(int j=0;j<W_c;j++){
                w(i,j)=Rand(l,r);
            }
        }
    }
    ConvLayer<H_in,W_in,H_c,W_c>(){reset_weight();}
    void clear(){
        in_val.clear();
        diff_val.clear();
    }
    void forward_solve(){
        out_val.clear();
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)+=w(r,c)*in_val(i+r,j+c);
            }
        }
    }
    void update_w(double eta){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++)if(fabs(diff_val(i,j))>(1e-8)){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                w(r,c)-=eta*diff_val(i,j)*in_val(i+r,j+c);
            }
        }
    }
};
template <const int H_in,const int W_in,const int H_c,const int W_c>
class MaxPoolLayer:public Layer{
public:
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    Matrix<H_in,W_in> in_val;
    Matrix<H_out,W_out> out_val,diff_val;
    void clear(){}
    void forward_solve(){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            out_val(i,j)=in_val(i*H_c,j*W_c);
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)=max(out_val(i,j),in_val(i*H_c+r,j*W_c+c));
            }
        }
    }
    void update_w(double eta){
        ;
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
template <class LayerType1,class LayerType2>
ComplateEdge<LayerType1::output_size,LayerType2::input_size> full_connect(LayerType1 &A,LayerType2 &B){
    ComplateEdge<LayerType1::output_size,LayerType2::input_size> E;
    E.init(-0.5,0.5);
    return E;
}

/*???Layer -> ActiveLayer*/
template <class LayerType,const int N,const int M>
void push_forward(LayerType &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            B.in_val[j]+=A.out_val[i]*E.w[i][j];
        }
    }
    B.forward_solve();
}
template <class LayerType,const int N,const int M>
void push_backward(LayerType &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E,double eta){
    for(int j=0;j<M;j++){
        B.diff_val[j]*=B.f_(B.in_val[j]);
        for(int i=0;i<N;i++){
            A.diff_val[i]+=B.diff_val[j]*E.w[i][j];
            E.w[i][j]-=eta*B.diff_val[j]*A.out_val[i];
        }
    }
}

/*???Layer -> ConvLayer*/
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_forward(LayerType &A,ConvLayer<H_in,W_in,H_c,W_c> &B){
    for(int i=0;i<H_in*W_in;i++)B.in_val[i]=A.out_val[i];
    B.forward_solve();
}
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_backward(LayerType &A,ConvLayer<H_in,W_in,H_c,W_c> &B,double eta){
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    for(int i=0;i<H_in;i++)for(int j=0;j<W_in;j++){
        A.diff_val[i*H_in+j]=0;
        for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
            int ii=i+r-H_c+1,jj=j+c-W_c+1;
            if(ii>=0 && ii<H_out && jj>=0 && jj<W_out){
                A.diff_val[i*H_in+j]+=B.diff_val(ii,jj)*B.w(H_c-r-1,W_c-c-1);
            }
        }
    }
}

/*???Layer -> MaxPoolLayer*/
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_forward(LayerType &A,MaxPoolLayer<H_in,W_in,H_c,W_c> &B){
    for(int i=0;i<A.n;i++)B.in_val[i]=A.out_val[i];
    B.forward_solve();
}
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_backward(LayerType &A,MaxPoolLayer<H_in,W_in,H_c,W_c> &B){
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    int max_i=0,max_j=0;
    for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
        for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++)if(B.out_val(i,j)==B.in_val(i*H_c+r,j*W_c+c)){
            max_i=i*H_c+r;
            max_j=j*W_c+c;
        }
        A.diff_val(max_i,max_j)=B.diff_val(i,j);
    }
}

namespace net1{
    double eta=0.05;
    ActiveLayer<784> input_layer;
    ActiveLayer<100> hidden_layer(sigmoid,sigmoid_diff);
    ActiveLayer<10> output_layer(sigmoid,sigmoid_diff);
    auto E1=full_connect(input_layer,hidden_layer);
    auto E2=full_connect(hidden_layer,output_layer);
    auto loss=crossEntropy;
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
        static Vector y(output_layer.output_size,0);
        for(int i=0;i<output_layer.output_size;i++){
            y[i]=output_layer.out_val[i];
        }
        return y;
    }
    void train(const Vector &x,const Vector &y){
        /*正向传值*/
        Vector y_=predict(x);
        /*逆向传值*/
        Vector2Array(loss(y,y_),output_layer.diff_val);
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
        Vector a = net1::predict(testx.data[it]);
        int ans = 0;
        rep(i, 1, 9) {
            if (a[i] > a[ans]) ans = i;
        }
        if (testy.data[it][ans] > 0.5) ac++;
    }
    cout << (ac * 1.0 / all) << endl;
}
void demo1(){
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
    net1::init();
    judge(testx, testy);
    ll epoch = 1000000, goal = 1;
    rep(it, 1, epoch) {
        int idx = randint(0, split_position - 1);
        net1::train(trainx.data[idx], trainy.data[idx]);
        if (it * 100 >= epoch * goal) {
            cout << it * 100.0 / epoch << "%" << endl;
            if (goal % 10 == 0) {
                cout << "accuracy:";
                judge(testx, testy);
            }
            goal++;
        }
    }
}

namespace net2{
    double eta=0.05;
    ConvLayer<28,28,5,5> C1;
    MaxPoolLayer<14,14,2,2> S2;
    ActiveLayer<49> L3;
    auto E3=full_connect(S2,L3);
    auto loss=mse;
    void init(){
        ;
    }
    Vector predict(const Vector &x){
        return x;
    }
    void train(const Vector &x,const Vector &y){
        ;
    }
}

int main() {
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

    return 0;
}
/*

*///
