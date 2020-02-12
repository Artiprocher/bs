#include <bits/stdc++.h>
#include "ML_Vector.h"
#include "ML_Rand.h"
#include "ML_Data_Reader.h"
using namespace std;

const double init_L=-0.5,init_R=0.5;

//智能数组 用[]使用一维索引 用()使用二维索引
template <const int N,const int M>
class SmartArray{
public:
    double data[N*M];
    void show(){for(int i=0;i<N;i++)for(int j=0;j<M;j++)std::cout<<data[i*M+j]<<",\n"[j+1==M];}
    void clear(){for(int i=0;i<N*M;i++)data[i]=0;}
    double& operator [] (int x){return data[x];}
    double& operator () (int x,int y){return data[x*M+y];}
    void reset_weight(double l,double r){for(int i=0;i<N*M;i++)data[i]=Rand(l,r);}
};
#define ComplateEdge SmartArray

/*损失函数(导数)*/
function<Vector(Vector,Vector)> mse=[](Vector y,Vector y_){
    each_index(i,y)y[i]=y_[i]-y[i];
    return y;
};
function<Vector(Vector,Vector)> crossEntropy=[](Vector y,Vector y_){
    each_index(i,y)y[i]=-y[i]/y_[i]+(1-y[i])/(1-y_[i]);
    return y;
};
function<Vector(Vector,Vector)> singleCrossEntropy=[](Vector y,Vector y_){
    each_index(i,y)if(y[i]>0.5){
        y[i]=-1.0/y_[i];
        break;
    }
    return y;
};
function<Vector(Vector,Vector)> softmax_crossEntropy=[](Vector y,Vector y_){
    double sum=0;
    each_index(i,y)y_[i]=exp(y_[i]),sum+=y_[i];
    each_index(i,y){
        y_[i]=y_[i]/sum;
        if(y[i]>0.5)y_[i]-=1.0;
    }
    return y_;
};

//激活函数
function<double(double)> constant = [](double x) { return x; };
function<double(double)> constant_diff = [](double x) { return 1.0; };
function<double(double)> sigmoid = [](double x) {
    return 1.0 / (1.0 + exp(-x));
};
function<double(double)> sigmoid_diff = [](double x) {
    double temp = exp(-x);
    return temp / ((1.0 + temp) * (1.0 + temp));
};
function<double(double)> relu = [](double x) {
    return x>0.0?x:0.0;
};
function<double(double)> relu_diff = [](double x) {
    return x>0.0?1.0:0.0;
};
function<double(double)> Tanh = [](double x) {
    double a=exp(x),b=exp(-x);
    return (a-b)/(a+b);
};
function<double(double)> Tanh_diff = [](double x) {
    double a=exp(x)+exp(-x);
    return 2.0/(a*a);
};
class Activation{
public:
    virtual void calc(double x[],double y[],int n);
    virtual void calc_diff(double x[],double y[],int n);
};
//一元激活函数
class SingleActivation:public Activation{
public:
    function<double(double)> f,f_;
    SingleActivation(){f=constant,f_=constant_diff;}
    SingleActivation(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){}
    virtual void calc(double x[],double y[],int n){
        for(int i=0;i<n;i++)y[i]=f(x[i]);
    }
    virtual void calc_diff(double x[],double y[],int n){
        for(int i=0;i<n;i++)y[i]=f_(x[i]);
    }
};

//层
class Layer{};

//Dense层
template <const int N>
class DenseLayer:public Layer{
public:
    Activation *f;
    static const int input_size=N,output_size=N;
    SmartArray<1,N> in_val,out_val,in_diff,out_diff,c;
    int threshold_flag=0;
    void reset_weight(double l=init_L,double r=init_R){
        for(int i=0;i<N;i++)c[i]=Rand(l,r);
    }
    DenseLayer(){f=new SingleActivation;}
    DenseLayer(function<double(double)> f1,function<double(double)> f2){
        threshold_flag=1;
        reset_weight();
        f=new SingleActivation(f1,f2);
    }
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        if(threshold_flag==1){
            for(int i=0;i<N;i++)in_val[i]+=c[i];
        }
        f->calc(in_val.data,out_val.data,N);
    }
    void backward_solve(){
        static double temp[N];
        f->calc_diff(in_val.data,temp,N);
        for(int i=0;i<N;i++)out_diff[i]=in_diff[i]*temp[i];
    }
    void update_w(double eta){
        if(threshold_flag==1){
            for(int i=0;i<N;i++)c[i]-=eta*out_diff[i];
        }
    }
};

//卷积层
template <const int H_in,const int W_in,const int H_c,const int W_c>
class ConvLayer:public Layer{
public:
    static const int H_out=H_in-H_c+1,W_out=W_in-W_c+1;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_c,W_c> w;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    double c;
    void reset_weight(double l=init_L,double r=init_R){
        w.reset_weight(l,r);
        c=Rand(l,r);
    }
    ConvLayer<H_in,W_in,H_c,W_c>(){reset_weight();}
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        out_val.clear();
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)+=w(r,c)*in_val(i+r,j+c);
            }
            out_val(i,j)+=c;
        }
    }
    void backward_solve(){
        for(int i=0;i<H_in;i++)for(int j=0;j<W_in;j++){
            out_diff[i*H_in+j]=0;
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                int ii=i+r-H_c+1,jj=j+c-W_c+1;
                if(ii>=0 && ii<H_out && jj>=0 && jj<W_out){
                    out_diff[i*H_in+j]+=in_diff(ii,jj)*w(H_c-r-1,W_c-c-1);
                }
            }
        }
    }
    void update_w(double eta){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            assert(!isnan(in_diff(i,j)));
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                w(r,c)-=eta*in_diff(i,j)*in_val(i+r,j+c);
            }
            c-=eta*in_diff(i,j);
        }
    }
};

/*最大值池化层*/
template <const int H_in,const int W_in,const int H_c,const int W_c>
class MaxPoolLayer:public Layer{
public:
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    void clear(){
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            out_val(i,j)=in_val(i*H_c,j*W_c);
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)=max(out_val(i,j),in_val(i*H_c+r,j*W_c+c));
            }
        }
    }
    void backward_solve(){
        out_diff.clear();
        int max_i=0,max_j=0;
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++)if(out_val(i,j)==in_val(i*H_c+r,j*W_c+c)){
                max_i=i*H_c+r;
                max_j=j*W_c+c;
            }
            assert(max_i>=i*H_c && max_i<i*H_c+H_c && max_j>=j*W_c && max_j<j*W_c+W_c);
            out_diff(max_i,max_j)=in_diff(i,j);
        }
    }
    void update_w(double eta){}
};

//正向传播
template<class LayerType1,class LayerType2>
void push_forward(LayerType1 &A,LayerType2 &B){
    assert(LayerType1::output_size==LayerType2::input_size);
    for(int i=0;i<LayerType1::output_size;i++)B.in_val[i]+=A.out_val[i];
}
//逆向传播
template<class LayerType1,class LayerType2>
void push_backward(LayerType1 &A,LayerType2 &B){
    assert(LayerType1::output_size==LayerType2::input_size);
    for(int i=0;i<LayerType1::output_size;i++)A.in_diff[i]+=B.out_diff[i];
}
//全连接
template <class LayerType1,class LayerType2>
ComplateEdge<LayerType1::output_size,LayerType2::input_size> full_connect(LayerType1 &A,LayerType2 &B){
    ComplateEdge<LayerType1::output_size,LayerType2::input_size> E;
    E.reset_weight(init_L,init_R);
    return E;
}
template<class LayerType1,class LayerType2>
void push_forward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E){
    for(int i=0;i<LayerType1::output_size;i++){
        for(int j=0;j<LayerType2::input_size;j++){
            B.in_val[j]+=A.out_val[i]*E(i,j);
        }
    }
}
template<class LayerType1,class LayerType2>
void push_backward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E,double eta){
    for(int j=0;j<LayerType2::input_size;j++){
        for(int i=0;i<LayerType1::output_size;i++){
            A.in_diff[i]+=B.out_diff[j]*E(i,j);
            E(i,j)-=eta*B.out_diff[j]*A.out_val[i];
        }
    }
}
