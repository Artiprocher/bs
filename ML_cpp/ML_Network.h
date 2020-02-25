#include <bits/stdc++.h>
#include "ML_Vector.h"
#include "ML_Rand.h"
#include "ML_Optimazer.h"
using namespace std;

const double init_L=-0.5,init_R=0.5;

//全连接边
template <const int N,const int M>
class ComplateEdge{
public:
    SmartArray<N,M> w,dw;
    ComplateEdge<N,M>(){w.reset_weight(init_L,init_R);}
    void reset_weight(double l,double r){w.reset_weight(l,r);}
    double& operator () (int x,int y){return w(x,y);}
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N*M;i++)PL.add_parameter(w[i],dw[i]);
    }
};

/*损失函数(导数)*/
function<Vector(Vector,Vector)> mse=[](Vector y,Vector y_){
    each_index(i,y)y[i]=y_[i]-y[i];
    return y;
};
function<Vector(Vector,Vector)> mae=[](Vector y,Vector y_){
    each_index(i,y)y[i]=(y[i]<y_[i])?1.0:-1.0;
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
    void get_parameters(ParameterList &PL){
        if(threshold_flag==1){
            for(int i=0;i<N;i++)PL.add_parameter(c[i],out_diff[i]);
        }
    }
};

//卷积层
template <const int H_in,const int W_in,const int H_c,const int W_c>
class ConvLayer:public Layer{
public:
    static const int H_out=H_in-H_c+1,W_out=W_in-W_c+1;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_c,W_c> w,dw;
    SmartArray<H_in,W_in> in_val,out_diff;
    SmartArray<H_out,W_out> out_val,in_diff;
    double c,dc;
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
        dw.clear();
        dc=0;
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            assert(!isnan(in_diff(i,j)));
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                dw(r,c)+=in_diff(i,j)*in_val(i+r,j+c);
            }
            dc+=in_diff(i,j);
        }
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<H_c;i++)for(int j=0;j<W_c;j++)PL.add_parameter(w(i,j),dw(i,j));
        PL.add_parameter(c,dc);
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
    void get_parameters(ParameterList &PL){}
};

//并行层
template<class LayerType,const int N>
class Parallel:public Layer{
public:
    static const int input_size=LayerType::input_size*N;
    static const int output_size=LayerType::output_size*N;
    SmartArray<1,input_size> in_val,out_diff;
    SmartArray<1,output_size> in_diff,out_val;
    LayerType L[N];
    LayerType& operator [] (int x){return L[x];}
    void clear(){
        for(int i=0;i<N;i++)L[i].clear();
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                L[i].in_val[j]=in_val[tot++];
            }
        }
        for(int i=0;i<N;i++)L[i].forward_solve();
        tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                out_val[tot++]=L[i].out_val[j];
            }
        }
    }
    void backward_solve(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                L[i].in_diff[j]=in_diff[tot++];
            }
        }
        for(int i=0;i<N;i++)L[i].backward_solve();
        tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                out_diff[tot++]=L[i].out_diff[j];
            }
        }
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N;i++)L[i].get_parameters(PL);
    }
};
template<class LayerType,const int N>
class ExpandParallel:public Layer{
public:
    static const int input_size=LayerType::input_size;
    static const int output_size=LayerType::output_size*N;
    SmartArray<1,input_size> in_val,out_diff;
    SmartArray<1,output_size> in_diff,out_val;
    LayerType L[N];
    LayerType& operator [] (int x){return L[x];}
    void clear(){
        for(int i=0;i<N;i++)L[i].clear();
        in_val.clear();
        in_diff.clear();
    }
    void forward_solve(){
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                L[i].in_val[j]=in_val[j];
            }
        }
        for(int i=0;i<N;i++)L[i].forward_solve();
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                out_val[tot++]=L[i].out_val[j];
            }
        }
    }
    void backward_solve(){
        int tot=0;
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::output_size;j++){
                L[i].in_diff[j]=in_diff[tot++];
            }
        }
        for(int i=0;i<N;i++)L[i].backward_solve();
        out_diff.clear();
        for(int i=0;i<N;i++){
            for(int j=0;j<LayerType::input_size;j++){
                out_diff[j]=L[i].out_diff[j];
            }
        }
    }
    void get_parameters(ParameterList &PL){
        for(int i=0;i<N;i++)L[i].get_parameters(PL);
    }
};

//Dropout层
template<const int N>
class DropoutLayer:public Layer{
public:
    static const int input_size=N;
    static const int output_size=N;
    double p=0.5,k=2.0;
    SmartArray<1,N> in_val,out_diff;
    SmartArray<1,N> in_diff,out_val;
    bool drop[N];
    void clear(){
        in_val.clear();
        out_val.clear();
        for(int i=0;i<N;i++)drop[i]=0;
    }
    void forward_solve(){
        for(int i=0;i<N;i++)if(Rand(0,1)<p)drop[i]=1;
        for(int i=0;i<N;i++)out_val[i]=drop[i]?0:in_val[i]*k;
    }
    void backward_solve(){
        for(int i=0;i<N;i++)out_diff[i]=drop[i]?0:in_diff[i]*k;
    }
    void get_parameters(ParameterList &PL){}
};

//乘法层
template <const int N>
class MultiplicationLayer:public Layer{
public:
    static const int input_size=N,output_size=N;
    SmartArray<1,N> A,B,out_val;
    SmartArray<1,N> in_diff,A_diff,B_diff;
    void clear(){
        A.clear();
        B.clear();
        in_diff.clear();
    }
    void forward_solve(){
        for(int i=0;i<N;i++)out_val[i]=A[i]*B[i];
    }
    void backward_solve(){
        for(int i=0;i<N;i++)A_diff[i]=in_diff[i]*B[i];
        for(int i=0;i<N;i++)B_diff[i]=in_diff[i]*A[i];
    }
    void get_parameters(ParameterList &PL){}
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
template<class LayerType1,class LayerType2>
void push_forward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E){
    for(int i=0;i<LayerType1::output_size;i++){
        for(int j=0;j<LayerType2::input_size;j++){
            B.in_val[j]+=A.out_val[i]*E(i,j);
        }
    }
}
template<class LayerType1,class LayerType2>
void push_backward(LayerType1 &A,LayerType2 &B,ComplateEdge<LayerType1::output_size,LayerType2::input_size> &E){
    for(int j=0;j<LayerType2::input_size;j++){
        for(int i=0;i<LayerType1::output_size;i++){
            A.in_diff[i]+=B.out_diff[j]*E(i,j);
            E.dw(i,j)=B.out_diff[j]*A.out_val[i];
        }
    }
}
//乘法层传播
template <class LayerType1,class LayerType2,const int N>
void push_forward(LayerType1 &L1,LayerType2 &L2,MultiplicationLayer<N> &L3){
    assert(LayerType1::output_size==N && LayerType2::output_size==N);
    for(int i=0;i<N;i++)L3.A[i]+=L1.out_val[i];
    for(int i=0;i<N;i++)L3.B[i]+=L2.out_val[i];
}
template <class LayerType1,class LayerType2,const int N>
void push_backward(LayerType1 &L1,LayerType2 &L2,MultiplicationLayer<N> &L3){
    assert(LayerType1::output_size==N && LayerType2::output_size==N);
    for(int i=0;i<N;i++)L1.in_diff[i]+=L3.A_diff[i];
    for(int i=0;i<N;i++)L2.in_diff[i]+=L3.B_diff[i];
}
