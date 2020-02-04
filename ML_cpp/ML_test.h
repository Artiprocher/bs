#include <bits/stdc++.h>
#include "ML_Model.h"
#define rep(i, a, b) for (int i = (a); i <= (int)(b); i++)
using namespace std;
typedef long long ll;

/*智能数组 用[]使用一维索引 用()使用二维索引*/
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

/*层*/
class Layer{
public:
    virtual void show(ostream &o) const {o << " Empty Layer" << endl;}
};

/*普通激活层*/
template <const int N>
class ActiveLayer:public Layer{
public:
    /*神经元个数*/
    static const int input_size=N,output_size=N;
    /*神经元的传入数据、传出数据、导数值、阈值*/
    SmartArray<1,N> in_val,out_val,diff_val,c;
    /*激活函数与激活函数导数*/
    function<double(double)> f=constant,f_=constant_diff;
    /*是否需要启用阈值 1:需要 0:不需要*/
    int threshold=0;
    void reset_weight(double l=0.5,double r=0.5){
        for(int i=0;i<N;i++)c[i]=Rand(l,r);
    }
    ActiveLayer<N>(){reset_weight();}
    ActiveLayer<N>(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){
        threshold=1;
    }
    void clear(){
        for(int i=0;i<N;i++)in_val[i]=0;
        for(int i=0;i<N;i++)diff_val[i]=0;
    }
    void forward_solve(){
        if(threshold==1){
            for(int j=0;j<N;j++)in_val[j]+=c[j];
        }
        for(int j=0;j<N;j++)out_val[j]=f(in_val[j]);
    }
    void update_w(double eta){
        if(threshold==1){
            for(int i=0;i<N;i++)c[i]-=eta*diff_val[i]*f_(in_val[i]);
        }
    }
};

/*卷积层*/
template <const int H_in,const int W_in,const int H_c,const int W_c>
class ConvLayer:public Layer{
public:
    static const int H_out=H_in-H_c+1,W_out=W_in-W_c+1;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    SmartArray<H_c,W_c> w;
    SmartArray<H_in,W_in> in_val;
    SmartArray<H_out,W_out> out_val,diff_val;
    double c;
    void reset_weight(double l=-0.5,double r=0.5){
        for(int i=0;i<H_c;i++){
            for(int j=0;j<W_c;j++){
                w(i,j)=Rand(l,r);
            }
        }
        c=Rand(l,r);
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
            out_val(i,j)+=c;
        }
    }
    void update_w(double eta){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            assert(!isnan(diff_val(i,j)));
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                w(r,c)-=eta*diff_val(i,j)*in_val(i+r,j+c);
            }
            c-=eta*diff_val(i,j);
        }
    }
};
/*最大值池化层*/
template <const int H_in,const int W_in,const int H_c,const int W_c>
class MaxPoolLayer:public Layer{
public:
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    static const int input_size=H_in*W_in,output_size=H_out*W_out;
    function<double(double)> f,f_;
    SmartArray<H_in,W_in> in_val;
    SmartArray<H_out,W_out> mid_val,out_val,diff_val;
    MaxPoolLayer<H_in,W_in,H_c,W_c>(){}
    MaxPoolLayer<H_in,W_in,H_c,W_c>(function<double(double)> f,function<double(double)> f_):f(f),f_(f_){}
    void clear(){
        in_val.clear();
        diff_val.clear();
    }
    void forward_solve(){
        for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
            out_val(i,j)=in_val(i*H_c,j*W_c);
            for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++){
                out_val(i,j)=max(out_val(i,j),in_val(i*H_c+r,j*W_c+c));
            }
            mid_val(i,j)=out_val(i,j);
            out_val(i,j)=f(out_val(i,j));
        }
    }
    void update_w(double eta){}
};

/*全连接层*/
template <class LayerType1,class LayerType2>
ComplateEdge<LayerType1::output_size,LayerType2::input_size> full_connect(LayerType1 &A,LayerType2 &B){
    ComplateEdge<LayerType1::output_size,LayerType2::input_size> E;
    E.reset_weight(-0.5,0.5);
    return E;
}

/*???Layer -> ActiveLayer*/
template <class LayerType,const int N,const int M>
void push_forward(LayerType &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            B.in_val[j]+=A.out_val[i]*E(i,j);
        }
    }
    B.forward_solve();
}
template <class LayerType,const int N,const int M>
void push_backward(LayerType &A,ActiveLayer<M> &B,ComplateEdge<N,M> &E,double eta){
    for(int j=0;j<M;j++){
        double temp=B.diff_val[j]*B.f_(B.in_val[j]);
        for(int i=0;i<N;i++){
            A.diff_val[i]+=temp*E(i,j);
            E(i,j)-=eta*temp*A.out_val[i];
        }
    }
}

/*???Layer -> ConvLayer*/
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_forward(LayerType &A,ConvLayer<H_in,W_in,H_c,W_c> &B){
    for(int i=0;i<H_in*W_in;i++)B.in_val[i]+=A.out_val[i];
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
    assert(A.output_size==B.input_size);
    for(int i=0;i<A.input_size;i++)B.in_val[i]=A.out_val[i];
    B.forward_solve();
}
template <class LayerType,const int H_in,const int W_in,const int H_c,const int W_c>
void push_backward(LayerType &A,MaxPoolLayer<H_in,W_in,H_c,W_c> &B,double eta){
    static const int H_out=H_in/H_c,W_out=W_in/W_c;
    int max_i=0,max_j=0;
    for(int i=0;i<H_out;i++)for(int j=0;j<W_out;j++){
        for(int r=0;r<H_c;r++)for(int c=0;c<W_c;c++)if(B.mid_val(i,j)==B.in_val(i*H_c+r,j*W_c+c)){
            max_i=i*H_c+r;
            max_j=j*W_c+c;
        }
        assert(max_i>=i*H_c && max_i<i*H_c+H_c && max_j>=j*W_c && max_j<j*W_c+W_c);
        A.diff_val(max_i,max_j)=B.diff_val(i,j)*B.f_(B.mid_val(i,j));
    }
}