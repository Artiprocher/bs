#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

template <const int N>
class AdditionLayer:public Layer{
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
        for(int i=0;i<N;i++)out_val[i]=A[i]+B[i];
    }
    void backward_solve(){
        for(int i=0;i<N;i++)A_diff[i]=in_diff[i];
        for(int i=0;i<N;i++)B_diff[i]=in_diff[i];
    }
    void update_w(double eta){}
};
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
    void update_w(double eta){}
};
template <class LayerType1,class LayerType2,class LayerType3>
void push_forward(LayerType1 &L1,LayerType2 &L2,LayerType3 &L3){
    assert(LayerType1::output_size==LayerType3::input_size);
    assert(LayerType2::output_size==LayerType3::input_size);
    for(int i=0;i<LayerType1::output_size;i++)L3.A[i]+=L1.out_val[i];
    for(int i=0;i<LayerType1::output_size;i++)L3.B[i]+=L2.out_val[i];
}
template <class LayerType1,class LayerType2,class LayerType3>
void push_backward(LayerType1 &L1,LayerType2 &L2,LayerType3 &L3){
    assert(LayerType1::output_size==LayerType3::input_size);
    assert(LayerType2::output_size==LayerType3::input_size);
    for(int i=0;i<LayerType1::output_size;i++)L1.in_diff[i]=L3.A_diff[i];
    for(int i=0;i<LayerType1::output_size;i++)L2.in_diff[i]=L3.B_diff[i];
}

namespace LSTM{
    const int n=1,m=1;
    double eta=0.0005;
    DenseLayer<n> H,X;
    DenseLayer<m> D[4],C0;
    ComplateEdge<n,m> H_D[4],X_D[4];
    MultiplicationLayer<m> M0,M1,C1;
    AdditionLayer<m> C2;
    auto loss=mse;
    void init(){
        D[0]=DenseLayer<m>(sigmoid,sigmoid_diff);
        D[1]=DenseLayer<m>(sigmoid,sigmoid_diff);
        D[2]=DenseLayer<m>(sigmoid,sigmoid_diff);
        D[3]=DenseLayer<m>(sigmoid,sigmoid_diff);
        for(int i=0;i<4;i++)H_D[i]=full_connect(H,D[i]);
        for(int i=0;i<4;i++)X_D[i]=full_connect(X,D[i]);
    }
    void recurrent(){
        push_forward(C2,C0);
        push_forward(M1,H);
    }
    void forward_disseminate(){
        X.forward_solve();
        H.forward_solve();
        for(int i=0;i<4;i++){
            push_forward(X,D[i],X_D[i]);
            push_forward(H,D[i],H_D[i]);
            D[i].forward_solve();
        }
        C0.forward_solve();push_forward(C0,D[0],C1);
        C1.forward_solve();push_forward(D[1],D[2],M0);
        M0.forward_solve();push_forward(C1,M0,C2);
        C2.forward_solve();push_forward(C2,D[3],M1);
        M1.forward_solve();
    }
    void backward_disseminate(){
        M1.backward_solve();push_backward(C2,D[3],M1);
        C2.backward_solve();push_backward(C1,M0,C2);
        C1.backward_solve();push_backward(C0,D[0],C1);
        M0.backward_solve();push_backward(D[1],D[2],M0);
        for(int i=0;i<4;i++){
            D[i].backward_solve();
            push_backward(H,D[i],H_D[i],eta);
            push_backward(X,D[i],X_D[i],eta);
        }
    }
    Vector predict(const Vector &x){
        for(int i=0;i<n;i++)X.in_val[i]=x[i];
        recurrent();
        forward_disseminate();
        static Vector y(m,0);
        for(int i=0;i<m;i++)y[i]=M1.out_val[i];
        return y;
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),M1.in_diff);
        backward_disseminate();
        for(int i=0;i<4;i++)D[i].update_w(eta);
    }
}
namespace RNN{
    const int n=1,m=1;
    double eta=0.0005;
    DenseLayer<n> H,X,Y;
    auto H_Y=full_connect(H,Y);
    auto X_Y=full_connect(X,Y);
    auto loss=mse;
    void init(){
        ;
    }
    void recurrent(){
        H.clear();
        push_forward(Y,H);
    }
    void forward_disseminate(){
        X.forward_solve();
        H.forward_solve();
        push_forward(H,Y,H_Y);
        push_forward(X,Y,X_Y);
        Y.forward_solve();
    }
    void backward_disseminate(){
        Y.backward_solve();
        push_backward(H,Y,H_Y,eta);
        push_backward(X,Y,X_Y,eta);
    }
    Vector predict(const Vector &x){
        X.clear();
        Y.clear();
        for(int i=0;i<n;i++)X.in_val[i]=x[i];
        recurrent();
        forward_disseminate();
        static Vector y(m,0);
        for(int i=0;i<m;i++)y[i]=Y.out_val[i];
        return y;
    }
    void train(const Vector &x,const Vector &y){
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),Y.in_diff);
        backward_disseminate();
        X.update_w(eta);
        Y.update_w(eta);
    }
}

double f(int x){
    return (double)(x%2==0?1:-1);
}
void demo(){
    RNN::init();
    int epoch=100000;
    for(int i=1;i<=epoch;i++){
        RNN::train({f(i-1)},{f(i)});
    }
    for(int i=epoch+1;i<=epoch+10;i++){
        cout<<f(i)<<" "<<RNN::predict({f(i-1)})<<endl;
    }
}

int main() {
    demo();
    return 0;
}
/*

*///
