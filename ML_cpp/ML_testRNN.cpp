#include "ML_Model.h"
#define rep(i,a,b) for(int i=(int)a;i<=(int)b;i++)
typedef long long ll;

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

namespace LSTM{
    const int n=1,m=3,k=1;
    double eta=0.05;
    DenseLayer<n> X;
    DenseLayer<m> H0,D[4],C0,C2,C3;
    DenseLayer<k> Y;
    MultiplicationLayer<m> C1,H1,M;
    ComplateEdge<n,m> X_D[4];
    ComplateEdge<m,m> H_D[4];
    ComplateEdge<m,k> H1_Y;
    auto loss=mse;
    void init(){
        D[0]=DenseLayer<m>(sigmoid,sigmoid_diff);
        D[1]=DenseLayer<m>(sigmoid,sigmoid_diff);
        D[2]=DenseLayer<m>(Tanh,Tanh_diff);
        D[3]=DenseLayer<m>(sigmoid,sigmoid_diff);
        for(int i=0;i<4;i++)H_D[i]=full_connect(H0,D[i]);
        for(int i=0;i<4;i++)X_D[i]=full_connect(X,D[i]);
        H1_Y=full_connect(H1,Y);
        Y.threshold_flag=1;Y.reset_weight();
        C3=DenseLayer<m>(Tanh,Tanh_diff);
    }
    void recurrent(){
        C0.clear();
        push_forward(C2,C0);
        H0.clear();
        push_forward(H1,H0);
    }
    void forward_disseminate(){
        X.forward_solve();
        H0.forward_solve();
        for(int i=0;i<4;i++){
            push_forward(X,D[i],X_D[i]);
            push_forward(H0,D[i],H_D[i]);
            D[i].forward_solve();
        }
        C0.forward_solve();
        push_forward(C0,D[0],C1);C1.forward_solve();
        push_forward(D[1],D[2],M);M.forward_solve();
        push_forward(C1,C2);push_forward(M,C2);C2.forward_solve();
        push_forward(C2,C3);C3.forward_solve();
        push_forward(C3,D[3],H1);H1.forward_solve();
        push_forward(H1,Y,H1_Y);Y.forward_solve();
    }
    void backward_disseminate(){
        Y.backward_solve();push_backward(H1,Y,H1_Y,eta);
        H1.backward_solve();push_backward(C3,D[3],H1);
        C3.backward_solve();push_backward(C2,C3);
        C2.backward_solve();push_backward(C1,C2);push_backward(M,C2);
        C1.backward_solve();push_backward(C0,D[0],C1);
        M.backward_solve();push_backward(D[1],D[2],M);
        for(int i=0;i<4;i++){
            D[i].backward_solve();
            push_backward(H0,D[i],H_D[i],eta);
            push_backward(X,D[i],X_D[i],eta);
        }
    }
    Vector predict(const Vector &x){
        assert(x.size()==n);
        for(int i=0;i<n;i++)X.in_val[i]=x[i];
        recurrent();
        C0.in_diff.clear();X.in_diff.clear();H0.in_diff.clear();
        for(int i=0;i<4;i++)D[i].clear();
        M.clear();C1.clear();C2.clear();C3.clear();H1.clear();Y.clear();
        forward_disseminate();
        static Vector y(k,0);
        for(int i=0;i<k;i++)y[i]=Y.out_val[i];
        return y;
    }
    void train(const Vector &x,const Vector &y){
        assert(x.size()==n && y.size()==k);
        Vector y_=predict(x);
        Vector2Array(loss(y,y_),Y.in_diff);
        backward_disseminate();
        for(int i=0;i<4;i++)D[i].update_w(eta);
    }
}

double f(int x){
    return (double)(x%7);
}
#define NET LSTM
void demo1(){
    NET::init();
    cout<<fixed<<setprecision(2);
    int epoch=100000;
    for(int i=10;i<epoch;i++){
        NET::train({f(i-1)},{f(i)});
    }
    for(int i=epoch;i<epoch+15;i++){
        cout<<f(i)<<"  "<<NET::predict({f(i-1)})<<endl;
    }
}
CSV_Reader csv_reader;
DataSet data;
double real_val(double x){
    double m=10.3808630941,s=8.3260192097;
    return x*s+m;
}
void demo(){
    csv_reader.open("weather/weather.csv");
    //csv_reader.describe();
    int all=csv_reader.size()[0];
    csv_reader.export_number_data(0,all-1,2,2,data);
    cout<<"min:"<<data.min(0)<<" max:"<<data.max(0)<<" mean:"<<data.mean(0)<<" std:"<<data.std_dev(0)<<endl;
    data.normalization(0);
    int train=200000;
    int T=144;
    NET::init();
    Vector x(T,0),y(1,0);
    for(int i=T;i<=train;i++){
        for(int j=0;j<T;j++)x[j]=data(i-T+j,0);
        y[0]=data(i,0);
        NET::train(x,y);
        if(i%(train/100)==0)cout<<(i*100.0/train)<<"%"<<endl;
    }
    double loss=0,fake_loss=0;
    for(int i=train+1;i<all;i++){
        for(int j=0;j<T;j++)x[j]=data(i-T+j,0);
        double a=real_val(data(i,0)),b=real_val(NET::predict(x)[0]);
        loss+=abs(a-b);
        fake_loss+=abs(a-real_val(x[T-1]));
        //cout<<a<<"    "<<b<<endl;
    }
    cout<<"loss="<<loss<<endl;
    cout<<"fake_loss="<<fake_loss<<endl;
}

int main() {
    demo1();
    return 0;
}
/*

*///
