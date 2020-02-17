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

namespace SimpleRNN{
    const int n=5,m=32,k=1;
    const int timestep=20,epoch=10;
    const double eta=0.05,eta_=eta*timestep;
    auto loss=mse;
    class cell{
    public:
        DenseLayer<n> X;
        DenseLayer<m> H;
        DenseLayer<k> Y;
        ComplateEdge<n,m> X_H=full_connect(X,H);
        ComplateEdge<m,k> H_Y=full_connect(H,Y);
        ComplateEdge<m,m> H_H=full_connect(H,H);
        cell(){
            H=DenseLayer<m>(sigmoid,sigmoid_diff);
            Y=DenseLayer<k>(constant,constant_diff);
        }
        void clear(){
            X.clear();
            H.clear();
            Y.clear();
        }
        void show(){
            cout<<"  X_H"<<endl;
            X_H.show();
            cout<<"  H_Y"<<endl;
            H_Y.show();
        }
        void forward_disseminate(){
            X.forward_solve();
            push_forward(X,H,X_H);
            H.forward_solve();
            push_forward(H,Y,H_Y);
            Y.forward_solve();
        }
        void backward_disseminate(){
            Y.backward_solve();
            push_backward(H,Y,H_Y,eta_);
            H.backward_solve();
            push_backward(X,H,X_H,eta_);
        }
        void update_w(){
            X.update_w(eta_);
            H.update_w(eta_);
            Y.update_w(eta_);
        }
    }c[timestep];
    void init(){}
    vector<Vector> predict(const vector<Vector> &vx){
        assert(vx.size()==timestep);
        //auto last=c[1].H.out_val;
        for(int i=0;i<timestep;i++)c[i].clear();
        //c[0].H.in_val=last;
        for(int i=0;i<timestep;i++)Vector2Array(vx[i],c[i].X.in_val);
        for(int i=0;i<timestep;i++){
            if(i>0)push_forward(c[i-1].H,c[i].H,c[i-1].H_H);
            c[i].forward_disseminate();
        }
        vector<Vector> vy;
        for(int i=0;i<timestep;i++)vy.emplace_back(Array2Vector(c[i].Y.out_val));
        return vy;
    }
    void train(const vector<Vector> &vx,const vector<Vector> &vy){
        assert(vx.size()==timestep && vy.size()==timestep);
        for(int it=0;it<10;it++){
            vector<Vector> y_=predict(vx);
            //沿时间轴逆向传播
            for(int i=0;i<timestep;i++)Vector2Array(loss(vy[i],y_[i]),c[i].Y.in_diff);
            for(int i=timestep-1;i>=0;i--){
                c[i].backward_disseminate();
                if(i>0)push_backward(c[i-1].H,c[i].H,c[i-1].H_H,eta_);
            }
            //更新权重
            for(int i=0;i<timestep;i++)c[i].update_w();
            for(int i=1;i<timestep;i++){
                c[0].H.c+=c[i].H.c;c[0].Y.c+=c[i].Y.c;
                c[0].X_H+=c[i].X_H;c[0].H_Y+=c[i].H_Y;
                if(i!=timestep-1)c[0].H_H+=c[i].H_H;
            }
            static const double temp=1.0/timestep;
            c[0].H.c*=temp;
            c[0].Y.c*=temp;
            c[0].X_H*=temp;
            c[0].H_Y*=temp;
            c[0].H_H*=1.0/(timestep-1);
            for(int i=1;i<timestep;i++)c[i]=c[0];
        }
    }
    template <class fun>
    void train_with_sequence(fun &f,int L,int R){
        vector<Vector> vx(timestep,Vector(n,0)),vy(timestep,Vector(k,0));
        for(int l=L+n,r=l+timestep-1;r<=R;l++,r++){
            for(int i=l;i<=r;i++){
                for(int j=0;j<n;j++)vx[i-l][j]=f(i-n+j);
                vy[i-l]=(Vector){f(i)};
            }
            train(vx,vy);
            if((r-L)%((R-L)/100)==0)cout<<(r-L)*100.0/(R-L)<<"%"<<endl;
        }
    }
    template <class fun>
    Vector predict(fun &f,int p){
        vector<Vector> vx(timestep,Vector(n,0));
        int l=p-timestep+1,r=p;
        for(int i=l;i<=r;i++){
            for(int j=0;j<n;j++)vx[i-l][j]=f(i-n+j);
        }
        return predict(vx).back();
    }
}

namespace LSTM{
    const int n=5,m=32,k=1;
    const int timestep=20,epoch=20;
    const double eta=0.05,eta_=eta*timestep;
    auto loss=mse;
    class cell{
    public:
        DenseLayer<n> X;
        DenseLayer<m> H0,D[4],C0,C2,C3;
        DenseLayer<k> Y;
        MultiplicationLayer<m> C1,H1,M;
        ComplateEdge<n,m> X_D[4];
        ComplateEdge<m,m> H_D[4];
        ComplateEdge<m,k> H1_Y;
        cell(){
            D[0]=DenseLayer<m>(sigmoid,sigmoid_diff);
            D[1]=DenseLayer<m>(sigmoid,sigmoid_diff);
            D[2]=DenseLayer<m>(Tanh,Tanh_diff);
            D[3]=DenseLayer<m>(sigmoid,sigmoid_diff);
            for(int i=0;i<4;i++)H_D[i]=full_connect(H0,D[i]);
            for(int i=0;i<4;i++)X_D[i]=full_connect(X,D[i]);
            H1_Y=full_connect(H1,Y);
            Y=DenseLayer<k>(constant,constant_diff);
            C3=DenseLayer<m>(Tanh,Tanh_diff);
        }
        void clear(){
            X.clear();H0.clear();
            D[0].clear();D[1].clear();D[2].clear();D[3].clear();
            M.clear();
            C0.clear();C1.clear();C2.clear();C3.clear();
            H1.clear();Y.clear();
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
            Y.backward_solve();push_backward(H1,Y,H1_Y,eta_);
            H1.backward_solve();push_backward(C3,D[3],H1);
            C3.backward_solve();push_backward(C2,C3);
            C2.backward_solve();push_backward(C1,C2);push_backward(M,C2);
            C1.backward_solve();push_backward(C0,D[0],C1);
            M.backward_solve();push_backward(D[1],D[2],M);
            for(int i=0;i<4;i++){
                D[i].backward_solve();
                push_backward(H0,D[i],H_D[i],eta_);
                push_backward(X,D[i],X_D[i],eta_);
            }
        }
        void update_w(){
            for(int i=0;i<4;i++)D[i].update_w(eta_);
            C3.update_w(eta_);
            Y.update_w(eta_);
        }
    }c[timestep];
    void init(){}
    vector<Vector> predict(const vector<Vector> &vx){
        assert(vx[0].size()==n);
        assert(vx.size()==timestep);
        for(int i=0;i<timestep;i++)c[i].clear();
        for(int i=0;i<timestep;i++)Vector2Array(vx[i],c[i].X.in_val);
        for(int i=0;i<timestep;i++){
            if(i>0){
                push_forward(c[i-1].H1,c[i].H0);
                push_forward(c[i-1].C2,c[i].C0);
            }
            c[i].forward_disseminate();
        }
        vector<Vector> vy;
        for(int i=0;i<timestep;i++)vy.emplace_back(Array2Vector(c[i].Y.out_val));
        return vy;
    }
    void train(const vector<Vector> &vx,const vector<Vector> &vy){
        assert(vx.size()==timestep && vy.size()==timestep);
        assert(vx[0].size()==n && vy[0].size()==k);
        for(int it=0;it<epoch;it++){
            vector<Vector> y_=predict(vx);
            //沿时间轴逆向传播
            for(int i=0;i<timestep;i++)Vector2Array(loss(vy[i],y_[i]),c[i].Y.in_diff);
            for(int i=timestep-1;i>=0;i--){
                c[i].backward_disseminate();
                if(i>0){
                    push_backward(c[i-1].H1,c[i].H0);
                    push_backward(c[i-1].C2,c[i].C0);
                }
            }
            //更新权重
            for(int i=0;i<timestep;i++)c[i].update_w();
            for(int i=1;i<timestep;i++){
                for(int j=0;j<4;j++){
                    c[0].X_D[j]+=c[i].X_D[j];
                    c[0].H_D[j]+=c[i].H_D[j];
                    c[0].D[j].c+=c[i].D[j].c;
                }
                c[0].C3.c+=c[i].C3.c;
                c[0].H1_Y+=c[i].H1_Y;
                c[0].Y.c+=c[i].Y.c;
            }
            static const double temp=1.0/timestep;
            for(int j=0;j<4;j++){
                c[0].X_D[j]*=temp;
                c[0].H_D[j]*=temp;
                c[0].D[j].c*=temp;
            }
            c[0].C3.c*=temp;
            c[0].H1_Y*=temp;
            c[0].Y.c*=temp;
            for(int i=1;i<timestep;i++)c[i]=c[0];
        }
    }
    template <class fun>
    void train_with_sequence(fun &f,int L,int R){
        vector<Vector> vx(timestep,Vector(n,0)),vy(timestep,Vector(k,0));
        for(int l=L+n,r=l+timestep-1;r<=R;l++,r++){
            for(int i=l;i<=r;i++){
                for(int j=0;j<n;j++)vx[i-l][j]=f(i-n+j);
                vy[i-l]=(Vector){f(i)};
            }
            train(vx,vy);
            if((r-L)%((R-L)/100)==0)cout<<(r-L)*100.0/(R-L)<<"%"<<endl;
        }
    }
    template <class fun>
    Vector predict(fun &f,int p){
        vector<Vector> vx(timestep,Vector(n,0));
        int l=p-timestep+1,r=p;
        for(int i=l;i<=r;i++){
            for(int j=0;j<n;j++)vx[i-l][j]=f(i-n+j);
        }
        return predict(vx).back();
    }
}

#define NET LSTM
CSV_Reader csv_reader;
DataSet data;
double f(int x){
    //return (double)(x%7);
    return data(x,0);
}
void demo(){
    //read data
    csv_reader.open("weather/weather.csv");
    int all=csv_reader.size()[0];
    csv_reader.export_number_data(0,all-1,2,2,data);
    data.normalization(0);
    //train
    int train=200000,test=9000;
    assert(train+test<all);
    NET::init();
    NET::train_with_sequence(f,0,train);
    //test
    double loss=0,baseline_loss=0;
    for(int i=train+1;i<=train+test;i++){
        double y=f(i),y_=NET::predict(f,i)[0];
        loss+=abs(y-y_);
        baseline_loss+=abs(y-f(i-1));
    }
    cout<<"loss="<<loss<<endl;
    cout<<"baseline_loss="<<baseline_loss<<endl;
}

int main() {
    demo();
    return 0;
}
/*

*///
